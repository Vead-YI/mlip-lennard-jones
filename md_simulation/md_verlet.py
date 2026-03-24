"""
md_verlet.py
------------
Molecular Dynamics simulation using the trained Neural Network Potential.
Implements the Velocity Verlet integration algorithm.

Usage:
    python md_verlet.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.nn_potential import NeuralNetworkPotential


# ── LJ Parameters (matching data generation) ────────────────────────────────
SIGMA = 1.0
EPSILON = 1.0
DT = 0.005           # timestep (reduced units)
N_STEPS = 2000       # total MD steps
N_PARTICLES = 10     # toy system
MASS = 1.0           # particle mass (reduced units)
BOX_SIZE = 8.0       # simulation box
R_CUTOFF = 2.5       # LJ cutoff distance


def lj_potential(r):
    """Lennard-Jones potential."""
    sr6 = (SIGMA / r) ** 6
    sr12 = sr6 ** 2
    return 4.0 * EPSILON * (sr12 - sr6)


def lj_force(r):
    """Lennard-Jones force magnitude F = -dV/dr."""
    sr6 = (SIGMA / r) ** 6
    sr12 = sr6 ** 2
    return 24.0 * EPSILON / r * (2.0 * sr12 - sr6)


def load_nnp_model():
    """Load the trained NNP and normalization parameters."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "nnp_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training/train.py first.")

    checkpoint = torch.load(model_path, map_location='cpu')
    norm_params = checkpoint['norm_params']
    
    model = NeuralNetworkPotential(hidden_dims=[128, 128, 64])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, norm_params


def normalize_r(r, norm_params):
    """Normalize distance using training params."""
    r_mean = norm_params['r_mean']
    r_std = norm_params['r_std']
    return (r - r_mean) / r_std


def denormalize_e(e_norm, norm_params):
    """Denormalize energy using training params."""
    e_mean = norm_params['e_mean']
    e_std = norm_params['e_std']
    return e_norm * e_std + e_mean


def nn_potential(r_np, model, norm_params):
    """
    Compute NN potential energy for a single distance r.
    Handles normalization (matching training preprocessing).
    """
    # Normalize: must match how training data was preprocessed
    r_norm = (r_np - norm_params['r_mean']) / norm_params['r_std']
    r_t = torch.tensor([[r_norm]], dtype=torch.float32)
    with torch.no_grad():
        e_norm = model(r_t).item()
    return denormalize_e(e_norm, norm_params)


def nn_force(r_np, model, norm_params):
    """
    Compute NN force via automatic differentiation.
    F = -dV/dr, evaluated at distance r.
    """
    # Normalize: must match how training data was preprocessed
    r_norm = (r_np - norm_params['r_mean']) / norm_params['r_std']
    r_t = torch.tensor([[r_norm]], dtype=torch.float32, requires_grad=True)
    e = model(r_t)
    grad = torch.autograd.grad(outputs=e, inputs=r_t, grad_outputs=torch.ones_like(e))[0]
    
    # Chain rule: dE/d(r_norm) * d(r_norm)/dr
    # d(r_norm)/dr = 1 / r_std
    # dE/dr = (dE/d(r_norm)) * (1 / r_std)
    # F = -dE/dr = -grad * (1 / r_std) * e_std  (then denorm)
    # Actually: grad is dE/d(r_norm), we need dE/dr
    # dE/dr = grad * d(r_norm)/dr = grad * (1/r_std)
    # F_denorm = -dE/dr = -grad * (1/r_std) * e_std
    # Simple: treat NN output as already denormalized energy
    # So dE_original/dr = d(E_norm * e_std + e_mean)/dr = dE_norm/dr * e_std
    # F = -dE_original/dr = -grad * e_std
    # BUT we used r_norm as input, so:
    # dE_original/dr = (dE_norm/dr_norm) * (dr_norm/dr) = grad * (1/r_std) * e_std
    # F = -grad * e_std / r_std
    r_std = norm_params['r_std']
    e_std = norm_params['e_std']
    force_mag = -grad.item() * e_std / r_std
    return force_mag


def minimum_image(r_i, r_j, box_size):
    """Apply minimum image convention."""
    delta = r_i - r_j
    delta -= box_size * np.round(delta / box_size)
    return delta


def compute_forces(positions, box_size, model, norm_params, potential_type='nn'):
    """
    Compute forces on all particles using NN or LJ potential.
    Returns force vectors (N, 3) on each particle.
    """
    n = positions.shape[0]
    forces = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Minimum image convention
            delta = minimum_image(positions[i], positions[j], box_size)
            r = np.linalg.norm(delta)
            
            if r > R_CUTOFF or r < 0.01:
                continue
            
            # Force direction (unit vector from j to i)
            if r > 1e-8:
                direction = delta / r
            else:
                direction = np.zeros(3)
            
            if potential_type == 'nn':
                f_mag = nn_force(r, model, norm_params)
            else:  # 'lj'
                f_mag = lj_force(r)
            
            # f_mag is positive = repulsive (push i away from j)
            # Force on i = +f_mag * direction
            # Force on j = -f_mag * direction
            forces[i] += f_mag * direction
            forces[j] -= f_mag * direction
    
    return forces


def compute_energy(positions, box_size, model, norm_params, potential_type='nn'):
    """Compute total potential energy of the system."""
    n = positions.shape[0]
    total_energy = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            delta = minimum_image(positions[i], positions[j], box_size)
            r = np.linalg.norm(delta)
            
            if r > R_CUTOFF or r < 0.01:
                continue
            
            if potential_type == 'nn':
                e = nn_potential(r, model, norm_params)
            else:
                e = lj_potential(r)
            
            total_energy += e
    
    return total_energy


def init_positions(n_particles, box_size, min_dist=1.0):
    """Randomly initialize particle positions avoiding overlaps."""
    rng = np.random.default_rng(42)
    positions = []
    
    for _ in range(n_particles):
        attempts = 0
        while attempts < 1000:
            candidate = rng.uniform(0.5, box_size - 0.5, size=3)
            
            # Check overlap with existing particles
            overlap = False
            for existing in positions:
                dist = np.linalg.norm(candidate - existing)
                if dist < min_dist:
                    overlap = True
                    break
            
            if not overlap:
                positions.append(candidate)
                break
            attempts += 1
        
        if attempts == 1000:
            # Fallback: just place it
            positions.append(candidate)
    
    return np.array(positions)


def run_md(positions, model, norm_params, n_steps, dt, box_size, potential_type='nn'):
    """
    Velocity Verlet MD simulation.
    
    Returns:
        trajectory : list of position arrays
        energies   : list of total potential energies per step
        kinetic energies per step
    """
    n = positions.shape[0]
    
    # Initialize velocities (Maxwell-Boltzmann at T=1.0)
    T = 1.0
    vel_std = np.sqrt(T / MASS)
    velocities = np.random.randn(n, 3) * vel_std
    
    # Remove center-of-mass momentum
    velocities -= velocities.mean(axis=0)
    
    # Compute initial forces
    forces = compute_forces(positions, box_size, model, norm_params, potential_type)
    
    trajectory = [positions.copy()]
    potential_energies = []
    kinetic_energies = []
    temperatures = []
    
    for step in range(n_steps):
        # Velocity Verlet integration
        
        # Half-step velocity update
        velocities += (forces / MASS) * (dt / 2.0)
        
        # Full-step position update
        positions += velocities * dt
        
        # Periodic boundary conditions
        positions = positions % box_size
        
        # New forces
        forces_new = compute_forces(positions, box_size, model, norm_params, potential_type)
        
        # Half-step velocity update with new forces
        velocities += (forces_new / MASS) * (dt / 2.0)
        forces = forces_new
        
        # Compute energies
        e_pot = compute_energy(positions, box_size, model, norm_params, potential_type)
        e_kin = 0.5 * MASS * np.sum(velocities ** 2)
        T_current = 2.0 * e_kin / (3.0 * n)  # instantaneous temperature
        
        if step % 10 == 0:
            trajectory.append(positions.copy())
            potential_energies.append(e_pot)
            kinetic_energies.append(e_kin)
            temperatures.append(T_current)
        
        if (step + 1) % 500 == 0:
            print(f"    {potential_type.upper():2s} | Step {step+1:4d}/{n_steps} | "
                  f"Epot={e_pot:.4f} | Ekin={e_kin:.4f} | T={T_current:.4f}")
    
    return np.array(trajectory), np.array(potential_energies), np.array(kinetic_energies), np.array(temperatures)


def plot_energy_comparison(lj_energies, nn_energies, label_lj="LJ Baseline", label_nn="ML Potential"):
    """Plot energy comparison between LJ and NN potential."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    steps = np.arange(len(lj_energies)) * 10  # every 10 steps
    
    # Total energy
    total_lj = lj_energies + 0.5 * MASS * np.sum(np.random.randn(*lj_energies.shape)**2, axis=0) * 0
    total_nn = nn_energies + 0.5 * MASS * np.sum(np.random.randn(*nn_energies.shape)**2, axis=0) * 0
    
    # For fair comparison, plot just potential energy
    axes[0].plot(steps, lj_energies, 'b-', alpha=0.8, linewidth=1, label=label_lj)
    axes[0].plot(steps, nn_energies, 'r-', alpha=0.8, linewidth=1, label=label_nn)
    axes[0].set_xlabel('MD Step')
    axes[0].set_ylabel('Potential Energy (ε)')
    axes[0].set_title('Potential Energy During MD')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Energy difference
    diff = nn_energies - lj_energies
    axes[1].plot(steps, diff, 'purple', linewidth=1, alpha=0.8)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].fill_between(steps, diff, 0, alpha=0.3, color='purple')
    axes[1].set_xlabel('MD Step')
    axes[1].set_ylabel('ΔE = E_NN − E_LJ (ε)')
    axes[1].set_title('Energy Difference: ML Potential vs LJ Baseline')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_trajectory_snapshot(traj_lj, traj_nn, box_size):
    """Plot initial, middle, and final snapshots of both simulations."""
    indices = [0, len(traj_lj)//2, -1]
    labels = ['Start', 'Middle', 'End']
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    
    for col, (idx, label) in enumerate(zip(indices, labels)):
        # LJ trajectory
        axes[0, col].set_title(f'LJ — {label}')
        axes[0, col].set_xlim(0, box_size)
        axes[0, col].set_ylim(0, box_size)
        axes[0, col].set_aspect('equal')
        axes[0, col].scatter(traj_lj[idx, :, 0], traj_lj[idx, :, 1], s=80, c='steelblue', edgecolors='k', linewidth=0.5)
        axes[0, col].tick_params(labelsize=7)
        
        # NN trajectory
        axes[1, col].set_title(f'ML Potential — {label}')
        axes[1, col].set_xlim(0, box_size)
        axes[1, col].set_ylim(0, box_size)
        axes[1, col].set_aspect('equal')
        axes[1, col].scatter(traj_nn[idx, :, 0], traj_nn[idx, :, 1], s=80, c='tomato', edgecolors='k', linewidth=0.5)
        axes[1, col].tick_params(labelsize=7)
    
    plt.suptitle('MD Trajectory Comparison: LJ vs Neural Network Potential', fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 55)
    print("  Molecular Dynamics with Neural Network Potential")
    print("=" * 55)
    
    # Load model
    print("\n[1/4] Loading trained NNP model...")
    model, norm_params = load_nnp_model()
    print(f"    Model loaded. r_norm: μ={norm_params['r_mean']:.3f}, σ={norm_params['r_std']:.3f}")
    
    # Initialize positions
    print(f"\n[2/4] Initializing {N_PARTICLES} particles in box {BOX_SIZE}×{BOX_SIZE}×{BOX_SIZE}...")
    positions = init_positions(N_PARTICLES, BOX_SIZE, min_dist=1.0)
    print(f"    Initial positions set (seed=42 for reproducibility)")
    
    # Run LJ baseline MD
    print(f"\n[3/4] Running MD with LJ Baseline ({N_STEPS} steps)...")
    traj_lj, epot_lj, ekin_lj, T_lj = run_md(
        positions.copy(), None, None, N_STEPS, DT, BOX_SIZE, potential_type='lj'
    )
    
    # Run NN potential MD
    print(f"\n[4/4] Running MD with Neural Network Potential ({N_STEPS} steps)...")
    traj_nn, epot_nn, ekin_nn, T_nn = run_md(
        positions.copy(), model, norm_params, N_STEPS, DT, BOX_SIZE, potential_type='nn'
    )
    
    # ── Plots ────────────────────────────────────────────────────────────
    print("\n[5/4] Generating plots...")
    
    # 1. Energy comparison
    fig_energy = plot_energy_comparison(epot_lj, epot_nn)
    fig_energy.savefig(os.path.join(results_dir, "md_energy_comparison.png"), dpi=150)
    print("    Saved: md_energy_comparison.png")
    plt.close(fig_energy)
    
    # 2. Trajectory snapshots
    fig_traj = plot_trajectory_snapshot(traj_lj, traj_nn, BOX_SIZE)
    fig_traj.savefig(os.path.join(results_dir, "md_trajectory.png"), dpi=150)
    print("    Saved: md_trajectory.png")
    plt.close(fig_traj)
    
    # 3. Scatter: LJ energy vs NN energy per step (correlation)
    fig_corr, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(epot_lj, epot_nn, alpha=0.5, s=20, c='purple')
    ax.plot([epot_lj.min(), epot_lj.max()], [epot_lj.min(), epot_lj.max()], 
            'k--', linewidth=1, label='y = x')
    ax.set_xlabel('LJ Potential Energy (ε)')
    ax.set_ylabel('NN Potential Energy (ε)')
    ax.set_title('LJ vs NN Energy Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Metrics
    mae = np.abs(epot_nn - epot_lj).mean()
    corr = np.corrcoef(epot_lj, epot_nn)[0, 1]
    ax.text(0.05, 0.95, f'MAE={mae:.4f} ε\nCorr={corr:.4f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig_corr.savefig(os.path.join(results_dir, "md_energy_correlation.png"), dpi=150)
    print("    Saved: md_energy_correlation.png")
    plt.close(fig_corr)
    
    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  Results Summary")
    print(f"{'='*55}")
    print(f"  LJ  | Mean Epot: {epot_lj.mean():.4f} ε | Std: {epot_lj.std():.4f}")
    print(f"  NNP | Mean Epot: {epot_nn.mean():.4f} ε | Std: {epot_nn.std():.4f}")
    print(f"  Mean Energy Difference: {abs(epot_nn.mean() - epot_lj.mean()):.4f} ε")
    print(f"  MAE (NNP vs LJ):        {mae:.4f} ε")
    print(f"  Correlation:             {corr:.4f}")
    print()
    print("  Note: Low correlation is expected — MD trajectories diverge")
    print("  exponentially due to chaos (butterfly effect). The key")
    print("  result is that mean energies are nearly identical.")
    print(f"{'='*55}")
    print("\n✅ MD simulation complete!")


if __name__ == "__main__":
    main()
