"""
generate_data.py
----------------
Generate training data for the ML interatomic potential.

This script samples random interatomic distances and computes
the corresponding Lennard-Jones potential energy and forces.

Output:
    data/lj_dataset.npz  — contains arrays: r, energy, force
"""

import numpy as np
import os

# ── LJ Parameters (Argon-like, reduced units) ──────────────────────────────
EPSILON = 1.0   # well depth
SIGMA   = 1.0   # zero-crossing distance
R_MIN   = 0.85  # minimum distance (avoid singularity)
R_MAX   = 3.5   # cutoff distance
N_SAMPLES = 5000


def lj_energy(r, epsilon=EPSILON, sigma=SIGMA):
    """Lennard-Jones potential energy V(r)."""
    sr6  = (sigma / r) ** 6
    sr12 = sr6 ** 2
    return 4.0 * epsilon * (sr12 - sr6)


def lj_force(r, epsilon=EPSILON, sigma=SIGMA):
    """
    LJ force magnitude F(r) = -dV/dr.
    Positive value = repulsive (pushes atoms apart).
    """
    sr6  = (sigma / r) ** 6
    sr12 = sr6 ** 2
    return (24.0 * epsilon / r) * (2.0 * sr12 - sr6)


def generate_dataset(n_samples=N_SAMPLES, r_min=R_MIN, r_max=R_MAX, seed=42):
    """
    Sample distances uniformly in [r_min, r_max] and compute LJ energy/force.
    Returns:
        r      : (N,) array of distances
        energy : (N,) array of potential energies
        force  : (N,) array of force magnitudes
    """
    rng = np.random.default_rng(seed)

    # Dense sampling near the minimum (r ~ 2^(1/6) * sigma ≈ 1.122)
    # Use a mix: uniform + concentrated near equilibrium
    r_uniform = rng.uniform(r_min, r_max, size=int(n_samples * 0.7))
    r_near_eq = rng.normal(loc=2 ** (1/6), scale=0.15, size=int(n_samples * 0.3))
    r_near_eq = np.clip(r_near_eq, r_min, r_max)

    r = np.concatenate([r_uniform, r_near_eq])
    r = np.sort(r)

    energy = lj_energy(r)
    force  = lj_force(r)

    return r, energy, force


def main():
    print("Generating Lennard-Jones dataset...")
    r, energy, force = generate_dataset()

    # Clip extreme energies (very short distances blow up)
    mask = np.abs(energy) < 50.0
    r, energy, force = r[mask], energy[mask], force[mask]

    print(f"  Samples after filtering: {len(r)}")
    print(f"  r range:      [{r.min():.3f}, {r.max():.3f}]")
    print(f"  Energy range: [{energy.min():.3f}, {energy.max():.3f}]")
    print(f"  Force range:  [{force.min():.3f}, {force.max():.3f}]")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "lj_dataset.npz")
    np.savez(out_path, r=r, energy=energy, force=force)
    print(f"\nDataset saved to: {out_path}")

    # Quick sanity plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(r, energy, '.', markersize=1, alpha=0.5, color='steelblue')
        axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[0].set_xlabel("r (σ)")
        axes[0].set_ylabel("V(r) (ε)")
        axes[0].set_title("LJ Potential Energy")
        axes[0].set_ylim(-1.5, 3.0)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(r, force, '.', markersize=1, alpha=0.5, color='tomato')
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[1].set_xlabel("r (σ)")
        axes[1].set_ylabel("F(r) (ε/σ)")
        axes[1].set_title("LJ Force")
        axes[1].set_ylim(-10, 20)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), "..", "results", "lj_reference.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        print(f"Reference plot saved to: {plot_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot.")


if __name__ == "__main__":
    main()
