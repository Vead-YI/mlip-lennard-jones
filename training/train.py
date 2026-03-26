"""
train.py
--------
Train the neural network potential on LJ data.
Energy-only training, but evaluate force accuracy as validation.

Key insight: If E(r) is learned accurately, F(r) = -dE/dr is automatically correct.
Force matching is mainly needed for multi-body systems where force directions matter.

Usage:
    python train.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.nn_potential import NeuralNetworkPotential


# ── Hyperparameters ─────────────────────────────────────────────────────────
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lj_dataset.npz")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "nnp_model.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_data(path):
    """Load LJ dataset from npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}. Run data/generate_data.py first.")

    data = np.load(path)
    r = torch.tensor(data['r'], dtype=torch.float32)
    energy = torch.tensor(data['energy'], dtype=torch.float32)
    force = torch.tensor(data['force'], dtype=torch.float32)

    return r, energy, force


def normalize(x):
    """Normalize to zero-mean, unit-variance (standardization)."""
    mean = x.mean()
    std = x.std()
    return (x - mean) / std, mean, std


def denormalize(x_norm, mean, std):
    """Reverse standardization."""
    if hasattr(mean, 'item'):
        mean = mean.item()
    if hasattr(std, 'item'):
        std = std.item()
    return x_norm * std + mean


def main():
    print("=" * 60)
    print("  Training NNP: Energy-only (Force evaluated post-hoc)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ── 1. Load Data ─────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    r, energy, force = load_data(DATA_PATH)
    print(f"    Dataset size: {len(r)} samples")
    print(f"    r range:      [{r.min():.3f}, {r.max():.3f}]")
    print(f"    Energy range: [{energy.min():.3f}, {energy.max():.3f}]")

    # ── 2. Normalize Data ─────────────────────────────────────────────────
    print("\n[2/6] Normalizing data...")
    r_norm, r_mean, r_std = normalize(r)
    e_norm, e_mean, e_std = normalize(energy)
    
    norm_params = {
        'r_mean': r_mean.item(),
        'r_std': r_std.item(),
        'e_mean': e_mean.item(),
        'e_std': e_std.item()
    }
    print(f"    r: μ={r_mean:.3f}, σ={r_std:.3f}")
    print(f"    E: μ={e_mean:.3f}, σ={e_std:.3f}")

    # ── 3. Create DataLoader ─────────────────────────────────────────────
    dataset = TensorDataset(r_norm, e_norm)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # ── 4. Initialize Model ───────────────────────────────────────────────
    print("\n[3/6] Initializing model...")
    model = NeuralNetworkPotential(hidden_dims=[128, 128, 64]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
    )
    
    criterion = nn.MSELoss()
    print(f"    Architecture: [128, 128, 64]")
    print(f"    Parameters:   {sum(p.numel() for p in model.parameters())}")

    # ── 5. Training Loop ──────────────────────────────────────────────────
    print(f"\n[4/6] Training for up to {EPOCHS} epochs...")
    
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    PATIENCE = 100

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_r, batch_e in dataloader:
            batch_r = batch_r.to(DEVICE)
            batch_e = batch_e.to(DEVICE)

            pred_e = model(batch_r).squeeze()
            loss = criterion(pred_e, batch_e)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1:4d}/{EPOCHS} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

        if patience_counter >= PATIENCE:
            print(f"\n    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.to(DEVICE)
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_params': norm_params,
        'losses': train_losses
    }, MODEL_SAVE_PATH)
    print(f"\n    Model saved to: {MODEL_SAVE_PATH}")

    # ── 6. Evaluation ─────────────────────────────────────────────────────
    print("\n[5/6] Evaluating model...")

    model.eval()
    with torch.no_grad():
        r_test = r_norm.to(DEVICE)
        e_pred_norm = model(r_test).cpu().squeeze()
        e_pred = denormalize(e_pred_norm, e_mean, e_std)

        # Smooth curve
        r_smooth = np.linspace(r.min().item(), r.max().item(), 500)
        r_smooth_t = torch.tensor(r_smooth, dtype=torch.float32)
        r_smooth_norm = (r_smooth_t - r_mean) / r_std
        e_smooth_norm = model(r_smooth_norm.unsqueeze(-1).to(DEVICE)).cpu().squeeze().numpy()
        e_smooth = denormalize(e_smooth_norm, e_mean, e_std)

    # LJ ground truth
    def lj_energy(r_arr):
        r_arr = np.asarray(r_arr, dtype=np.float64)
        sr6 = (1.0 / r_arr) ** 6
        sr12 = sr6 ** 2
        return 4.0 * (sr12 - sr6)

    def lj_force(r_arr):
        r_arr = np.asarray(r_arr, dtype=np.float64)
        sr6 = (1.0 / r_arr) ** 6
        sr12 = sr6 ** 2
        return 24.0 / r_arr * (2.0 * sr12 - sr6)

    e_true = lj_energy(r.numpy())
    f_true = lj_force(r.numpy())
    e_smooth_true = lj_energy(r_smooth)

    # Compute force predictions via autograd
    r_smooth_tensor = torch.tensor(r_smooth, dtype=torch.float32)
    r_smooth_norm_t = (r_smooth_tensor - r_mean) / r_std
    r_smooth_norm_t = r_smooth_norm_t.requires_grad_(True)
    e_smooth_tensor = model(r_smooth_norm_t.unsqueeze(-1))
    grad_smooth = torch.autograd.grad(e_smooth_tensor.sum(), r_smooth_norm_t)[0]
    f_smooth_pred = (-grad_smooth * (e_std / r_std)).detach().numpy()
    
    # Force predictions on training data
    r_tensor = r.clone().requires_grad_(True)
    r_norm_t = (r_tensor - r_mean) / r_std
    e_pred_t = model(r_norm_t.unsqueeze(-1))
    grad_t = torch.autograd.grad(e_pred_t.sum(), r_norm_t)[0]
    f_pred = (-grad_t * (e_std / r_std)).detach().numpy()

    # ── 7. Visualization ──────────────────────────────────────────────────
    print("\n[6/6] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Energy curve
    axes[0, 0].plot(r_smooth, e_smooth_true, 'b-', label='Lennard-Jones', linewidth=2)
    axes[0, 0].plot(r_smooth, e_smooth, 'r--', label='Neural Network', linewidth=2)
    axes[0, 0].scatter(r.numpy(), e_true, s=1, alpha=0.2, color='gray', label='Training Data')
    axes[0, 0].axhline(0, color='gray', linestyle=':', linewidth=0.8)
    axes[0, 0].set_xlabel('r (σ)')
    axes[0, 0].set_ylabel('V(r) (ε)')
    axes[0, 0].set_title('Potential Energy: LJ vs NNP')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-1.5, 3.0)
    axes[0, 0].set_xlim(0.8, 3.5)

    # Force curve
    axes[0, 1].plot(r_smooth, lj_force(r_smooth), 'b-', label='LJ Force', linewidth=2)
    axes[0, 1].plot(r_smooth, f_smooth_pred, 'r--', label='NNP Force (autograd)', linewidth=2)
    axes[0, 1].axhline(0, color='gray', linestyle=':', linewidth=0.8)
    axes[0, 1].set_xlabel('r (σ)')
    axes[0, 1].set_ylabel('F(r) (ε/σ)')
    axes[0, 1].set_title('Force: LJ vs NNP (derived from energy)')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-5, 20)
    axes[0, 1].set_xlim(0.8, 3.5)

    # Loss curve
    axes[1, 0].plot(train_losses, color='forestgreen', linewidth=1, alpha=0.8)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE Loss (normalized)')
    axes[1, 0].set_title('Training Loss Curve')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    best_epoch = np.argmin(train_losses)
    axes[1, 0].axvline(best_epoch, color='orange', linestyle='--', alpha=0.7, label=f'Best: epoch {best_epoch+1}')
    axes[1, 0].legend()

    # Force scatter
    axes[1, 1].scatter(f_true, f_pred, alpha=0.3, s=5, color='purple')
    axes[1, 1].plot([f_true.min(), f_true.max()], [f_true.min(), f_true.max()], 
                    'k--', linewidth=1, label='y = x')
    axes[1, 1].set_xlabel('LJ Force (ε/σ)')
    axes[1, 1].set_ylabel('NNP Force (ε/σ)')
    axes[1, 1].set_title('Force Prediction Correlation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Force metrics
    f_mae = np.abs(f_pred - f_true).mean()
    f_corr = np.corrcoef(f_true, f_pred.flatten())[0, 1]
    axes[1, 1].text(0.05, 0.95, f'MAE: {f_mae:.4f}\nCorr: {f_corr:.4f}',
                    transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "training_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"    Saved: {plot_path}")
    plt.close()

    # ── Metrics ───────────────────────────────────────────────────────────
    e_pred_np = e_pred.numpy() if hasattr(e_pred, 'numpy') else np.array(e_pred)
    e_true_np = energy.numpy()
    
    e_mse = ((e_pred_np - e_true_np) ** 2).mean()
    e_mae = np.abs(e_pred_np - e_true_np).mean()
    e_rmse = np.sqrt(e_mse)
    
    ss_res = np.sum((e_true_np - e_pred_np) ** 2)
    ss_tot = np.sum((e_true_np - e_true_np.mean()) ** 2)
    e_r2 = 1 - ss_res / ss_tot

    print(f"\n{'='*60}")
    print("  Final Metrics")
    print(f"{'='*60}")
    print(f"  ── Energy ──────────────────────────────")
    print(f"    MSE:  {e_mse:.6f}")
    print(f"    MAE:  {e_mae:.6f}")
    print(f"    RMSE: {e_rmse:.6f}")
    print(f"    R²:   {e_r2:.6f}")
    print(f"  ── Force (derived from energy) ────────")
    print(f"    MAE:  {f_mae:.6f}")
    print(f"    Corr: {f_corr:.6f}")
    print(f"{'='*60}")
    print("\n✅ Training complete!")
    print("\n📝 Note: Force is derived from energy via F = -dV/dr (autograd).")
    print("   For single-component LJ, this works well. For multi-body")
    print("   potentials, explicit force matching is recommended.")


if __name__ == "__main__":
    main()
