"""
train.py
--------
Train the neural network potential on LJ data.

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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.nn_potential import NeuralNetworkPotential


# ── Hyperparameters ─────────────────────────────────────────────────────────
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4       # 降低学习率，避免震荡
WEIGHT_DECAY = 1e-5        # L2 正则，防止过拟合
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
    """Reverse standardization. Handles both Tensor and float mean/std."""
    if hasattr(mean, 'item'):
        mean = mean.item()
    if hasattr(std, 'item'):
        std = std.item()
    return x_norm * std + mean


def main():
    print(f"Using device: {DEVICE}")

    # ── 1. Load Data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    r, energy, force = load_data(DATA_PATH)
    print(f"    Dataset size: {len(r)} samples")
    print(f"    r range:      [{r.min():.3f}, {r.max():.3f}]")
    print(f"    Energy range: [{energy.min():.3f}, {energy.max():.3f}]")

    # ── 2. Normalize Data ─────────────────────────────────────────────────
    print("\n[2/5] Normalizing data (zero-mean, unit-variance)...")
    r_norm, r_mean, r_std = normalize(r)
    e_norm, e_mean, e_std = normalize(energy)
    
    norm_params = {
        'r_mean': r_mean.item(),
        'r_std': r_std.item(),
        'e_mean': e_mean.item(),
        'e_std': e_std.item()
    }
    print(f"    r:  mean={r_mean:.3f}, std={r_std:.3f}")
    print(f"    E:  mean={e_mean:.3f}, std={e_std:.3f}")

    # ── 3. Create DataLoader ─────────────────────────────────────────────
    dataset = TensorDataset(r_norm, e_norm)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # ── 4. Initialize Model ───────────────────────────────────────────────
    print("\n[3/5] Initializing model...")
    model = NeuralNetworkPotential(hidden_dims=[128, 128, 64]).to(DEVICE)  # 加大网络
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # ── 学习率调度器：监控 loss，稳定下降 ──────────────────────────────
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
    )
    
    criterion = nn.MSELoss()
    print(f"    Architecture: [128, 128, 64]")
    print(f"    Parameters:  {sum(p.numel() for p in model.parameters())}")

    # ── 5. Training Loop ──────────────────────────────────────────────────
    print(f"\n[4/5] Training for {EPOCHS} epochs...")
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    PATIENCE = 100  # 早停：loss 连续 100 epochs 不下降就停止

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_r, batch_e in dataloader:
            batch_r = batch_r.to(DEVICE)
            batch_e = batch_e.to(DEVICE)

            # Forward pass
            pred_e = model(batch_r).squeeze()
            loss = criterion(pred_e, batch_e)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止爆炸
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        # 学习率调度
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最优模型
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1:4d}/{EPOCHS} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

        if patience_counter >= PATIENCE:
            print(f"\n    Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    # Restore best model
    model.load_state_dict(best_state)
    model.to(DEVICE)
    
    # Save model + norm params
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_params': norm_params,
        'losses': train_losses
    }, MODEL_SAVE_PATH)
    print(f"\n    Model saved to: {MODEL_SAVE_PATH}")

    # ── 6. Evaluation & Visualization ─────────────────────────────────────
    print("\n[5/5] Generating plots...")

    model.eval()
    with torch.no_grad():
        # Full dataset prediction
        r_test = r_norm.to(DEVICE)
        e_pred_norm = model(r_test).cpu().squeeze()
        e_pred = denormalize(e_pred_norm, e_mean, e_std)

        # Smooth curve for plotting
        r_smooth = np.linspace(r.min().item(), r.max().item(), 500)
        r_smooth_norm = ((torch.tensor(r_smooth, dtype=torch.float32) - r_mean) / r_std)
        e_smooth_norm = model(r_smooth_norm.unsqueeze(-1).to(DEVICE)).cpu().squeeze().numpy()
        e_smooth = denormalize(e_smooth_norm, e_mean, e_std)

    # True LJ energy
    def lj_energy(r_arr):
        r_arr = np.asarray(r_arr, dtype=np.float64)
        sr6 = (1.0 / r_arr) ** 6
        sr12 = sr6 ** 2
        return 4.0 * (sr12 - sr6)

    e_true = lj_energy(r.numpy())
    e_smooth_true = lj_energy(r_smooth)

    # ── Plot 1: Energy comparison ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(r_smooth, e_smooth_true, 'b-', label='Lennard-Jones (Ground Truth)', linewidth=2, zorder=2)
    axes[0].plot(r_smooth, e_smooth, 'r--', label='Neural Network Prediction', linewidth=2, zorder=3)
    axes[0].scatter(r.numpy(), e_true, s=1, alpha=0.3, color='gray', label='Training Data', zorder=1)
    axes[0].axhline(0, color='gray', linestyle=':', linewidth=0.8)
    axes[0].set_xlabel('Interatomic distance r (σ)', fontsize=11)
    axes[0].set_ylabel('Potential Energy V(r) (ε)', fontsize=11)
    axes[0].set_title('LJ Potential vs Neural Network Potential', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1.5, 3.0)
    axes[0].set_xlim(0.8, 3.5)

    # ── Plot 2: Training loss curve ──────────────────────────────────────
    axes[1].plot(train_losses, color='forestgreen', linewidth=1, alpha=0.8)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('MSE Loss (normalized)', fontsize=11)
    axes[1].set_title('Training Loss Curve', fontsize=12)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Mark best point
    best_epoch = np.argmin(train_losses)
    axes[1].axvline(best_epoch, color='orange', linestyle='--', alpha=0.7, label=f'Best: epoch {best_epoch+1}')
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "training_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"    Plot saved to: {plot_path}")
    plt.close()

    # ── Metrics ───────────────────────────────────────────────────────────
    e_pred_np = e_pred.numpy() if hasattr(e_pred, 'numpy') else np.array(e_pred)
    e_true_np = energy.numpy()
    
    mse = ((e_pred_np - e_true_np) ** 2).mean()
    mae = np.abs(e_pred_np - e_true_np).mean()
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = np.sum((e_true_np - e_pred_np) ** 2)
    ss_tot = np.sum((e_true_np - e_true_np.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n    ── Final Metrics ─────────────────────────")
    print(f"    MSE:  {mse:.6f}")
    print(f"    MAE:  {mae:.6f}")
    print(f"    RMSE: {rmse:.6f}")
    print(f"    R²:   {r2:.6f}")
    print(f"    Best epoch: {best_epoch+1}")
    print(f"\n✅ Training complete!")


if __name__ == "__main__":
    main()
