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
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
HIDDEN_DIMS = [64, 64, 32]
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


def normalize(x, x_min=None, x_max=None):
    """Min-max normalization to [0, 1]."""
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()
    return (x - x_min) / (x_max - x_min), x_min, x_max


def denormalize(x_norm, x_min, x_max):
    """Reverse normalization."""
    # Convert tensors to floats if needed
    if isinstance(x_min, torch.Tensor):
        x_min = x_min.item()
    if isinstance(x_max, torch.Tensor):
        x_max = x_max.item()
    return x_norm * (x_max - x_min) + x_min


def main():
    print(f"Using device: {DEVICE}")

    # ── 1. Load Data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    r, energy, force = load_data(DATA_PATH)
    print(f"    Dataset size: {len(r)} samples")
    print(f"    r range:      [{r.min():.3f}, {r.max():.3f}]")
    print(f"    Energy range: [{energy.min():.3f}, {energy.max():.3f}]")

    # ── 2. Normalize Data ─────────────────────────────────────────────────
    print("\n[2/5] Normalizing data...")
    r_norm, r_min, r_max = normalize(r)
    e_norm, e_min, e_max = normalize(energy)
    
    # Save normalization params for later use
    norm_params = {
        'r_min': r_min.item(),
        'r_max': r_max.item(),
        'e_min': e_min.item(),
        'e_max': e_max.item()
    }
    print(f"    r:  [{r_min:.3f}, {r_max:.3f}] -> [0, 1]")
    print(f"    E:  [{e_min:.3f}, {e_max:.3f}] -> [0, 1]")

    # ── 3. Create DataLoader ─────────────────────────────────────────────
    dataset = TensorDataset(r_norm, e_norm)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ── 4. Initialize Model ───────────────────────────────────────────────
    print("\n[3/5] Initializing model...")
    model = NeuralNetworkPotential(hidden_dims=HIDDEN_DIMS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"    Architecture: {HIDDEN_DIMS}")
    print(f"    Parameters:   {sum(p.numel() for p in model.parameters())}")

    # ── 5. Training Loop ──────────────────────────────────────────────────
    print(f"\n[4/5] Training for {EPOCHS} epochs...")
    train_losses = []

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
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6f}")

    # Save model + norm params
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_params': norm_params,
        'losses': train_losses
    }, MODEL_SAVE_PATH)
    print(f"\n    Model saved to: {MODEL_SAVE_PATH}")

    # ── 6. Evaluation & Visualization ─────────────────────────────────────
    print("\n[5/5] Generating plots...")

    # Test on full dataset
    model.eval()
    with torch.no_grad():
        r_test = r_norm.to(DEVICE)
        e_pred_norm = model(r_test).cpu().squeeze()
        e_pred = denormalize(e_pred_norm, e_min, e_max)

    # Plot 1: Energy comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # LJ vs ML Energy Curve
    r_plot = np.linspace(r_min.item(), r_max.item(), 200)
    with torch.no_grad():
        r_plot_tensor = torch.tensor(r_plot, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        e_plot_pred = model(r_plot_tensor).cpu().squeeze().numpy()
    # Denormalize predictions
    e_plot_pred = denormalize(e_plot_pred, e_min, e_max)
    
    # True LJ energy
    def lj_energy(r):
        sr6 = (1.0 / r) ** 6
        sr12 = sr6 ** 2
        return 4.0 * (sr12 - sr6)

    e_plot_true = lj_energy(r_plot)
    
    # Convert to numpy for plotting
    r_plot = r_plot

    axes[0].plot(r_plot, e_plot_true, 'b-', label='Lennard-Jones', linewidth=2)
    axes[0].plot(r_plot, e_plot_pred, 'r--', label='Neural Network', linewidth=2)
    axes[0].set_xlabel('Interatomic distance r (σ)')
    axes[0].set_ylabel('Potential Energy V(r) (ε)')
    axes[0].set_title('LJ vs ML Potential Energy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1.5, 3.0)

    # Training loss curve
    axes[1].plot(train_losses, 'g-', linewidth=1)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('Training Loss Curve')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "training_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"    Plot saved to: {plot_path}")
    plt.close()

    # Calculate final metrics
    mse = ((e_pred - energy.numpy()) ** 2).mean()
    print(f"\n    Final MSE: {mse:.6f}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
