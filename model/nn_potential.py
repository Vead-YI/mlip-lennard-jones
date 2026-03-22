"""
nn_potential.py
---------------
Neural Network Potential (NNP) architecture for approximating
the Lennard-Jones potential energy surface.

Architecture:
    Input  : interatomic distance r (scalar or batch)
    Hidden : 3 fully connected layers with ReLU activations
    Output : potential energy V(r)
"""

import torch
import torch.nn as nn


class NeuralNetworkPotential(nn.Module):
    """
    Simple feedforward neural network to approximate V(r).

    Args:
        hidden_dims (list): sizes of hidden layers, e.g. [64, 64, 32]
        activation  : activation function (default: ReLU)
    """

    def __init__(self, hidden_dims=None, activation=None):
        
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64, 32]
        if activation is None:
            activation = nn.ReLU()

        layers = []
        in_dim = 1  # single input: distance r

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation)
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))  # output: energy

        self.net = nn.Sequential(*layers)

    def forward(self, r):
        """
        Args:
            r : Tensor of shape (N,) or (N, 1) — interatomic distances
        Returns:
            energy : Tensor of shape (N, 1) — predicted potential energy
        """
        if r.dim() == 1:
            r = r.unsqueeze(-1)  # (N,) → (N, 1)
        return self.net(r)

    def predict_force(self, r):
        """
        Compute force F = -dV/dr via automatic differentiation.

        Args:
            r : Tensor of shape (N,), requires_grad=True
        Returns:
            force : Tensor of shape (N,)
        """
        r = r.unsqueeze(-1).requires_grad_(True)
        energy = self.net(r)
        grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=r,
            create_graph=True
        )[0]
        force = -grad.squeeze(-1)
        return force


if __name__ == "__main__":
    # Quick sanity check
    model = NeuralNetworkPotential()
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    r_test = torch.linspace(0.9, 3.5, 10)
    with torch.no_grad():
        e_test = model(r_test)
    print(f"\nTest input  r: {r_test.numpy().round(2)}")
    print(f"Test output E: {e_test.squeeze().numpy().round(4)}")
