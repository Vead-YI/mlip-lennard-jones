# Machine Learning Interatomic Potential for Lennard-Jones System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A mini research project exploring data-driven interatomic potentials using neural networks, trained on a classical Lennard-Jones system.

---

## 1. Introduction

Machine Learning Interatomic Potentials (MLIPs) have emerged as a powerful bridge between the accuracy of quantum mechanical calculations and the efficiency of classical force fields. Traditional potentials like Lennard-Jones (LJ) are fast but limited in transferability. Neural network potentials (NNPs) can learn complex energy landscapes directly from data.

This project implements a simple but complete pipeline:

```
Classical LJ Potential → Generate Data → Train Neural Network → Run MD Simulation
```

**Why this matters:**
- Classical MD is fast but physically limited
- Ab initio MD is accurate but computationally expensive
- MLIPs offer a practical middle ground — and this project demonstrates the core idea from scratch

---

## 2. Method

### 2.1 Lennard-Jones Potential (Baseline)

The classical LJ potential between two atoms separated by distance *r*:

$$V(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]$$

where:
- $\epsilon$ = well depth (energy scale)
- $\sigma$ = finite distance at which inter-particle potential is zero

The force is derived analytically:

$$F(r) = -\frac{dV}{dr} = \frac{24\epsilon}{r} \left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]$$

### 2.2 Neural Network Potential

A feedforward neural network is trained to approximate the LJ potential energy surface:

- **Input:** Interatomic distance *r* (or descriptor vector)
- **Architecture:** 3 fully connected layers with ReLU activations
- **Output:** Potential energy *V(r)*
- **Loss:** Mean Squared Error (MSE) on energy (+ force matching, optional)

### 2.3 Molecular Dynamics Integration

The trained ML potential is integrated into a simple MD loop using the **Velocity Verlet** algorithm:

```
v(t + dt/2) = v(t) + (F(t)/m) * dt/2
r(t + dt)   = r(t) + v(t + dt/2) * dt
F(t + dt)   = -dV_ML/dr evaluated at r(t + dt)
v(t + dt)   = v(t + dt/2) + (F(t + dt)/m) * dt/2
```

---

## 3. Results

*(Results and figures will be added as the project progresses)*

- [ ] LJ vs ML energy curve comparison
- [ ] Training loss curve
- [ ] MD trajectory comparison (LJ vs ML)
- [ ] Energy conservation test

---

## 4. Project Structure

```
mlip-lennard-jones/
├── data/                   # Generated atomistic datasets
│   └── generate_data.py    # LJ data generation script
├── model/                  # Neural network architecture
│   └── nn_potential.py     # NNP model definition
├── training/               # Training scripts and logs
│   └── train.py            # Training loop
├── md_simulation/          # MD integration
│   └── md_verlet.py        # Velocity Verlet MD with ML potential
├── results/                # Output figures and metrics
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 5. Installation

```bash
git clone https://github.com/Vead-YI/mlip-lennard-jones.git
cd mlip-lennard-jones
pip install -r requirements.txt
```

---

## 6. Usage

```bash
# Step 1: Generate training data
python data/generate_data.py

# Step 2: Train the neural network potential
python training/train.py

# Step 3: Run MD simulation with ML potential
python md_simulation/md_verlet.py
```

---

## 7. Future Work

This project is a starting point. Planned extensions include:

- **Multi-body descriptors:** Incorporate angular information (e.g., symmetry functions à la Behler-Parrinello)
- **Equivariant neural networks:** Explore NequIP / MACE architectures for better physical symmetry handling
- **Force matching:** Train on forces in addition to energies for improved accuracy
- **Real materials:** Apply the pipeline to DFT-generated datasets (e.g., Cu, Si)
- **Active learning:** Iteratively improve the potential by querying uncertain configurations

---

## 8. References

1. Behler, J. & Parrinello, M. (2007). *Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces.* PRL 98, 146401.
2. Batzner, S. et al. (2022). *E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials.* Nature Communications.
3. Lennard-Jones, J. E. (1924). *On the determination of molecular fields.* Proc. R. Soc. Lond. A.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*This project is part of my journey into computational materials science and machine learning interatomic potentials.*
