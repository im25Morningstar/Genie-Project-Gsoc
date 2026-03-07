# ML4SCI GSoC 2026 – Evaluation Tasks
### Project: Learning Parametrization with Implicit Neural Representations

This repository contains my solutions to the **ML4SCI GENIE GSoC 2026 evaluation tasks**.

The goal of these tasks is to explore different machine learning approaches for representing and analyzing **particle physics jet events** from calorimeter detector data.

Dataset used: **Quark/Gluon jet dataset** consisting of three channels:
- ECAL
- HCAL
- Tracks

Each event is represented as **125 × 125 images across 3 detector channels**.

---

---

# Dataset Description

The dataset consists of **quark and gluon jet events** stored in an HDF5 file.

Each event includes:

- **X_jets** → detector images (ECAL, HCAL, Tracks)
- **y** → jet label (quark or gluon)
- **pt** → transverse momentum
- **m0** → jet mass

Shape:

```
X_jets : (139306, 125, 125, 3)

Common Task 1 – Autoencoder for Jet Event Representation
Objective

Train an autoencoder to learn a compressed representation of jet events using the three detector channels.

Method

Input: 3-channel detector images

Model: Convolutional Autoencoder

Encoder: Convolutional layers

Decoder: Transposed Convolutions

Loss Function: Mean Squared Error (MSE)

Evaluation Metrics

MSE

PSNR

SSIM

Results

The trained autoencoder successfully reconstructs jet events while preserving the spatial structure of detector energy deposits.

Example visualizations include:

Original jet event

Reconstructed jet event

Reconstruction error

Common Task 2 – Jets as Graphs (GNN Classification)
Objective

Represent jet events as graphs and perform quark/gluon classification using a Graph Neural Network.

Method

Convert detector images into point clouds by selecting non-zero energy pixels.

Treat each point as a node with features:

spatial coordinates

energy values

Construct graph edges using k-nearest neighbors (k-NN).

Train a Graph Convolutional Network (GCN) for classification.

Results

Validation Performance:

Accuracy: ~70%

ROC-AUC: ~0.75

This demonstrates that graph representations capture structural patterns in jet events.

Specific Task – Implicit Neural Representation (INR)
Objective

Represent particle detector signals as continuous functions instead of discrete images.

Idea

Instead of modeling events as grids:

E(i,j,c)

we learn a continuous mapping:

fθ(x,y,c) → E

Where:

(x,y) = spatial coordinates

c = detector channel

E = energy deposition

Model

Coordinate-based neural network

MLP architecture

Fourier feature positional encoding

Training

The model learns from coordinate–energy pairs extracted from detector images.

Loss function:

MSE = (1/N) Σ (Ei − fθ(xi, yi, ci))²
Evaluation

MSE

PSNR

SSIM

Reconstruction visualizations

Results show that Implicit Neural Representations can accurately model detector energy distributions while providing a compact continuous representation.
