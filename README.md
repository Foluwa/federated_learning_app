# Adaptive Federated Healthcare (Flower + Docker + Colab)

This is a portable app for **federated learning on MedMNIST PathMNIST** with:
- Elastic Weight Consolidation (EWC)
- Adapter-based personalization
- Simple compression controller (pruning + optional dynamic INT8)
- Optional Differential Privacy (Opacus)
- Works in **Colab**, **local simulation**, and **Docker** (server/clients)

## Quickstart: Local Simulation (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python adaptive_federated_healthcare/experiments/run_simulation.py
