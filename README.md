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
```


```
# find a 3.11 on your system (or install via pyenv)
python3.11 -V

# tell Poetry to use it
poetry env use 3.11
poetry lock --no-cache --regenerate
poetry install
poetry run python -m adaptive_federated_healthcare.experiments.run_simulation
```