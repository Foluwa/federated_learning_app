# =============================================================================
# server/app.py
# Flower server-side logic and orchestration.
# =============================================================================
import os
import flwr as fl
import torch
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from ..core.core import (
    CFG,
    get_model_parameters_ndarrays,
    central_eval_acc_from_ndarrays,
    server_eval_fn,
    fit_metrics_agg,
    teacher,
    save_params_ndarrays_to_pth
)

class FedAvgStore(fl.server.strategy.FedAvg):
    def __init__(self, central_eval_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.latest_params = None
        self.best_params = None
        self.best_acc = -1.0
        self.central_eval_fn = central_eval_fn

    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters(get_model_parameters_ndarrays(teacher))

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            nd = parameters_to_ndarrays(aggregated)
            self.latest_params = nd
            if self.central_eval_fn is not None:
                acc = self.central_eval_fn(nd)[1]["accuracy"]
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_params = nd
        return aggregated, metrics

class FedAvgMStore(fl.server.strategy.FedAvgM):
    def __init__(self, central_eval_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.latest_params = None
        self.best_params = None
        self.best_acc = -1.0
        self.central_eval_fn = central_eval_fn
        self.initial_parameters = None

    def initialize_parameters(self, client_manager):
        init_nd = get_model_parameters_ndarrays(teacher)
        params = ndarrays_to_parameters(init_nd)
        self.initial_parameters = params
        return params

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            nd = parameters_to_ndarrays(aggregated)
            self.latest_params = nd
            if self.central_eval_fn is not None:
                acc = self.central_eval_fn(nd)[1]["accuracy"]
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_params = nd
        return aggregated, metrics

def build_strategy(kind: str):
    common = dict(
        fraction_fit=CFG.client_fraction,
        fraction_evaluate=0.0,
        min_fit_clients=max(1, int(CFG.client_fraction * CFG.num_virtual_clients)),
        min_available_clients=CFG.num_virtual_clients,
        # min_fit_clients = max(1, int(CFG.client_fraction * CFG.num_virtual_clients)),
        # min_available_clients = max(min_fit_clients, int(CFG.client_fraction * CFG.num_virtual_clients)),

        evaluate_fn=server_eval_fn,
        fit_metrics_aggregation_fn=fit_metrics_agg,
    )
    if kind.lower() == "fedavg":
        return "fedavg", FedAvgStore(central_eval_fn=central_eval_acc_from_ndarrays, **common)
    elif kind.lower() == "fedavgm":
        return "fedavgm", FedAvgMStore(central_eval_fn=central_eval_acc_from_ndarrays, server_momentum=0.9, **common)
    else:
        raise ValueError(f"Unknown strategy kind: {kind}")

def main_server(strategy_kind: str):
    name, strategy = build_strategy(strategy_kind)
    print(f"\n=== Running strategy: {name} ===")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=CFG.rounds),
        strategy=strategy,
    )

    if strategy.latest_params is not None:
        last_path = os.path.join(CFG.root_dir, CFG.server_dir, f"server_last_{name}.pth")
        save_params_ndarrays_to_pth(strategy.latest_params, last_path)
        print(f"Saved last checkpoint: {last_path}")

    if strategy.best_params is not None:
        best_path = os.path.join(CFG.root_dir, CFG.server_dir, f"server_best_{name}.pth")
        save_params_ndarrays_to_pth(strategy.best_params, best_path)
        print(f"Saved best-global checkpoint: {best_path}")

if __name__ == "__main__":
    import sys
    strategy_kind = sys.argv[1] if len(sys.argv) > 1 else "fedavg"
    main_server(strategy_kind)
