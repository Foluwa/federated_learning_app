# =============================================================================
# experiments/run_simulation.py
# Script to run the full simulation, including deployment and analysis.
# =============================================================================
import os
import json
import flwr as fl
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.core import (
    CFG,
    teacher,
    # teacher_mets,
    make_model,
    set_model_parameters_from_ndarrays,
    get_model_parameters_ndarrays,
    # central_eval_acc_from_ndarrays,
    save_history_json,
    save_params_ndarrays_to_pth,
    add_summary_row,
    DEVICE_PROFILES,
    assign_profile,
    # count_params,
    # model_size_mb,
    # meet_latency_target,
    full_train,
    test_loader,
    device,
    EWCOnline,
    # make_loader_from_indices,
    train_loop,
    split_indices,
    _compute_forgetting_metrics,
    compute_metrics,
    PARTITIONS_JSON,
    num_classes,
    client_full_loaders,
    BATCH_VAL
)
from ..core.core import teacher_mets, test_loader 
from ..server.app import build_strategy
from ..client.app import FedClient
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
import math

def client_fn(cid: str) -> fl.client.Client:
    return FedClient(cid).to_client()

def run_fl_simulation():
    results_by_strategy = {}

    for skind in CFG.strategies_to_run:
        name, strategy = build_strategy(skind)
        print(f"\n=== Running simulation for strategy: {name} ===")

        hist = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=CFG.num_virtual_clients,
            config=fl.server.ServerConfig(num_rounds=CFG.rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1}
        )

        hist_path = os.path.join(CFG.root_dir, f"history_{name}.json")
        save_history_json(hist, hist_path)
        print(f"Saved {hist_path}")

        if strategy.latest_params is not None:
            last_path = os.path.join(CFG.root_dir, CFG.server_dir, f"server_last_{name}.pth")
            save_params_ndarrays_to_pth(strategy.latest_params, last_path)
            print(f"Saved last checkpoint: {last_path}")

        if strategy.best_params is not None:
            best_path = os.path.join(CFG.root_dir, CFG.server_dir, f"server_best_{name}.pth")
            save_params_ndarrays_to_pth(strategy.best_params, best_path)
            print(f"Saved best-global checkpoint: {best_path}")

        results_by_strategy[name] = {"strategy": strategy, "history": hist}
        
    return results_by_strategy

def meet_latency_target(model, profile, dev_measure, example_bs, shape):
    return model, np.nan, 0.0, "none" # Placeholder for a more complex function

def deploy_and_analyze(results_by_strategy):
    chosen_name = None
    chosen_params = None
    best_acc_seen = -1.0
    for name, blob in results_by_strategy.items():
        strat = blob["strategy"]
        acc = strat.best_acc if strat.best_params is not None else -1.0
        if acc > best_acc_seen:
            best_acc_seen = acc
            chosen_name = name
            chosen_params = strat.best_params if strat.best_params is not None else strat.latest_params

    if chosen_params is None:
        fa = results_by_strategy.get("fedavg", None)
        if fa is not None and fa["strategy"].latest_params is not None:
            chosen_name = "fedavg"
            chosen_params = fa["strategy"].latest_params
        else:
            chosen_name = "teacher"
            chosen_params = get_model_parameters_ndarrays(teacher)

    print(f"\nDeploying from: {chosen_name} (best_acc={best_acc_seen:.4f})")
    global_model = make_model(with_adapters=False).to(device)
    set_model_parameters_from_ndarrays(global_model, chosen_params)

    with open(PARTITIONS_JSON, "r") as f:
        partitions = json.load(f)

    summary_rows = []

    add_summary_row(summary_rows, "teacher", "global", "fp32", teacher, teacher_mets, np.nan, np.nan, ewc_lambda=0.0,
                    bytes_up=np.nan, bytes_down=np.nan, avg_fit_latency_ms=np.nan, reliability=np.nan)

    client_profile = {f"{i}": assign_profile(f"{i}") for i in range(CFG.num_virtual_clients)}

    for cid_str in [str(i) for i in range(CFG.num_virtual_clients)]:
        cname = f"client_{int(cid_str):02d}"
        print(f"\n=== Deployment build for {cname} ===")
        cdir = os.path.join(CFG.root_dir, f"deploy_{cname}"); os.makedirs(cdir, exist_ok=True)
        student = make_model(with_adapters=CFG.use_adapters).to(device)
        # student.base.load_state_dict(global_model.base.state_dict(), strict=True)
        # Copy the global (BaseCNN) weights into the student.
        if hasattr(student, "base"):  # AdapterCNN case
            student.base.load_state_dict(global_model.state_dict(), strict=True)
        else:  # BaseCNN case
            student.load_state_dict(global_model.state_dict(), strict=True)

        profile_name = client_profile[cid_str]
        profile = DEVICE_PROFILES[profile_name]
        
        reptile_delta_norm = np.nan
        if CFG.use_adapters and CFG.adapter_inner_steps > 0:
            for p in student.base.parameters(): p.requires_grad = False
            for p in student.adapter_parameters(): p.requires_grad = True

            idxs_all = partitions[cid_str]
            tr_idx, va_idx = split_indices(idxs_all, val_frac=0.1, seed=CFG.seed, salt=("deploy", cid_str))
            cl_tr_loader = DataLoader(Subset(full_train, tr_idx), batch_size=max(profile["batch_size_range"]), shuffle=True,  num_workers=0)
            cl_va_loader = DataLoader(Subset(full_train, va_idx), batch_size=BATCH_VAL,   shuffle=False, num_workers=0)

            theta_before = deepcopy(student.state_dict())
            student, _ = train_loop(
                student, cl_tr_loader, cl_va_loader,
                epochs=min(3, CFG.adapter_inner_steps), lr=profile["training_params"]["lr"], weight_decay=0.0,
                ewc_state=None, ewc_lambda=0.0, tag="personalize_adapters", dev=device,
                dp_cfg={"enabled": False}
            )
            theta_after = student.state_dict()
            with torch.no_grad():
                num = 0; ssd = 0.0
                for k in theta_before:
                    if k in theta_after and theta_before[k].shape == theta_after[k].shape:
                        d = (theta_after[k] - theta_before[k]).detach().float().pow(2).sum().item()
                        ssd += d; num += 1
                reptile_delta_norm = float(math.sqrt(ssd)) if num > 0 else np.nan
            beta = 0.5
            new_state = {}
            for k in theta_before:
                if k in theta_after and theta_before[k].shape == theta_after[k].shape:
                    new_state[k] = theta_before[k] + (theta_after[k] - theta_before[k]) * beta
                else:
                    new_state[k] = theta_after[k]
            student.load_state_dict(new_state)
            for p in student.base.parameters(): p.requires_grad = True

        used_sparsity = 0.0; quant_mode = "none"; achieved_ms = np.nan
        if CFG.prune_for_deploy:
            student, achieved_ms, used_sparsity, quant_mode = meet_latency_target(
                student, profile, dev_measure=device, example_bs=max(profile["batch_size_range"]), shape=(3,28,28)
            )
        
        if CFG.use_ewc and CFG.deploy_ft_epochs > 0:
            idxs_all = partitions[cid_str]
            tr_idx, va_idx = split_indices(idxs_all, val_frac=0.1, seed=CFG.seed, salt=("ewc_deploy", cid_str))
            cl_tr_loader = DataLoader(Subset(full_train, tr_idx), batch_size=max(profile["batch_size_range"]), shuffle=True,  num_workers=0)
            cl_va_loader = DataLoader(Subset(full_train, va_idx), batch_size=BATCH_VAL,   shuffle=False, num_workers=0)
            ewc_deploy = EWCOnline(student).to(device)
            ewc_deploy.update(student, cl_tr_loader, device, gamma=CFG.ewc_gamma, max_batches=CFG.ewc_max_batches)
            student, _ = train_loop(
                student, cl_tr_loader, cl_va_loader,
                epochs=profile["training_params"]["epochs"], lr=profile["training_params"]["lr"], weight_decay=CFG.weight_decay,
                ewc_state=ewc_deploy, ewc_lambda=CFG.ewc_lambda, tag="deploy_ft", dev=device,
                dp_cfg={"enabled": False}
            )

        deploy_dev = torch.device("cpu")
        fp32_overall = compute_metrics(student, test_loader, deploy_dev, num_classes)
        specialty = compute_metrics(student, client_full_loaders[cid_str], deploy_dev, num_classes)["accuracy"]
        transfer_scores = []
        for other in [str(i) for i in range(CFG.num_virtual_clients) if str(i) != cid_str]:
            acc_other = compute_metrics(student, client_full_loaders[other], deploy_dev, num_classes)["accuracy"]
            transfer_scores.append(acc_other)
        transfer_avg = float(np.mean(transfer_scores)) if transfer_scores else np.nan

        avg_acc_seq, bwt, avg_forgetting = _compute_forgetting_metrics(cid_str)

        add_summary_row(summary_rows, "deploy", cname, "fp32_or_int8", student, fp32_overall, specialty, transfer_avg,
                        CFG.ewc_lambda, bytes_up=0.0, bytes_down=0.0,
                        avg_fit_latency_ms=np.nan, reliability=100.0,
                        forgetting=avg_acc_seq, bwt=bwt, avg_forgetting=avg_forgetting,
                        latency_ms=achieved_ms, used_sparsity=used_sparsity, quant_mode=quant_mode,
                        dp_enabled=False, epsilon=np.nan, mia_auc=np.nan, meta_algo="reptile", reptile_delta_norm=reptile_delta_norm)

    summary_df = pd.DataFrame(summary_rows)
    summary_df_path = os.path.join(CFG.root_dir, "metrics_summary.csv")
    summary_df.to_csv(summary_df_path, index=False)
    print("\nSaved comparison table to:", summary_df_path)
    print(summary_df.head(20))
    
if __name__ == "__main__":
    results = run_fl_simulation()
    deploy_and_analyze(results)
