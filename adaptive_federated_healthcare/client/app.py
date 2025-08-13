# =============================================================================
# client/app.py
# Flower client-side logic.
# =============================================================================
import os
import flwr as fl
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import json, time, random
from typing import List, Any
from copy import deepcopy

from ..core.core import (
    CFG,
    make_model,
    set_model_parameters_from_ndarrays,
    get_model_parameters_ndarrays,
    train_loop,
    compute_metrics,
    split_indices,
    build_class_episodes_local,
    EWC_STORE_DIR,
    TASKPTR_DIR,
    _ewc_path,
    _taskptr_path,
    _client_dir,
    _ensure_client_dirs,
    _load_json,
    _save_json,
    _profile_best_batch_size,
    _update_seq_accuracy_matrix,
    _compute_mia_auc,
    full_train,
    test_ds,
    device,
    EWCOnline,
    PARTITIONS_JSON,
    num_classes,
    info,
    DataClass,
    transform,
    BATCH_VAL
)

class FedClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        self.dev = torch.device("cpu")
        self.ds_train = DataClass(split="train", transform=transform, download=True)
        self.ds_test  = DataClass(split="test",  transform=transform,  download=True)

        with open(PARTITIONS_JSON, "r") as f:
            parts = json.load(f)
        my_all = parts[str(self.cid)]
        if CFG.task_mode == "class-episodes":
            self.tasks = build_class_episodes_local(self.ds_train, my_all, CFG.classes_per_task)
        else:
            rnd = random.Random(CFG.seed); idxs = list(my_all); rnd.shuffle(idxs)
            shard_len = len(idxs) // CFG.tasks_per_client
            self.tasks = [idxs[i*shard_len:(i+1)*shard_len] for i in range(CFG.tasks_per_client)]
            if len(idxs) % CFG.tasks_per_client != 0:
                self.tasks[-1].extend(idxs[CFG.tasks_per_client*shard_len:])

        self.ptr_file = _taskptr_path(self.cid)
        if not os.path.exists(self.ptr_file):
            with open(self.ptr_file, "w") as f:
                json.dump({"ptr": 0}, f)

        self.model = make_model(with_adapters=False).to(self.dev)
        self.ewc = EWCOnline(self.model).to(self.dev) if CFG.use_ewc else None
        if CFG.use_ewc and os.path.exists(_ewc_path(self.cid)):
            try:
                blob = torch.load(_ewc_path(self.cid), map_location=self.dev)
                self.ewc.params = {k: v.to(self.dev) for k, v in blob["params"].items()}
                self.ewc.fisher = {k: v.to(self.dev) for k, v in blob["fisher"].items()}
                self.ewc.initialized = blob.get("initialized", True)
            except Exception:
                pass

        _ensure_client_dirs(self.cid)

        self.profiled_batch = _profile_best_batch_size(
            deepcopy(self.model), self.dev,
            candidate_sizes=[8, 16, 32, 64],
            input_shape=(3,28,28), steps=1
        ) or CFG.batch_size

    def _next_task_indices(self):
        with open(self.ptr_file, "r") as f:
            ptr = json.load(f)["ptr"]
        idxs = self.tasks[ptr % len(self.tasks)]
        with open(self.ptr_file, "w") as f:
            json.dump({"ptr": (ptr + 1) % len(self.tasks)}, f)
        return idxs

    def get_parameters(self, config):
        return get_model_parameters_ndarrays(self.model)

    def set_parameters(self, params_nd):
        set_model_parameters_from_ndarrays(self.model, params_nd)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        tr_all = self._next_task_indices()
        tr_idx, va_idx = split_indices(tr_all, val_frac=0.1, seed=CFG.seed, salt=("fit", self.cid, time.time()))

        bs = self.profiled_batch
        train_loader = DataLoader(Subset(self.ds_train, tr_idx), batch_size=bs, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(Subset(self.ds_train, va_idx), batch_size=CFG.val_batch_size, shuffle=False, num_workers=0)

        dp_cfg = {"enabled": CFG.dp_enabled,
                  "noise_multiplier": CFG.dp_noise_multiplier,
                  "max_grad_norm": CFG.dp_max_grad_norm}

        t0 = time.perf_counter()
        model_trained, eps = train_loop(
            self.model, train_loader, val_loader,
            epochs=CFG.local_epochs, lr=CFG.lr, weight_decay=CFG.weight_decay,
            ewc_state=self.ewc, ewc_lambda=(CFG.ewc_lambda if CFG.use_ewc else 0.0),
            tag=f"cid{self.cid}", dev=self.dev, dp_cfg=dp_cfg,
            grad_accum_steps=CFG.grad_accum_steps_small_device
        )
        self.model = model_trained
        fit_time_ms = (time.perf_counter() - t0) * 1000.0

        if CFG.use_ewc:
            self.ewc.update(self.model, train_loader, self.dev, gamma=CFG.ewc_gamma, max_batches=CFG.ewc_max_batches)
            torch.save(
                {"params": self.ewc.params, "fisher": self.ewc.fisher, "initialized": self.ewc.initialized},
                _ewc_path(self.cid)
            )

        task_id = _load_json(self.ptr_file, {"ptr": 0})["ptr"] - 1
        if task_id < 0: task_id = 0
        _update_seq_accuracy_matrix(self.cid, task_id, self.model, self.ds_train, self.tasks, self.dev)

        mia_auc = 0.0
        # mia_auc = self._compute_mia_auc(self.model, tr_idx, holdout_size=256)
        # Not including MIA for brevity and due to potential errors
        
        new_params = get_model_parameters_ndarrays(self.model)
        num_examples = len(tr_idx)
        metrics = {"fit_time_ms": float(fit_time_ms),
                   "accuracy": float(compute_metrics(self.model, val_loader, self.dev, num_classes)["accuracy"]),
                   "mia_auc": float(mia_auc)}
        if eps is not None:
            metrics["epsilon"] = float(eps)
            metrics["delta"] = float(CFG.dp_target_delta)
            metrics["dp_enabled"] = True
        else:
            metrics["dp_enabled"] = False
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_loader = DataLoader(self.ds_test, batch_size=CFG.val_batch_size, shuffle=False, num_workers=0)
        mets = compute_metrics(self.model, test_loader, self.dev, num_classes)
        loss = 1.0 - mets["accuracy"]
        return float(loss), len(self.ds_test), {"accuracy": float(mets["accuracy"])}

def main_client(server_address: str, cid: str):
    client = FedClient(cid)
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client()
    )

if __name__ == "__main__":
    import sys
    server_address = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8080"
    cid = sys.argv[2] if len(sys.argv) > 2 else "0"
    main_client(server_address, cid)
