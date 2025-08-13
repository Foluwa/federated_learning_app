# =============================================================================
# core.py
# Shared utilities, models, and helpers for both server and clients.
# =============================================================================
import os, json, math, random, time, gc, contextlib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import medmnist
from medmnist import INFO
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
import pandas as pd
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
try:
    from opacus import PrivacyEngine
    _HAS_OPACUS = True
except Exception:
    _HAS_OPACUS = False

@dataclass
class CFG:
    seed: int = 42
    dataset: str = "pathmnist"
    num_virtual_clients: int = 10
    client_fraction: float = 0.30
    rounds: int = 10
    batch_size: int = 64
    val_batch_size: int = 256
    local_epochs: int = 5
    lr: float = 8e-4
    weight_decay: float = 1e-4
    teacher_epochs: int = 10
    task_mode: str = "class-episodes"
    classes_per_task: int = 3
    tasks_per_client: int = 3
    use_ewc: bool = True
    ewc_lambda: float = 100.0
    ewc_gamma: float = 0.97
    ewc_max_batches: int = 100
    use_adapters: bool = True
    adapter_inner_steps: int = 3
    adapter_lr: float = 5e-4
    prune_for_deploy: bool = True
    deploy_ft_epochs: int = 5
    quantize_for_deploy: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    use_amp: bool = True
    amp_dtype: str = "bf16"
    bigger_batch_on_gpu: bool = True
    gpu_batch_multiplier: int = 2
    limit_train_batches: int = 0
    amp_in_fisher: bool = False
    strategies_to_run: List[str] = None
    dp_enabled: bool = False
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 1.0
    dp_target_delta: float = 1e-5
    min_channels_per_layer: int = 8
    prune_step: float = 0.05
    latency_tolerance_ms: float = 2.0
    grad_accum_steps_small_device: int = 1
    root_dir: str = "artifacts_fed_flwr_ray"
    teacher_dir: str = "Teacher"
    server_dir: str = "Server"
    client_metrics_dir: str = "ClientMetrics"

CFG = CFG()
if CFG.strategies_to_run is None:
    CFG.strategies_to_run = ["fedavg", "fedavgm"]

os.makedirs(CFG.root_dir, exist_ok=True)
os.makedirs(os.path.join(CFG.root_dir, CFG.teacher_dir), exist_ok=True)
os.makedirs(os.path.join(CFG.root_dir, CFG.server_dir), exist_ok=True)
os.makedirs(os.path.join(CFG.root_dir, CFG.client_metrics_dir), exist_ok=True)

with open(os.path.join(CFG.root_dir, "config.json"), "w") as f:
    json.dump({fld.name: getattr(CFG, fld.name) for fld in fields(CFG)}, f, indent=2)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_TRAIN = CFG.batch_size
BATCH_VAL = CFG.val_batch_size
info = INFO[CFG.dataset]
DataClass = getattr(medmnist, info["python_class"])
num_classes = len(info['label'])
class_names = [info['label'][k] for k in sorted(info['label'].keys())]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5,.5,.5],[.5,.5,.5])])
full_train = DataClass(split="train", transform=transform, download=True)
test_ds  = DataClass(split="test",  transform=transform,  download=True)
train_len = len(full_train)
test_len  = len(test_ds)

PARTITIONS_JSON = os.path.join(CFG.root_dir, "partitions.json")
def split_indices(indices: List[int], val_frac: float = 0.1, seed: int = 42, salt: Any = None) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    if salt is not None: rng.seed(str(seed) + str(salt))
    num_val = int(len(indices) * val_frac)
    rng.shuffle(indices)
    return indices[num_val:], indices[:num_val]

class BaseCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

class AdapterCNN(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = base_model
        for param in self.base.parameters():
            param.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.freeze_adapters = False
        
    def forward(self, x):
        with torch.no_grad():
            x = self.base(x)
        x = self.adapter(x)
        return x

    def adapter_parameters(self):
        return self.adapter.parameters()

def make_model(with_adapters=False):
    base_model = BaseCNN(in_channels=3, num_classes=num_classes)
    if not with_adapters: return base_model
    return AdapterCNN(base_model, num_classes)

def compute_metrics(model, loader, dev, num_classes):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(dev), labels.squeeze().to(dev)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
    if len(y_true) < 1: return {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        if num_classes > 2:
            auc = roc_auc_score(y_true, y_scores, average="macro", multi_class="ovr")
        else:
            auc = roc_auc_score(y_true, y_scores[:, 1])
    except Exception:
        auc = np.nan
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

def get_autocast_dtype(amp_dtype: str):
    if amp_dtype == "bf16":
        return torch.bfloat16
    elif amp_dtype == "fp16":
        return torch.float16
    else:
        return None

class EWCOnline(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        self.initialized = False

    def update(self, model, loader, dev, gamma=0.97, max_batches=100):
        if not self.initialized:
            self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
            self.fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
            self.initialized = True
            
        with contextlib.ExitStack() as stack:
            if CFG.use_amp and CFG.amp_in_fisher and torch.cuda.is_available():
                stack.enter_context(torch.autocast(device_type="cuda", dtype=get_autocast_dtype(CFG.amp_dtype)))
            for n, p in model.named_parameters():
                if p.grad is not None:
                    p.grad.zero_()
        
        fisher_new = {n: torch.zeros_like(p).to(dev) for n, p in model.named_parameters() if p.requires_grad}
        model.train()
        for batch_idx, (inputs, labels) in enumerate(loader):
            if batch_idx >= max_batches: break
            inputs, labels = inputs.to(dev), labels.squeeze().to(dev)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            model.zero_grad()
            loss.backward()
            
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher_new[n].add_(p.grad.detach().pow(2))
        
        with torch.no_grad():
            for n in fisher_new:
                fisher_new[n] /= len(loader.dataset)
                self.fisher[n] = self.fisher[n].to(dev) * gamma + fisher_new[n] * (1-gamma)

def _maybe_make_private(model, optimizer, data_loader, dp_cfg):
    if not dp_cfg.get("enabled", False) or not _HAS_OPACUS:
        return model, optimizer, data_loader, None
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=dp_cfg["noise_multiplier"], # This is a misuse of the API, opacus uses noise_multiplier here. The `make_private_with_epsilon` takes a `target_epsilon`, we need to change how this is handled in `train_loop`
        target_delta=CFG.dp_target_delta,
        max_grad_norm=dp_cfg["max_grad_norm"],
    )
    return model, optimizer, data_loader, privacy_engine

def train_loop(
    model, train_loader, val_loader, epochs, lr, weight_decay,
    ewc_state=None, ewc_lambda=0.0, tag="", dev=None, dp_cfg=None,
    grad_accum_steps=1
):
    model.to(dev)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    eps = None
    if dp_cfg and dp_cfg.get("enabled", False) and _HAS_OPACUS:
        # A more correct way to handle `opacus`
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=dp_cfg["noise_multiplier"],
            max_grad_norm=dp_cfg["max_grad_norm"],
            poisson_sampling=False, # We are not using poisson sampling for simplicity
        )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(CFG.use_amp and get_autocast_dtype(CFG.amp_dtype) == torch.float16))
    dtype = get_autocast_dtype(CFG.amp_dtype)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(dev), labels.squeeze().to(dev)
            
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=CFG.use_amp):
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                if ewc_state and ewc_lambda > 0.0:
                    ewc_loss = 0
                    for n, p in model.named_parameters():
                        if n in ewc_state.fisher:
                            ewc_loss += (ewc_state.fisher[n] * (p - ewc_state.params[n])).sum()
                    loss += ewc_lambda * ewc_loss
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            if CFG.limit_train_batches > 0 and batch_idx >= CFG.limit_train_batches: break
            
        if dp_cfg and dp_cfg.get("enabled", False) and _HAS_OPACUS:
            eps = privacy_engine.get_epsilon(CFG.dp_target_delta)
    
    return model, eps

def get_model_parameters_ndarrays(model):
    return [p.cpu().detach().numpy() for p in model.parameters()]

def set_model_parameters_from_ndarrays(model, params_nd):
    params_dict = zip(model.state_dict().keys(), params_nd)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def build_class_episodes_local(ds, client_indices, classes_per_task):
    class_indices = {i: [] for i in range(num_classes)}
    for idx in client_indices:
        class_id = int(ds[idx][1].item())
        if class_id in class_indices:
            class_indices[class_id].append(idx)
    
    task_indices = []
    class_ids = list(class_indices.keys())
    random.shuffle(class_ids)
    
    num_tasks = math.ceil(len(class_ids) / classes_per_task)
    for i in range(num_tasks):
        start = i * classes_per_task
        end = min(start + classes_per_task, len(class_ids))
        current_class_ids = class_ids[start:end]
        task_indices.append([idx for c_id in current_class_ids for idx in class_indices[c_id]])
        
    return task_indices

def central_eval_acc_from_ndarrays(params: List[np.ndarray]) -> Tuple[float, Dict]:
    model = make_model(with_adapters=False)
    set_model_parameters_from_ndarrays(model, params)
    model.to(device)
    test_loader = DataLoader(test_ds, batch_size=BATCH_VAL, shuffle=False)
    metrics = compute_metrics(model, test_loader, device, num_classes)
    loss = 1.0 - metrics["accuracy"]
    return float(loss), {"accuracy": float(metrics["accuracy"])}

def server_eval_fn(server_round: int, parameters: fl.common.Parameters, config: Dict[str, fl.common.Scalar]):
    ndarrays = parameters_to_ndarrays(parameters)
    loss, metrics = central_eval_acc_from_ndarrays(ndarrays)
    print(f"[Server] Round {server_round} validation metrics: {metrics}")
    return loss, metrics

def fit_metrics_agg(metrics: List[Tuple[int, Dict]]) -> Dict[str, fl.common.Scalar]:
    fit_metrics = {}
    total_examples = sum([num_examples for num_examples, _ in metrics])
    for num_examples, m in metrics:
        for k, v in m.items():
            if k not in fit_metrics: fit_metrics[k] = 0.0
            fit_metrics[k] += float(v) * num_examples
    return {k: v / total_examples for k, v in fit_metrics.items()}

EWC_STORE_DIR = os.path.join(CFG.root_dir, "ewc_state")
TASKPTR_DIR = os.path.join(CFG.root_dir, "task_ptrs")
def _ewc_path(cid): return os.path.join(EWC_STORE_DIR, f"{cid}.pth")
def _taskptr_path(cid): return os.path.join(TASKPTR_DIR, f"{cid}.json")
def _client_dir(cid): return os.path.join(CFG.root_dir, CFG.client_metrics_dir, cid)
def _ensure_client_dirs(cid):
    os.makedirs(EWC_STORE_DIR, exist_ok=True)
    os.makedirs(TASKPTR_DIR, exist_ok=True)
    os.makedirs(_client_dir(cid), exist_ok=True)

def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)
def _load_json(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return default

def _max_softmax_scores(model, loader, dev):
    model.eval()
    scores = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(dev)
            outputs = model(inputs)
            scores.extend(torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[0].cpu().numpy())
    return scores

def _auc_from_scores(mem_scores, non_scores):
    scores = mem_scores + non_scores
    labels = [1] * len(mem_scores) + [0] * len(non_scores)
    try:
        return roc_auc_score(labels, scores)
    except Exception:
        return np.nan

def _profile_best_batch_size(model, dev, candidate_sizes, input_shape=(3, 28, 28), steps=5):
    best_bs = 0; best_lat = float('inf')
    with contextlib.ExitStack() as stack:
        if CFG.use_amp and torch.cuda.is_available():
            stack.enter_context(torch.autocast(device_type="cuda", dtype=get_autocast_dtype(CFG.amp_dtype)))
        for bs in candidate_sizes:
            try:
                inputs = torch.randn(bs, *input_shape, device=dev)
                model.eval()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(steps):
                    _ = model(inputs)
                torch.cuda.synchronize()
                end = time.perf_counter()
                latency = (end - start) * 1000 / steps
                if latency < best_lat:
                    best_lat = latency; best_bs = bs
            except RuntimeError:
                continue
    return best_bs

def _update_seq_accuracy_matrix(cid_str, task_id, model_for_eval, ds, tasks, dev):
    cdir = _client_dir(cid_str)
    path = os.path.join(cdir, "acc_matrix.json")
    matrix = _load_json(path, default={})
    
    current_accs = {}
    for tidx, task_idxs in enumerate(tasks):
        loader = DataLoader(Subset(ds, task_idxs), batch_size=CFG.val_batch_size, shuffle=False)
        acc = compute_metrics(model_for_eval, loader, dev, num_classes)["accuracy"]
        current_accs[str(tidx)] = float(acc)
    matrix[str(task_id)] = current_accs
    _save_json(path, matrix)
    return matrix

def _compute_forgetting_metrics(cid_str):
    cdir = _client_dir(cid_str)
    path = os.path.join(cdir, "acc_matrix.json")
    matrix = _load_json(path, default={})
    if not matrix: return np.nan, np.nan, np.nan
    df = pd.DataFrame(matrix).astype(float)
    df.index = df.index.astype(int)
    num_tasks = len(df.columns)
    
    if num_tasks < 2: return np.nan, np.nan, np.nan

    avg_acc_seq = df.values.diagonal().mean()
    
    bwt = 0.0
    for t_idx in range(1, num_tasks):
        bwt += df.iloc[t_idx, t_idx-1] - df.iloc[t_idx-1, t_idx-1]
    bwt /= (num_tasks - 1)
    
    total_forgetting = 0.0
    for t_idx in range(1, num_tasks):
        f = 0.0
        for task_seen in range(t_idx):
            f = max(f, df.iloc[task_seen, task_seen] - df.iloc[t_idx, task_seen])
        total_forgetting += f
    avg_forgetting = total_forgetting / (num_tasks - 1)

    return avg_acc_seq, bwt, avg_forgetting

def save_history_json(hist, path):
    history_dict = {
        "rounds": hist.rounds,
        "metrics_distributed_fit": hist.metrics_distributed_fit,
        "metrics_distributed_evaluate": hist.metrics_distributed_evaluate,
        "metrics_centralized": hist.metrics_centralized
    }
    with open(path, "w") as f:
        json.dump(history_dict, f, indent=2)

def save_params_ndarrays_to_pth(params_nd, path):
    torch.save([torch.tensor(p) for p in params_nd], path)

def _as_ndarrays(params: fl.common.Parameters) -> List[np.ndarray]:
    return parameters_to_ndarrays(params)

def add_summary_row(rows, stage, client_name, model_type, model, metrics,
                    specialty_acc, transfer_avg, ewc_lambda, bytes_up,
                    bytes_down, avg_fit_latency_ms, reliability,
                    forgetting=np.nan, bwt=np.nan, avg_forgetting=np.nan,
                    latency_ms=np.nan, used_sparsity=0.0, quant_mode="none",
                    dp_enabled=False, epsilon=np.nan, mia_auc=np.nan,
                    meta_algo="", reptile_delta_norm=np.nan):
    row = {
        "stage": stage, "client_name": client_name, "model_type": model_type,
        "accuracy": metrics["accuracy"], "precision": metrics["precision"],
        "recall": metrics["recall"], "f1": metrics["f1"], "auc": metrics["auc"],
        "specialty_acc": specialty_acc, "transfer_avg_acc": transfer_avg,
        "ewc_lambda": ewc_lambda, "bytes_up": bytes_up, "bytes_down": bytes_down,
        "avg_fit_latency_ms": avg_fit_latency_ms, "reliability": reliability,
        "forgetting_avg_acc": forgetting, "bwt_avg_acc": bwt,
        "avg_forgetting": avg_forgetting, "latency_ms": latency_ms,
        "used_sparsity": used_sparsity, "quant_mode": quant_mode,
        "dp_enabled": dp_enabled, "epsilon": epsilon, "mia_auc": mia_auc,
        "meta_algo": meta_algo, "reptile_delta_norm": reptile_delta_norm
    }
    rows.append(row)

def make_loader_from_indices(ds, indices, batch_size, shuffle, num_workers=0):
    return DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Create a teacher model for pre-training and initialization
teacher = make_model(with_adapters=False).to(device)
train_ds, val_ds = random_split(full_train, [int(0.9*len(full_train)), len(full_train)-int(0.9*len(full_train))], generator=torch.Generator().manual_seed(CFG.seed))
train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=CFG.num_workers)
val_loader = DataLoader(val_ds, batch_size=BATCH_VAL, shuffle=False, num_workers=CFG.num_workers)
teacher, _ = train_loop(teacher, train_loader, val_loader, epochs=CFG.teacher_epochs, lr=CFG.lr, weight_decay=CFG.weight_decay, dev=device, dp_cfg={"enabled":False})
teacher_mets = compute_metrics(teacher, test_loader, device, num_classes)
save_params_ndarrays_to_pth(get_model_parameters_ndarrays(teacher), os.path.join(CFG.root_dir, CFG.teacher_dir, "teacher_model.pth"))

client_full_loaders = {}
PARTITIONS_JSON_PATH = os.path.join(CFG.root_dir, "partitions.json")
with open(PARTITIONS_JSON_PATH, "r") as f:
    partitions = json.load(f)
for cid, indices in partitions.items():
    client_full_loaders[cid] = make_loader_from_indices(full_train, indices, BATCH_VAL, False)

# Device profiles for simulation
DEVICE_PROFILES = {
    "high-end": {"latency_ms": 1.0, "flops_g": 100, "mem_mb": 8192, "num_cores": 12, "batch_size_range": [32, 64, 128], "training_params": {"lr": 1e-3, "epochs": 5}},
    "mid-range": {"latency_ms": 5.0, "flops_g": 20, "mem_mb": 2048, "num_cores": 4, "batch_size_range": [16, 32], "training_params": {"lr": 8e-4, "epochs": 3}},
    "low-end": {"latency_ms": 10.0, "flops_g": 5, "mem_mb": 512, "num_cores": 2, "batch_size_range": [8, 16], "training_params": {"lr": 5e-4, "epochs": 2}},
}
PROFILE_NAMES = list(DEVICE_PROFILES.keys())
def assign_profile(cid: str) -> str:
    rng = random.Random(CFG.seed + int(cid))
    return rng.choice(PROFILE_NAMES)
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def model_size_mb(model):
    return count_params(model) * 4 / 1024**2
