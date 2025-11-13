# train_piezo_equivariant.py
# 端到端预测笛卡尔 rank-3 压电张量 e_{ijk}（标签为笛卡尔 3×3×3；标准化/反标准化均在笛卡尔空间）
from __future__ import annotations

import os
import time
import logging
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# PyTorch 2.6+ weights_only 兼容
try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass

# 依赖你的数据工具
from data_utils import load_piezo_dataset, PiezoTensorScaler

# 依赖我们新增的模型模块
from models.model_piezo import PiezoEnd2End
from models.losses import (mse_piezo, symmetry_penalty_jk)

# ----------------------------- 配置 ----------------------------- #
CONFIG: Dict[str, Any] = {
    "train_json": "mp_jarvis_piezo_le40_cartesian.json",
    "structure_key": "structure",
    "piezo_key": "total",   # 标签为 3x3x3（Cartesian）
    "radial_cutoff": 5.0,
    "num_elements": 118,
    "limit_n": None,

    "save_dir": "results_equivariant_piezo",

    "epochs": 20,
    "batch_size": 1,
    "num_workers": 0,
    "val_fraction": 0.1,
    "seed": 42,

    "lr": 2e-4,
    "weight_decay": 1e-5,
    "scheduler_patience": 5,
}


# ----------------------------- 日志 ----------------------------- #
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    logger = logging.getLogger("train_piezo_equivariant")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"training_log_equivariant_{ts}.txt"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger


# ----------------------------- DataLoader ----------------------------- #
def collate_loaders(dataset, batch_size: int, num_workers: int, shuffle: bool):
    try:
        from torch_geometric.loader import DataLoader as PyGDataLoader  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "torch_geometric is required. Please install torch_geometric and torch_scatter/torch_sparse."
        ) from exc
    return PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)


# 与你原脚本一致的 batch 兼容助手
def ensure_batch(batch):
    if hasattr(batch, "x") and not hasattr(batch, "node_attrs"):
        batch.node_attrs = batch.x
    if hasattr(batch, "pos") and not hasattr(batch, "positions"):
        batch.positions = batch.pos
    if not hasattr(batch, "shifts"):
        if hasattr(batch, "edge_index") and hasattr(batch, "positions"):
            e = batch.edge_index.size(1)
            dev = batch.positions.device
            batch.shifts = torch.zeros(e, 3, device=dev, dtype=batch.positions.dtype)
        else:
            batch.shifts = torch.zeros(0, 3)
    if not hasattr(batch, "pbc"):
        dev = batch.positions.device
        batch.pbc = torch.tensor([True, True, True], dtype=torch.bool, device=dev)
    if not hasattr(batch, "total_charge"):
        num_graphs = batch.ptr.numel() - 1 if hasattr(batch, "ptr") else batch.y.shape[0]
        batch.total_charge = torch.zeros(num_graphs, device=batch.positions.device)
    return batch


# ----------------------------- 构建模型 ----------------------------- #
def build_model(cfg: Dict[str, Any], device: torch.device):
    from models.model_piezo import PiezoEnd2End
    model = PiezoEnd2End(
        num_elements=cfg["num_elements"],
        n_layers=3,      # 你可按需改
        c0=32, c1=16, c2=8, c3=4,
        hidden=256,      # 你可按需改
        device=device,
        # mode="sym",    # 如果你已经按我上一条消息加了分档开关，这里可以传入 "full"/"sym"/"sym+lt"
    )
    return model.to(device)

def collect_trainable_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    # 先全参训练；若你只想训练 to_nat/projector，可在此筛选子模块
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def evaluate_epoch(model, loader, scaler_y, device):
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    absolute_mae = 0.0
    element_count = 0

    for batch in loader:
        batch = batch.to(device)
        batch = ensure_batch(batch)

        e_pred = model(batch)  # [B,3,3,3]

        # y_norm 统一按 3x3x3 处理
        y_norm = batch.y
        if y_norm.dim() == 5 and y_norm.size(1) == 1:
            y_norm = y_norm.squeeze(1)               # [B,3,3,3]
        elif y_norm.dim() == 4 and y_norm.size(0) == 1 and e_pred.size(0) > 1:
            y_norm = y_norm.expand(e_pred.size(0), -1, -1, -1)
        elif y_norm.dim() == 4:
            pass
        else:
            y_norm = y_norm.view(e_pred.size(0), 3, 3, 3)

        e_true = scaler_y.inverse_transform(y_norm)  # [B,3,3,3]

        loss = mse_piezo(e_pred, e_true) + 1e-4 * symmetry_penalty_jk(e_pred)
        total_loss += loss.item() * e_pred.size(0)
        total_graphs += e_pred.size(0)

        mae_sum = F.l1_loss(e_pred, e_true, reduction="sum")
        absolute_mae += mae_sum.item()
        element_count += e_true.numel()

    avg_loss = total_loss / max(1, total_graphs)
    mae = absolute_mae / max(1, element_count)
    return avg_loss, mae


def train_epoch(model, loader, scaler_y, optimizer, device):
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        batch = ensure_batch(batch)

        e_pred = model(batch)  # [B,3,3,3]

        # 同 evaluate：y_norm -> inverse_transform -> e_true (3x3x3)
        y_norm = batch.y
        if y_norm.dim() == 5 and y_norm.size(1) == 1:
            y_norm = y_norm.squeeze(1)
        elif y_norm.dim() == 4 and y_norm.size(0) == 1 and e_pred.size(0) > 1:
            y_norm = y_norm.expand(e_pred.size(0), -1, -1, -1)
        elif y_norm.dim() == 4:
            pass
        else:
            y_norm = y_norm.view(e_pred.size(0), 3, 3, 3)

        e_true = scaler_y.inverse_transform(y_norm)  # [B,3,3,3]

        loss = mse_piezo(e_pred, e_true) + 1e-4 * symmetry_penalty_jk(e_pred)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * e_pred.size(0)
        total_graphs += e_pred.size(0)

    # 训练集 MAE（原始单位，和 eval 口径一致）
    model.eval()
    absolute_mae = 0.0
    element_count = 0
    for batch in loader:
        batch = batch.to(device)
        batch = ensure_batch(batch)
        with torch.no_grad():
            pred_cart = model(batch)

            y_norm = batch.y
            if y_norm.dim() == 5 and y_norm.size(1) == 1:
                y_norm = y_norm.squeeze(1)
            elif y_norm.dim() == 4 and y_norm.size(0) == 1 and pred_cart.size(0) > 1:
                y_norm = y_norm.expand(pred_cart.size(0), -1, -1, -1)
            elif y_norm.dim() == 4:
                pass
            else:
                y_norm = y_norm.view(pred_cart.size(0), 3, 3, 3)

            target_cart = scaler_y.inverse_transform(y_norm)
            mae_sum = F.l1_loss(pred_cart, target_cart, reduction="sum")
            absolute_mae += mae_sum.item()
            element_count += target_cart.numel()

    mae = absolute_mae / max(1, element_count)
    avg_loss = total_loss / max(1, total_graphs)
    return avg_loss, mae


# ----------------------------- 主入口 ----------------------------- #
def main() -> None:
    cfg = CONFIG
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(cfg["save_dir"], exist_ok=True)

    dataset = load_piezo_dataset(
        json_file_path=cfg["train_json"],
        structure_key=cfg["structure_key"],
        piezo_tensor_key=cfg["piezo_key"],
        radial_cutoff=cfg["radial_cutoff"],
        num_elements=cfg["num_elements"],
        limit_n=cfg["limit_n"],
        dtype=torch.float32,
    )
    if not dataset:
        raise RuntimeError("Dataset loading failed.")
    logger.info(f"Loaded {len(dataset)} structures.")

    train_set, val_set = train_test_split(
        dataset, test_size=cfg["val_fraction"], random_state=cfg["seed"]
    )

    scaler_y = PiezoTensorScaler().fit(train_set)
    # 注意：此 scaler 是按你原脚本对 y(Voigt 3x6) 做的归一化；我们在训练/验证里都用 inverse_transform 还原到原始单位
    for data in train_set:
        normalized = scaler_y.transform(data.y.squeeze(0))
        data.y = normalized.unsqueeze(0)
    for data in val_set:
        normalized = scaler_y.transform(data.y.squeeze(0))
        data.y = normalized.unsqueeze(0)

    train_loader = collate_loaders(train_set, cfg["batch_size"], cfg["num_workers"], shuffle=True)
    val_loader   = collate_loaders(val_set,   cfg["batch_size"], cfg["num_workers"], shuffle=False)

    model = build_model(cfg, device)
    cfg["output_mode"] = "cartesian"

    # 可选：投影器权重 dtype/device san-check
    for n, m in model.named_modules():
        if hasattr(m, "linear"):
            w = m.linear.weight
            print(f"[sanity] projector linear weight dtype={w.dtype}, device={w.device}")
            break

    params = collect_trainable_parameters(model)
    optimizer = AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=cfg["scheduler_patience"], verbose=True)

    best_val_mae = float("inf")
    best_path = os.path.join(cfg["save_dir"], "best_equivariant_piezo.pth")

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_mae = train_epoch(model, train_loader, scaler_y, optimizer, device)
        val_loss, val_mae = evaluate_epoch(model, val_loader, scaler_y, device)
        scheduler.step(val_mae)

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train L1(cart): {train_loss:.6f} | Train MAE(cart): {train_mae:.6f} | "
            f"Val L1(cart): {val_loss:.6f} | Val MAE(cart): {val_mae:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler": {"mean": scaler_y.mean, "std": scaler_y.std},
                    "config": cfg,
                },
                best_path,
            )
            logger.info(f"New best validation MAE: {best_val_mae:.6f}. Saved to {best_path}")

    logger.info(f"Training complete. Best validation MAE (cartesian): {best_val_mae:.6f}")


if __name__ == "__main__":
    main()
