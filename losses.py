# models/losses.py
import torch
import torch.nn.functional as F
from typing import Tuple

# ---------------------------------------------------------------------
# 1) 基础损失与对称性约束
# ---------------------------------------------------------------------
def mse_piezo(e_pred: torch.Tensor, e_true: torch.Tensor) -> torch.Tensor:
    """
    实际是 L1 损失（更稳健，与你当前训练脚本一致）。
    e_*: [B, 3, 3, 3]
    """
    # 强制对齐 dtype/device，避免无意的 double/float 或 cpu/cuda 混用
    if e_pred.dtype != e_true.dtype or e_pred.device != e_true.device:
        e_true = e_true.to(dtype=e_pred.dtype, device=e_pred.device)
    return F.l1_loss(e_pred, e_true)


def symmetry_penalty_jk(e: torch.Tensor) -> torch.Tensor:
    """
    约束 j<->k 对称：mean((e - e^{T_jk})^2)。
    e: [B, 3, 3, 3]
    """
    return torch.mean((e - e.transpose(-2, -1)) ** 2)


def ensure_sym_jk(e: torch.Tensor) -> torch.Tensor:
    """
    推理或可视化时可用：将输出在最后两维强制对称化。
    训练时一般不需要强制（会影响梯度），使用 symmetry_penalty_jk 即可。
    """
    return 0.5 * (e + e.transpose(-2, -1))


# ---------------------------------------------------------------------
# 2) 旋转工具（用于数据增强或一致性检查）
# ---------------------------------------------------------------------
def rotate_rank3(e: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    e'_{ijk} = R_{ip} R_{jq} R_{kr} e_{pqr}
    e: [B, 3, 3, 3], R: [B, 3, 3]
    """
    # 对齐 dtype/device
    if e.dtype != R.dtype or e.device != R.device:
        R = R.to(dtype=e.dtype, device=e.device)
    return torch.einsum('bip,bjq,bkr,bpqr->bijk', R, R, R, e)


# ---------------------------------------------------------------------
# 3) Voigt <-> Cartesian（保留 shear_factor 钩子）
#    注意：piezo e_ijk 的 Voigt 约定常用 [xx, yy, zz, yz, xz, xy]
# ---------------------------------------------------------------------
def voigt_to_cartesian_e(e_voigt: torch.Tensor, shear_factor: float = 1.0) -> torch.Tensor:
    """
    e_voigt: [B, 3, 6] -> e: [B, 3, 3, 3]
    列顺序: [xx, yy, zz, yz, xz, xy]
    shear_factor=1.0（默认）；如需工程剪切，可设为 2.0。
    """
    B = e_voigt.size(0)
    e = torch.zeros(B, 3, 3, 3, device=e_voigt.device, dtype=e_voigt.dtype)
    e[:, :, 0, 0] = e_voigt[:, :, 0]
    e[:, :, 1, 1] = e_voigt[:, :, 1]
    e[:, :, 2, 2] = e_voigt[:, :, 2]
    e[:, :, 1, 2] = e[:, :, 2, 1] = e_voigt[:, :, 3] * shear_factor
    e[:, :, 0, 2] = e[:, :, 2, 0] = e_voigt[:, :, 4] * shear_factor
    e[:, :, 0, 1] = e[:, :, 1, 0] = e_voigt[:, :, 5] * shear_factor
    return e


def cartesian_to_voigt_e(e_cart: torch.Tensor, shear_factor: float = 1.0) -> torch.Tensor:
    """
    逆映射：e_cart: [B, 3, 3, 3] -> e_voigt: [B, 3, 6]
    列顺序: [xx, yy, zz, yz, xz, xy]
    """
    B = e_cart.size(0)
    e_voigt = torch.zeros(B, 3, 6, device=e_cart.device, dtype=e_cart.dtype)
    e_voigt[:, :, 0] = e_cart[:, :, 0, 0]
    e_voigt[:, :, 1] = e_cart[:, :, 1, 1]
    e_voigt[:, :, 2] = e_cart[:, :, 2, 2]
    e_voigt[:, :, 3] = e_cart[:, :, 1, 2] / shear_factor
    e_voigt[:, :, 4] = e_cart[:, :, 0, 2] / shear_factor
    e_voigt[:, :, 5] = e_cart[:, :, 0, 1] / shear_factor
    return e_voigt


# ---------------------------------------------------------------------
# 4) 诊断：组件级 MAE （用于发现“恒定输出/塌陷”的具体维度）
# ---------------------------------------------------------------------
def mae_per_component_cartesian(e_pred: torch.Tensor, e_true: torch.Tensor) -> torch.Tensor:
    """
    返回形状 [3, 3, 3] 的分量级 MAE（先在 batch 上平均）。
    便于用 heatmap 查看哪几项学不到/坍塌。
    """
    if e_pred.dtype != e_true.dtype or e_pred.device != e_true.device:
        e_true = e_true.to(dtype=e_pred.dtype, device=e_pred.device)
    return torch.mean(torch.abs(e_pred - e_true), dim=0)  # [3,3,3]


def mae_per_component_voigt(e_pred: torch.Tensor, e_true: torch.Tensor, shear_factor: float = 1.0) -> torch.Tensor:
    """
    将笛卡尔预测/真值都投到 Voigt 后计算分量级 MAE。
    返回形状 [3, 6]（先在 batch 上平均）。
    """
    v_pred = cartesian_to_voigt_e(e_pred, shear_factor=shear_factor)
    v_true = cartesian_to_voigt_e(e_true, shear_factor=shear_factor)
    return torch.mean(torch.abs(v_pred - v_true), dim=0)  # [3,6]


# ---------------------------------------------------------------------
# 5) 小工具：Voigt 索引映射
# ---------------------------------------------------------------------
_VOIGT_NAME_to_IDX = {"xx": 0, "yy": 1, "zz": 2, "yz": 3, "xz": 4, "xy": 5}

def voigt_index_name(a: int) -> str:
    names = ["xx", "yy", "zz", "yz", "xz", "xy"]
    if not (0 <= a < 6):
        raise ValueError("Voigt column index must be in [0..5]")
    return names[a]


def jk_to_voigt_index(j: int, k: int) -> int:
    """
    将 (j,k)（0..2）映射到 Voigt 列索引（0..5），按 {xx,yy,zz,yz,xz,xy}。
    """
    pair = tuple(sorted((j, k)))
    table = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (0, 2): 4, (0, 1): 5}
    if pair not in table:
        raise ValueError("j,k must be in 0..2")
    return table[pair]
