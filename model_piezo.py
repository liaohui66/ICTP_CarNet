# models/model_piezo.py
import math
import torch
import torch.nn as nn

from .ictp_backbone import ICTPBackbone
from .projector import CarNetProjector
from .losses import cartesian_to_voigt_e


class PiezoEnd2End(nn.Module):
    """
    ICTP/CarNet faithful:
      - Backbone yields a graph-level rank-3 feature G3 (ST)
      - Split G3 into irreps coefficients: (l=3) 7-dim + (l=1) 3-dim via learned heads
      - CarNetProjector maps (7+3) -> physical e_{ijk} (symmetric in j<->k)
    """

    def __init__(
        self,
        num_elements: int = 100,
        n_layers: int = 3,
        c0: int = 32,
        c1: int = 16,
        c2: int = 8,
        c3: int = 4,
        hidden: int = 128,
        device: str | torch.device = "cpu",
        output_mode: str = "cartesian",
        # 可选：微调初始化尺度，避免早期常数输出
        head_init_scale: float = 1e-3,
    ):
        super().__init__()

        # Backbone: 产生图级 rank-3（ST）特征
        self.backbone = ICTPBackbone(
            num_elements=num_elements, c0=c0, c1=c1, c2=c2, c3=c3, n_layers=n_layers
        )

        # 共享的非线性投影（27 -> hidden）
        self.pre = nn.Sequential(
            nn.Linear(27, hidden, bias=True),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden, bias=True),
            nn.SiLU(),
        )

        # 轻量专门化：分别给 l=3 与 l=1 做一层归一化（也方便后续扩展不同门控/正则）
        self.pre_l3 = nn.Sequential(nn.LayerNorm(hidden))
        self.pre_l1 = nn.Sequential(nn.LayerNorm(hidden))

        # 线性头：bias=False 更稳定；小范围初始化避免输出塌缩到常数
        self.head_l3 = nn.Linear(hidden, 7, bias=False)
        self.head_l1 = nn.Linear(hidden, 3, bias=False)

        # 注意：不要在 __init__ 固定 dtype；forward 动态对齐
        self.projector = CarNetProjector(device=device, dtype=torch.float32)

        assert output_mode in ("cartesian", "voigt"), "output_mode must be 'cartesian' or 'voigt'"
        self.output_mode = output_mode

        # 初始化
        self._init_weights(head_init_scale=head_init_scale)

        # 放到 device
        self.to(device)

        # projector dtype/device 缓存（供 _ensure_projector_dtype_device 使用）
        self.projector._cached_dtype = None
        self.projector._cached_device = None

    def _init_weights(self, head_init_scale: float = 1e-3):
        # pre MLP: Kaiming
        for m in self.pre.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # heads：小权重均匀初始化，bias 为 0（bias=False 已禁用）
        for m in [self.head_l3, self.head_l1]:
            nn.init.uniform_(m.weight, -head_init_scale, head_init_scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _ensure_projector_dtype_device(self, ref: torch.Tensor):
        """将 projector 递归对齐到 ref 的 dtype/device（覆盖线性层权重与注册 buffer）。"""
        want_dtype, want_device = ref.dtype, ref.device
        if getattr(self.projector, "_cached_dtype", None) != want_dtype or \
           getattr(self.projector, "_cached_device", None) != want_device:
            # 一次性递归转换所有参数与 buffers 的 dtype & device（包括 linear.weight、注册的 B）
            self.projector.to(device=want_device, dtype=want_dtype)
            self.projector._cached_dtype = want_dtype
            self.projector._cached_device = want_device

    def forward(self, batch, return_coeff: bool = False):
        """
        return:
          - if output_mode == 'cartesian': [B,3,3,3]
          - if output_mode == 'voigt'    : [B,3,6]
          - if return_coeff == True      : (pred, (c3, c1))
        """
        # 1) Backbone: 图级 rank-3（ST）特征
        G3 = self.backbone(batch.pos, batch.z, batch.edge_index, batch.batch)  # [B,3,3,3]
        B = G3.size(0)
        flat = G3.reshape(B, 27)

        # 2) 共享 MLP，再分叉两路
        h = self.pre(flat)
        h3 = self.pre_l3(h)
        h1 = self.pre_l1(h)

        # 3) l=3 & l=1 系数
        c3 = self.head_l3(h3)  # [B,7]
        c1 = self.head_l1(h1)  # [B,3]

        # 4) Projector：动态对齐 dtype/device，再映射到 e_{ijk}
        self._ensure_projector_dtype_device(G3)
        e_cart = self.projector(c3, c1)  # [B,3,3,3]

        if return_coeff:
            pred = e_cart if self.output_mode == "cartesian" else cartesian_to_voigt_e(e_cart)
            return pred, (c3, c1)

        # 5) 输出域
        if self.output_mode == "voigt":
            return cartesian_to_voigt_e(e_cart)
        return e_cart
