# models/ictp_backbone.py
import torch
import torch.nn as nn
import torch_scatter

# =========================
# ST projectors (Cartesian NAT) for l=1,2,3
# =========================
def st_rank1(u):  # (..., 3)
    return u

def st_rank2(M):  # (..., 3,3)
    trace = torch.einsum("...ii->...", M)
    I = torch.eye(3, device=M.device, dtype=M.dtype)
    return M - trace[..., None, None] * I / 3.0

def symmetrize3(T):  # full sym over (i,j,k)
    # six permutations over the last three axes (i,j,k)
    return (
        T
        + T.transpose(-1, -2)              # ...ijk -> ...ikj
        + T.transpose(-3, -2)              # ...ijk -> ...jik
        + T.transpose(-3, -1)              # ...ijk -> ...kji
        + T.transpose(-2, -1).transpose(-3, -2)  # ...ijk -> ...jki
        + T.transpose(-2, -1).transpose(-3, -1)  # ...ijk -> ...kij
    ) / 6.0

def st_rank3(T):  # (..., 3,3,3) fully sym & traceless
    Ts = symmetrize3(T)
    I = torch.eye(3, device=T.device, dtype=T.dtype)
    # c_i = Ts_{i q q}
    c = torch.einsum("...ijj->...i", Ts)
    # (δ_jk c_i + δ_ik c_j + δ_ij c_k) / 5
    term = (
        torch.einsum("ab,...i->...iab", I, c)   # δ_jk c_i
        + torch.einsum("ab,...j->...jab", I, c) # δ_ik c_j
        + torch.einsum("ab,...k->...abk", I, c) # δ_ij c_k
    ) / 5.0
    return Ts - term

# =========================
# Edge irreps bases U^(l)(rhat)
# =========================
def edge_basis(rij):
    r = torch.norm(rij, dim=-1, keepdim=True) + 1e-9
    u = rij / r
    U1 = st_rank1(u)  # [E,3]
    U2 = st_rank2(torch.einsum("...i,...j->...ij", u, u))         # [E,3,3]
    U3 = st_rank3(torch.einsum("...i,...j,...k->...ijk", u, u, u))# [E,3,3,3]
    return (U1, U2, U3), r.squeeze(-1)

# =========================
# Helper: outer & ST project for any l1 x l2 -> l3 (Cartesian)
# =========================
def irrep_dims(l):  # Cartesian rank (number of indices)
    return {0: (), 1: (3,), 2: (3,3), 3: (3,3,3)}[l]

def st_project_l(l, T):
    if l == 0:
        return T
    if l == 1:
        return st_rank1(T)
    if l == 2:
        return st_rank2(T)
    if l == 3:
        return st_rank3(T)
    raise NotImplementedError

def natural_product(A, l1, B, l2, l3):
    """
    A: (..., *dims(l1)), B: (..., *dims(l2)) -> ST-projected to l3
    Only supports results with l3 <= 3.
    """
    if l1 == 1 and l2 == 1:
        if l3 == 0:
            return torch.einsum("...i,...i->...", A, B)
        if l3 == 1:
            return torch.cross(A, B, dim=-1)
        if l3 == 2:
            T = 0.5 * (torch.einsum("...i,...j->...ij", A, B) + torch.einsum("...j,...i->...ij", A, B))
            return st_rank2(T)
        return None

    if (l1 == 1 and l2 == 2) or (l1 == 2 and l2 == 1):
        vec, ten = (A, B) if l1 == 1 else (B, A)
        if l3 == 1:
            return torch.einsum("...jk,...k->...j", ten, vec)
        if l3 == 2:
            v = torch.einsum("...jk,...k->...j", ten, vec)
            T = torch.einsum("...i,...j->...ij", vec, v)
            T = 0.5 * (T + T.transpose(-1, -2))
            return st_rank2(T)
        if l3 == 3:
            T = torch.einsum("...i,...jk->...ijk", vec, ten)
            return st_rank3(T)
        return None

    if l1 == 0:
        T = B
    elif l2 == 0:
        T = A
    else:
        T = None
        if l1 == 2 and l2 == 2:
            T4 = torch.einsum("...ij,...kl->...ijkl", A, B)  # rank-4
            if l3 == 2:
                # trace over the last two indices -> rank-2
                T = torch.einsum("...ijkk->...ij", T4)
            elif l3 == 0:
                # double trace -> scalar
                T = torch.einsum("...iijj->...", T4)
            else:
                return None
        elif (l1 == 1 and l2 == 3) or (l1 == 3 and l2 == 1):
            # 1x3 -> 2 (keep l3<=3)
            if l3 == 3:
                return None
            V1, T3 = (A, B) if l1 == 1 else (B, A)
            T4 = torch.einsum("...i,...jkl->...ijkl", V1, T3)  # rank-4
            # contract last two to get rank-2
            T = torch.einsum("...ijkk->...ij", T4)
        elif (l1 == 3 and l2 == 3):
            # 3x3 -> 0,2,4,6; we only keep 0 or 2 via traces; rank-6 outer too large, skip
            return None
        else:
            return None

    return st_project_l(l3, T) if T is not None else None

class RadialMLP(nn.Module):
    def __init__(self, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, r):  # r: [E] or [E,1]
        if r.dim() == 1:
            r = r[:, None]
        return self.net(r)  # [E,out_dim]

# =========================
# ICTP/CarNet-like equivariant block with ALL l1xl2->l3 paths (l<=3)
# =========================
class ICTPBlock(nn.Module):
    """
    feats[l] shapes:
      l=0: [N, c0]
      l=1: [N, c1, 3]
      l=2: [N, c2, 3,3]  (ST)
      l=3: [N, c3, 3,3,3] (ST)
    """
    def __init__(self, c0=32, c1=16, c2=8, c3=4):
        super().__init__()
        self.c0, self.c1, self.c2, self.c3 = c0, c1, c2, c3
        self.rad = nn.ModuleDict()
        self.scalar_gate = nn.ModuleDict()
        for l1 in range(0, 4):
            for l2 in range(0, 4):
                for l3 in range(abs(l1 - l2), min(l1 + l2, 3) + 1):
                    key = f"{l1}{l2}{l3}"
                    self.rad[key] = RadialMLP(out_dim=getattr(self, f"c{l3}"))
                    if l1 == 0 and l2 >= 1:
                        out_dim = getattr(self, f"c{l3}")
                        self.scalar_gate[key] = nn.Sequential(
                            nn.LayerNorm(c0),
                            nn.Linear(c0, c0),
                            nn.SiLU(),
                            nn.Linear(c0, out_dim),
                        )

        inv_dim = c0 + c1 + c2 + c3
        self.gate = nn.Sequential(
            nn.Linear(inv_dim, 2 * inv_dim), nn.SiLU(),
            nn.Linear(2 * inv_dim, inv_dim), nn.Sigmoid()
        )

    def forward(self, pos, edge_index, feats):
        # ---- 修正方向：dst = receiver = i，src = sender = j
        dst = edge_index[0]  # i
        src = edge_index[1]  # j
        rij = pos[dst] - pos[src]  # r_ij = r_i - r_j  (j->i)
        (U1, U2, U3), r = edge_basis(rij)
        U = {1: U1, 2: U2, 3: U3}

        N = pos.size(0)
        msg0 = torch.zeros(N, self.c0, device=pos.device, dtype=pos.dtype)
        msg1 = torch.zeros(N, self.c1, 3,  device=pos.device, dtype=pos.dtype)
        msg2 = torch.zeros(N, self.c2, 3,3,  device=pos.device, dtype=pos.dtype)
        msg3 = torch.zeros(N, self.c3, 3,3,3,device=pos.device, dtype=pos.dtype)

        for l1 in range(0, 4):
            X = feats[l1]
            if l1 == 0:
                scalar_src = X[src]             # [E,c0]
            elif l1 == 1:
                Xsrc = X[src]                   # [E,c1,3]
            elif l1 == 2:
                Xsrc = X[src]                   # [E,c2,3,3]
            else:
                Xsrc = X[src]                   # [E,c3,3,3,3]

            for l2 in range(1, 4):              # 这里从1开始：只用方向基 U^l, 不引入额外标量边基
                B = U[l2]                       # [E, *dims(l2)]
                for l3 in range(abs(l1 - l2), min(l1 + l2, 3) + 1):
                    key = f"{l1}{l2}{l3}"
                    alpha = self.rad[key](r)    # [E, c(l3)]

                    if l1 == 0:
                        gate = self.scalar_gate[key](scalar_src).to(
                            dtype=alpha.dtype, device=alpha.device
                        )  # [E,c(l3)]
                        alpha = alpha + 0.1 * torch.tanh(gate)
                        base = B                # 无依赖源通道
                    else:
                        base = Xsrc.mean(dim=1) # 简单的通道平均，可替换为更强的 1x1 mixing
                        if l2 == 1 and l1 == 1 and l3 == 0:
                            base = torch.einsum("...i,...i->...", base, B)
                        else:
                            base = natural_product(base, l1, B, l2, l3)
                        if base is None:
                            continue

                    if l3 == 0:
                        scalar = base.unsqueeze(-1) if base.dim() == 1 else base  # [E,1] or [E]
                        msg0 = torch_scatter.scatter_add(alpha * scalar, dst, out=msg0, dim=0)
                    elif l3 == 1:
                        M = alpha[..., None] * base[:, None, :]                   # [E,c1,3]
                        msg1 = torch_scatter.scatter_add(M, dst, out=msg1, dim=0)
                    elif l3 == 2:
                        M = alpha[..., None, None] * base[:, None, :, :]          # [E,c2,3,3]
                        msg2 = torch_scatter.scatter_add(M, dst, out=msg2, dim=0)
                    else:
                        M = alpha[..., None, None, None] * base[:, None, :, :, :] # [E,c3,3,3,3]
                        msg3 = torch_scatter.scatter_add(M, dst, out=msg3, dim=0)

        # gates based on invariants of current feats
        inv0 = torch.sqrt((feats[0]**2).mean(dim=1) + 1e-12)                  # [N]
        inv1 = torch.sqrt((feats[1]**2).mean(dim=(1,2)) + 1e-12)              # [N]
        inv2 = torch.sqrt((feats[2]**2).mean(dim=(1,2,3)) + 1e-12)            # [N]
        inv3 = torch.sqrt((feats[3]**2).mean(dim=(1,2,3,4)) + 1e-12)          # [N]
        inv = torch.stack([inv0, inv1, inv2, inv3], dim=1)                     # [N,4]
        inv_expanded = torch.cat([
            inv[:, [0]].repeat(1, self.c0),
            inv[:, [1]].repeat(1, self.c1),
            inv[:, [2]].repeat(1, self.c2),
            inv[:, [3]].repeat(1, self.c3),
        ], dim=1)  # [N, c0+c1+c2+c3]

        gates = self.gate(inv_expanded)  # [N, c_sum]
        p0, p1, p2, p3 = 0, self.c0, self.c0 + self.c1, self.c0 + self.c1 + self.c2

        out0 = feats[0] + msg0 * gates[:, p0:p1]                              # [N,c0]
        out1 = feats[1] + msg1 * gates[:, p1:p2][..., None]                   # [N,c1,3]
        out2 = feats[2] + msg2 * gates[:, p2:p3][..., None, None]             # [N,c2,3,3]
        out3 = feats[3] + msg3 * gates[:, p3:][..., None, None, None]         # [N,c3,3,3,3]

        # keep l=2,3 strictly ST to avoid drift
        out2 = st_rank2(out2)
        out3 = st_rank3(out3)
        return {0: out0, 1: out1, 2: out2, 3: out3}

# =========================
# Backbone stack + pooling -> graph rank-3 feature
# =========================
class ICTPBackbone(nn.Module):
    """
    Embedding -> K x ICTPBlock (l<=3, all paths) -> pool -> rank-3 graph feature
    """
    def __init__(self, num_elements=100, c0=32, c1=16, c2=8, c3=4, n_layers=3):
        super().__init__()
        self.embed = nn.Embedding(num_elements, c0)
        self.blocks = nn.ModuleList([ICTPBlock(c0=c0, c1=c1, c2=c2, c3=c3) for _ in range(n_layers)])
        self.c1, self.c2, self.c3 = c1, c2, c3

    def forward(self, pos, z, edge_index, batch):
        N = z.size(0)
        feats = {
            0: self.embed(z),                         # [N, c0]
            1: torch.zeros(N, self.c1, 3,  device=pos.device, dtype=pos.dtype),
            2: torch.zeros(N, self.c2, 3,3,  device=pos.device, dtype=pos.dtype),
            3: torch.zeros(N, self.c3, 3,3,3,device=pos.device, dtype=pos.dtype),
        }
        for blk in self.blocks:
            feats = blk(pos, edge_index, feats)

        # Graph-level rank-3: average over multiplicity then pool nodes
        h3 = feats[3].mean(dim=1)  # [N,3,3,3]
        g3 = torch_scatter.scatter_mean(h3, batch, dim=0)  # [B,3,3,3]
        return g3
