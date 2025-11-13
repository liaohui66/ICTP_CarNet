# models/projector.py
import torch
import torch.nn as nn

def symmetrize_last_two(e: torch.Tensor) -> torch.Tensor:
    # Hard-enforce j<->k symmetry on the output e_{ijk}.
    return 0.5 * (e + e.transpose(-2, -1))

# --- Strict rank-3 symmetrization and ST projector ---

def _symmetrize_rank3_full(T: torch.Tensor) -> torch.Tensor:
    """Symmetrize a rank-3 tensor over (i,j,k) by averaging the 6 permutations."""
    return (
        T
        + T.permute(0, 2, 1)  # (i,k,j)
        + T.permute(1, 0, 2)  # (j,i,k)
        + T.permute(1, 2, 0)  # (j,k,i)
        + T.permute(2, 0, 1)  # (k,i,j)
        + T.permute(2, 1, 0)  # (k,j,i)
    ) / 6.0

def _st_rank3_strict(T: torch.Tensor) -> torch.Tensor:
    """Symmetric-traceless projector for rank-3 tensors (on the last three indices)."""
    Ts = _symmetrize_rank3_full(T)
    I = torch.eye(3, device=T.device, dtype=T.dtype)
    # c_i = Ts_{i q q}
    c = torch.einsum("ijj->i", Ts)
    term = (
        torch.einsum("ab,i->iab", I, c) +
        torch.einsum("ab,j->ajb", I, c) +
        torch.einsum("ab,k->abk", I, c)
    ) / 5.0
    return Ts - term

# --- Build orthonormal physical bases: 7 (l=3) + 3 (l=1) = 10 columns ---

def build_rank3_bases(device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Build a constant set of basis tensors for the physical space (sym in j<->k):
      - l=3 block: 7 basis tensors S3_m (symmetric & traceless)
      - l=1 block: 3 basis tensors S1_p constructed from Kronecker deltas and unit vectors e_p
    Return: orthonormal basis matrix B_phys with shape [27, 10] (columns are vectorized bases).
    """
    ex = torch.tensor([1., 0., 0.], device=device, dtype=dtype)
    ey = torch.tensor([0., 1., 0.], device=device, dtype=dtype)
    ez = torch.tensor([0., 0., 1.], device=device, dtype=dtype)
    I  = torch.eye(3, device=device, dtype=dtype)

    # l=1 part: S1(v)_{ijk} = δ_jk v_i + δ_ij v_k + δ_ik v_j  (already sym in j<->k)
    def S1(v: torch.Tensor) -> torch.Tensor:
        return (
            torch.einsum("jk,i->ijk", I, v) +
            torch.einsum("ij,k->ijk", I, v) +
            torch.einsum("ik,j->ijk", I, v)
        )

    S1_list = [S1(ex), S1(ey), S1(ez)]  # 3 tensors

    # l=3 part: take ST(u ⊗ u ⊗ u) for several directions u, then QR to get 7-dim subspace.
    def ST3_of(u: torch.Tensor) -> torch.Tensor:
        T = torch.einsum("i,j,k->ijk", u, u, u)
        return _st_rank3_strict(T)

    u_list = [
        ex, -ex, ey, -ey, ez, -ez,
        (ex + ey) / torch.sqrt(torch.tensor(2., device=device, dtype=dtype)),
        (ex + ez) / torch.sqrt(torch.tensor(2., device=device, dtype=dtype)),
        (ey + ez) / torch.sqrt(torch.tensor(2., device=device, dtype=dtype)),
    ]
    S3_raw = [ST3_of(u) for u in u_list]                       # 9 seeds
    M = torch.stack([t.reshape(-1) for t in S3_raw], dim=1)    # [27, 9]
    Q, _ = torch.linalg.qr(M, mode="reduced")                  # Q: [27, r], r<=9
    r = Q.size(1)
    if r < 7:
        raise RuntimeError(
            f"[projector] rank-3 (l=3) basis rank={r} < 7; "
            "enrich u_list or check dtype/device."
        )
    Q = Q[:, :7]                                               # keep first 7 columns
    S3_list = [Q[:, i].reshape(3, 3, 3) for i in range(7)]     # 7 tensors

    # Concatenate (7 + 3) = 10, then orthonormalize again for stability
    cols = [t.reshape(-1) for t in S3_list + S1_list]          # list of 10 length-27 vectors
    B_phys = torch.stack(cols, dim=1)                          # [27, 10]
    Q2, _ = torch.linalg.qr(B_phys, mode="reduced")            # [27, 10]
    if Q2.size(1) != 10:
        raise RuntimeError(f"[projector] physical basis rank < 10; got {Q2.size(1)}.")
    return Q2  # final orthonormal columns

class CarNetProjector(nn.Module):
    """
    CarNet-style projector for rank-3 with j<->k symmetry.
    Input coeffs: concat([c^(l=3) (7 dims), c^(l=1) (3 dims)]) -> 10-dim
    Output: e_{ijk} [B,3,3,3] with hard symmetrization on (j,k).
    The basis is built once with QR and stored as a frozen Linear's weight.
    """
    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        B = build_rank3_bases(device=device, dtype=dtype)      # [27, 10]
        self.register_buffer("B", B)                           # for inspection & checkpoint
        self.linear = nn.Linear(10, 27, bias=False)
        with torch.no_grad():
            # PyTorch Linear: y = x @ W^T, so set W = B to compute flat = coeffs @ B^T.
            self.linear.weight.copy_(B)
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, c3: torch.Tensor, c1: torch.Tensor) -> torch.Tensor:
        coeffs = torch.cat([c3, c1], dim=-1)  # [B, 10]

        # ★ 自动对齐到 Linear 权重的 dtype/device，防 dtype/device 不一致
        W = self.linear.weight
        coeffs = coeffs.to(dtype=W.dtype, device=W.device)

        flat = self.linear(coeffs)            # [B, 27]
        e = flat.view(-1, 3, 3, 3)
        e = symmetrize_last_two(e)
        return e
