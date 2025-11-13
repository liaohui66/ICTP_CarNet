# utils/slsi.py
import torch
from torch_scatter import scatter_add, scatter_mean

# Minimal Pauling electronegativity table; unknown -> 0.0 (harmless fallback)
PAULING = {
    1:2.20, 3:0.98, 4:1.57, 5:2.04, 6:2.55, 7:3.04, 8:3.44, 9:3.98,
    11:0.93,12:1.31,13:1.61,14:1.90,15:2.19,16:2.58,17:3.16,
    19:0.82,20:1.00,22:1.54,23:1.63,24:1.66,25:1.55,26:1.83,
    28:1.91,29:1.90,30:1.65,31:1.81,32:2.01,33:2.18,34:2.55,35:2.96,
    37:0.82,38:0.95,39:1.22,40:1.33,41:1.60,42:2.16,43:1.90,44:2.20,
    45:2.28,46:2.20,47:1.93,48:1.69,49:1.78,50:1.96,51:2.05,52:2.10,
    53:2.66,55:0.79,56:0.89,72:1.33,73:1.60,74:2.36,75:1.90,76:2.20,
    77:2.20,78:2.28,79:2.54,80:2.00,81:1.62,82:2.33,83:2.02,
}
MAX_CHI_DIFF = 3.3  # ~F-Cs span for normalization

@torch.no_grad()
def slsi_from_batch(batch, power: int = 4, eps: float = 1e-9):
    """
    Compute graph-level features:
      gfeat: (B,4) = [SLSI, |p_bar|, anisotropy, signed_SLSI]
      n_pol: (B,3) main polar axis (unit vector)
    Requirements in `batch`: pos (N,3), z (N,), edge_index (2,E), batch (N,)
    Optional: edge_vec (E,3) shortest-image displacement (PBC). If missing, uses pos[j]-pos[i].
    """
    pos, Z, edge_index, bidx = batch.pos, batch.z, batch.edge_index, batch.batch
    device = pos.device
    chi = torch.tensor([PAULING.get(int(z), 0.0) for z in Z.tolist()], dtype=torch.float32, device=device)

    i, j = edge_index
    if hasattr(batch, "edge_vec") and batch.edge_vec is not None:
        rij = batch.edge_vec
    else:
        rij = pos[j] - pos[i]  # fallback; make sure your edges already obey shortest image if PBC
    r = torch.linalg.norm(rij, dim=-1).clamp_min(1e-6)
    rhat = rij / r.unsqueeze(-1)

    dchi = (chi[j] - chi[i]).abs() / MAX_CHI_DIFF
    w = (1.0 / r).pow(power)

    # local polarization p_i and graph mean p_bar
    pi = scatter_add((w * dchi).unsqueeze(-1) * rhat, i, dim=0, dim_size=pos.size(0))
    p_bar = scatter_mean(pi, bidx, dim=0)             # (B,3)
    p_norm = torch.linalg.norm(p_bar, dim=-1)         # (B,)
    n_pol = p_bar / (p_norm.unsqueeze(-1) + eps)      # (B,3)

    # A = sum w dchi rhat rhat^T (per graph)
    outer = rhat.unsqueeze(-1) * rhat.unsqueeze(-2)   # (E,3,3)
    A = scatter_add((w * dchi).unsqueeze(-1).unsqueeze(-1) * outer,
                    bidx[i], dim=0, dim_size=p_bar.size(0))  # (B,3,3)

    # signed SLSI along n_pol
    n_pol_e = n_pol[bidx[i]]                          # (E,3)
    proj = (rhat * n_pol_e).sum(-1)                   # (E,)
    num = scatter_add(w * dchi * proj, bidx[i], dim=0, dim_size=p_bar.size(0))
    den = scatter_add(w * dchi,       bidx[i], dim=0, dim_size=p_bar.size(0)).clamp_min(eps)
    slsi_signed = num / den
    slsi = slsi_signed.abs()

    # anisotropy from eigvals of A
    evals = torch.linalg.eigvalsh(A)                  # (B,3)
    ani = (evals[:, -1] - evals[:, 0]) / (evals.sum(-1).clamp_min(eps))

    # backup axis when |p_bar| very small: use principal eigenvector of A
    need = (p_norm < 1e-4)
    if need.any():
        v = torch.randn_like(p_bar)
        for _ in range(3):
            v = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
            v = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)
        n_pol[need] = v[need]
        n_pol_e = n_pol[bidx[i]]
        proj = (rhat * n_pol_e).sum(-1)
        num = scatter_add(w * dchi * proj, bidx[i], dim=0, dim_size=p_bar.size(0))
        slsi_signed = num / den
        slsi = slsi_signed.abs()

    gfeat = torch.stack([slsi, p_norm, ani, slsi_signed], dim=-1)  # (B,4)
    return {"gfeat": gfeat, "n_pol": n_pol}
