import torch
from models.projector import build_rank3_bases, CarNetProjector

B = build_rank3_bases("cpu", torch.float64)   # [27,10]
print("B^T B ≈ I ? ", torch.allclose(B.T @ B, torch.eye(10, dtype=B.dtype), atol=1e-10))

proj = CarNetProjector("cpu", torch.float64)
c3 = torch.randn(5, 7, dtype=torch.float64)
c1 = torch.randn(5, 3, dtype=torch.float64)
e  = proj(c3, c1)  # [5,3,3,3]
# j<->k symmetry
print("sym jk ok? ", torch.allclose(e, e.transpose(-2, -1), atol=1e-12))
