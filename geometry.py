# utils/geometry.py
import torch

def random_rotation_matrices(batch_size: int, device=None, dtype=None):
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    A = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    diag = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    Q = Q @ torch.diag_embed(diag)
    det = torch.linalg.det(Q).unsqueeze(-1).unsqueeze(-1)
    Q = torch.where(det < 0, Q * torch.tensor([[[1,0,0],[0,1,0],[0,0,-1]]], device=device, dtype=dtype), Q)
    return Q

@torch.no_grad()
def rotate_batch_positions(batch, R: torch.Tensor):
    pos = batch.pos
    b = batch.batch
    rotated = torch.empty_like(pos)
    for i in range(R.size(0)):
        mask = (b == i)
        if mask.any():
            rotated[mask] = pos[mask] @ R[i].T
    batch.pos = rotated
    batch.positions = rotated
    return batch
