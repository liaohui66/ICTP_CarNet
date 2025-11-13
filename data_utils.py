# data_utils.py  —— Cartesian-first（兼容 3x6/3x3x3 输入），y 存 [1,3,3,3]

import torch
import json
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import neighbor_list
from ase.atoms import Atom
from ase.data import atomic_numbers
from torch_geometric.data import Data
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Tuple, Sequence
import torch.nn.functional as F

# ---------------- 0. 小工具：Voigt<->Cartesian(秩3) ----------------
def _voigt_to_cartesian_rank3(v: torch.Tensor, shear_factor: float = 1.0) -> torch.Tensor:
    """
    v: [..., 3,6] -> e: [..., 3,3,3]
    Voigt columns: [xx, yy, zz, yz, xz, xy]
    """
    assert v.shape[-2:] == (3, 6), f"expected (...,3,6), got {v.shape}"
    *batch, _, _ = v.shape
    e = torch.zeros(*batch, 3, 3, 3, device=v.device, dtype=v.dtype)
    e[..., :, 0, 0] = v[..., :, 0]
    e[..., :, 1, 1] = v[..., :, 1]
    e[..., :, 2, 2] = v[..., :, 2]
    e[..., :, 1, 2] = e[..., :, 2, 1] = v[..., :, 3] * shear_factor
    e[..., :, 0, 2] = e[..., :, 2, 0] = v[..., :, 4] * shear_factor
    e[..., :, 0, 1] = e[..., :, 1, 0] = v[..., :, 5] * shear_factor
    return e

def _to_cartesian_tensor(piezo_tensor_target: Sequence, dtype: torch.dtype) -> torch.Tensor:
    """
    接受 3x6(Voigt) 或 3x3x3(Cartesian)；统一返回 3x3x3(Cartesian).
    """
    t = torch.tensor(piezo_tensor_target, dtype=dtype)
    if t.shape == (3, 6) or t.numel() == 18:
        t = t.view(3, 6)
        return _voigt_to_cartesian_rank3(t)
    if t.shape == (3, 3, 3) or t.numel() == 27:
        return t.view(3, 3, 3)
    raise ValueError(f"Unsupported piezo tensor shape {tuple(t.shape)}; expect (3,6) or (3,3,3).")

# --- 1. Atom Feature Initialization ---
_ATOM_FEATURE_DIM: int = -1
def initialize_atom_features(num_elements: int = 89) -> Tuple[None, int]:
    global _ATOM_FEATURE_DIM
    if _ATOM_FEATURE_DIM != -1:
        return None, _ATOM_FEATURE_DIM
    print(f"Initializing one-hot atom features for {num_elements} elements.")
    _ATOM_FEATURE_DIM = num_elements
    print(f"Atom feature dimension set to: {_ATOM_FEATURE_DIM}")
    return None, _ATOM_FEATURE_DIM

# --- 2. Helper for radial cutoff ---
def r_cut2D(radial_cutoff_base: float, ase_atoms_obj: Atom) -> float:
    cell_matrix = ase_atoms_obj.get_cell(complete=True).array
    if not ase_atoms_obj.pbc.any():
        return radial_cutoff_base
    norms = [np.linalg.norm(cell_matrix[i]) for i in range(3) if np.linalg.norm(cell_matrix[i]) > 1e-6]
    if not norms:
        return radial_cutoff_base
    return max(max(norms), radial_cutoff_base)

# --- 3. 结构->PyG（y 统一为 Cartesian 3x3x3） ---
def create_pyg_data(
    pymatgen_structure: Structure,
    piezo_tensor_target: Any,   # 允许 3x6 或 3x3x3
    num_elements: int,
    radial_cutoff: float,
    dtype: torch.dtype,
) -> Data:
    ase_atoms = AseAtomsAdaptor.get_atoms(pymatgen_structure)

    # Node features
    atomic_nums = torch.tensor([atomic_numbers[s] for s in ase_atoms.get_chemical_symbols()], dtype=torch.long)
    if torch.any(atomic_nums <= 0):
        raise ValueError("Atomic numbers must be positive integers.")
    type_indices = atomic_nums - 1
    max_index = int(type_indices.max().item())
    if max_index >= num_elements:
        offending = [int(z.item()) for z in atomic_nums if (int(z.item()) - 1) >= num_elements]
        raise ValueError(f"Encountered atomic numbers {offending} outside supported range (<= {num_elements}).")
    x = F.one_hot(type_indices, num_classes=num_elements).to(dtype)

    # Positions & lattice
    pos = torch.tensor(ase_atoms.get_positions(), dtype=dtype)
    lattice = torch.tensor(ase_atoms.get_cell(complete=True).array, dtype=dtype)

    # Edges & shifts
    if len(ase_atoms) > 1:
        effective_cutoff = r_cut2D(radial_cutoff, ase_atoms)
        ase_atoms_for_nl = ase_atoms.copy()
        ase_atoms_for_nl.set_pbc(True)
        edge_src, edge_dst, edge_shift_raw = neighbor_list(
            "ijS", a=ase_atoms_for_nl, cutoff=effective_cutoff, self_interaction=False
        )
        edge_index = torch.stack([torch.from_numpy(edge_src), torch.from_numpy(edge_dst)], dim=0).long()
        edge_shift = torch.from_numpy(edge_shift_raw).to(dtype)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_shift = torch.empty((0, 3), dtype=dtype)

    # --- 目标：统一成 3x3x3，并存成 [1,3,3,3] ---
    y_cart = _to_cartesian_tensor(piezo_tensor_target, dtype=dtype)  # [3,3,3]
    y = y_cart.unsqueeze(0)  # [1,3,3,3]

    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        cell=lattice,
        shifts=edge_shift,
        y=y,
        z=type_indices,
    )
    # 兼容字段
    data.positions = pos
    data.atomic_numbers = atomic_nums
    data.node_attrs = x
    data.pbc = torch.tensor([True, True, True], dtype=torch.bool)
    return data

# --- 4. JSON -> List[Data] ---
def load_piezo_dataset(
    json_file_path: str,
    structure_key: str,
    piezo_tensor_key: str,   # 既可指向 Voigt(3x6) 列表，也可指向 Cartesian(3x3x3) 列表
    radial_cutoff: float,
    num_elements: int = 89,
    limit_n: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> List[Data]:
    _, num_elements_dim = initialize_atom_features(num_elements)

    print(f"Loading JSON file: {json_file_path}")
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"FATAL: Could not read or parse JSON file '{json_file_path}'. Error: {e}")
        return []

    if structure_key not in full_data or piezo_tensor_key not in full_data:
        raise KeyError(
            f"Required keys '{structure_key}' or '{piezo_tensor_key}' not found in JSON. "
            f"Available keys: {list(full_data.keys())}"
        )

    structure_dicts = full_data[structure_key]
    piezo_tensors = full_data[piezo_tensor_key]

    if len(structure_dicts) != len(piezo_tensors):
        print(f"Warning: Mismatch in number of structures ({len(structure_dicts)}) and tensors ({len(piezo_tensors)}).")

    num_to_process = min(len(structure_dicts), len(piezo_tensors))
    if limit_n is not None and limit_n >= 0:
        num_to_process = min(num_to_process, limit_n)

    dataset: List[Data] = []
    print(f"Converting {num_to_process} entries to PyG Data objects...")
    for i in tqdm(range(num_to_process)):
        try:
            pmg_struct = Structure.from_dict(structure_dicts[i])
            piezo_target = piezo_tensors[i]  # could be 3x6 or 3x3x3

            data = create_pyg_data(
                pymatgen_structure=pmg_struct,
                piezo_tensor_target=piezo_target,
                num_elements=num_elements_dim,
                radial_cutoff=radial_cutoff,
                dtype=dtype,
            )
            dataset.append(data)
        except Exception as e:
            mat_id = structure_dicts[i].get("material_id", f"index_{i}")
            print(f"Skipping entry {mat_id} due to an error: {e}")

    return dataset

# --- 5. PiezoTensorScaler —— 3x3x3 Cartesian ---
class PiezoTensorScaler:
    """
    Standardize 3x3x3 Cartesian piezoelectric tensors component-wise.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.mean: Optional[torch.Tensor] = None  # [3,3,3]
        self.std: Optional[torch.Tensor] = None   # [3,3,3]
        self.epsilon = epsilon

    def fit(self, dataset: List[Data]):
        if not dataset:
            raise ValueError("Cannot fit scaler on an empty dataset.")

        print("Fitting PiezoTensorScaler on the training dataset (Cartesian 3x3x3, component-wise)...")

        ys = []
        for data in dataset:
            if not hasattr(data, "y"):
                raise AttributeError("Data object in dataset is missing the 'y' attribute.")
            y = data.y
            # 期望 [1,3,3,3] 或 [3,3,3]
            if y.dim() == 4 and y.size(0) == 1:
                y = y.squeeze(0)
            if y.shape != (3, 3, 3):
                raise ValueError(f"Expect y as 3x3x3 Cartesian; got {tuple(y.shape)}")
            ys.append(y)

        all_y = torch.stack(ys, dim=0)  # [N,3,3,3]
        self.mean = all_y.mean(dim=0)   # [3,3,3]
        self.std = all_y.std(dim=0)     # [3,3,3]
        self.std[torch.abs(self.std) < self.epsilon] = 1.0

        print("Scaler fitted successfully.")
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call .fit(dataset) first.")
        mean = self.mean.to(data.device, dtype=data.dtype)
        std = self.std.to(data.device, dtype=data.dtype)
        return (data - mean) / std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call .fit(dataset) first.")
        mean = self.mean.to(data.device, dtype=data.dtype)
        std = self.std.to(data.device, dtype=data.dtype)
        return data * std + mean

    def save(self, filepath: str):
        if self.mean is None or self.std is None:
            print("Warning: Trying to save an unfitted scaler.")
            return
        torch.save({"mean": self.mean, "std": self.std}, filepath)

    def load(self, filepath: str):
        state = torch.load(filepath, map_location="cpu")
        self.mean = state["mean"]
        self.std = state["std"]
        return self

if __name__ == "__main__":
    pass
