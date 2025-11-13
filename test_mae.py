# test_mae.py (fixed, Cartesian-aligned evaluation with proper shear handling)
import os
import csv
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from data_utils import load_piezo_dataset, PiezoTensorScaler
from models.model_piezo import PiezoEnd2End

# ---- Consistent Cartesian<->Voigt with shear_factor ----
@torch.no_grad()
def voigt_to_cartesian_e(v: torch.Tensor, shear_factor: float) -> torch.Tensor:
    """v: [B,3,6] -> e: [B,3,3,3] using columns [xx,yy,zz,yz,xz,xy]."""
    B = v.size(0)
    e = torch.zeros(B, 3, 3, 3, device=v.device, dtype=v.dtype)
    e[:, :, 0, 0] = v[:, :, 0]
    e[:, :, 1, 1] = v[:, :, 1]
    e[:, :, 2, 2] = v[:, :, 2]
    e[:, :, 1, 2] = e[:, :, 2, 1] = v[:, :, 3] * shear_factor
    e[:, :, 0, 2] = e[:, :, 2, 0] = v[:, :, 4] * shear_factor
    e[:, :, 0, 1] = e[:, :, 1, 0] = v[:, :, 5] * shear_factor
    return e

@torch.no_grad()
def cartesian_to_voigt_torch(e_cart: torch.Tensor, shear_factor: float) -> torch.Tensor:
    """
    Inverse of voigt_to_cartesian_e with same shear_factor.
    e_cart: [B,3,3,3] -> v: [B,3,6] (xx,yy,zz,yz,xz,xy).
    For engineering shear (shear_factor=0.5), we divide shear entries by 0.5.
    """
    B = e_cart.size(0)
    v = torch.zeros(B, 3, 6, device=e_cart.device, dtype=e_cart.dtype)
    v[:, :, 0] = e_cart[:, :, 0, 0]
    v[:, :, 1] = e_cart[:, :, 1, 1]
    v[:, :, 2] = e_cart[:, :, 2, 2]
    v[:, :, 3] = e_cart[:, :, 1, 2] / shear_factor
    v[:, :, 4] = e_cart[:, :, 0, 2] / shear_factor
    v[:, :, 5] = e_cart[:, :, 0, 1] / shear_factor
    return v

def ensure_batch(batch):
    # Align with training helpers
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

@torch.no_grad()
def evaluate(
    model: PiezoEnd2End,
    loader,
    scaler_y: PiezoTensorScaler,
    device: torch.device,
    output_mode: str,
    shear_factor: float,
) -> Tuple[float, float]:
    """
    If output_mode == 'cartesian': compare in Cartesian domain.
    If output_mode == 'voigt':    convert true Cartesian->Voigt (same shear_factor) and compare in Voigt.
    test_set y is stored normalized; we inverse-transform to original units for metrics.
    """
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    absolute_mae = 0.0
    element_count = 0

    for batch in loader:
        batch = batch.to(device)
        batch = ensure_batch(batch)

        pred = model(batch)
        # y: [B,1,3,3,3]; make it [B,3,3,3] normalized
        y_norm = batch.y.squeeze(1)
        # back to original units (Cartesian)
        y_cart = scaler_y.inverse_transform(y_norm)

        if output_mode == "cartesian":
            loss = F.l1_loss(pred, y_cart)
            mae_sum = F.l1_loss(pred, y_cart, reduction="sum")
            elem_num = y_cart.numel()
        else:
            # model predicts [B,3,6] in Voigt; convert true to Voigt with the same shear convention
            y_voigt = cartesian_to_voigt_torch(y_cart, shear_factor=shear_factor)
            loss = F.l1_loss(pred, y_voigt)
            mae_sum = F.l1_loss(pred, y_voigt, reduction="sum")
            elem_num = y_voigt.numel()

        total_loss += loss.item() * pred.size(0)
        absolute_mae += mae_sum.item()
        element_count += elem_num
        total_graphs += pred.size(0)

    avg_loss = total_loss / max(1, total_graphs)
    mae = absolute_mae / max(1, element_count)
    return avg_loss, mae

def build_model(num_elements: int, device: torch.device, output_mode: str) -> PiezoEnd2End:
    # Mirror training hyperparameters
    model = PiezoEnd2End(
        num_elements=num_elements,
        n_layers=3,
        c0=32, c1=16, c2=8, c3=4,
        hidden=256,
        device=device,
        output_mode=output_mode,  # your model should honor this
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate best_equivariant_piezo model MAE on a test dataset")
    parser.add_argument("--ckpt", default=os.path.join("results_equivariant_piezo", "best_equivariant_piezo.pth"))
    parser.add_argument("--test_json", required=False, help="Path to test JSON file (same schema as training)")
    parser.add_argument("--structure_key", default=None)
    parser.add_argument("--piezo_key", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit_n", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_predictions", type=str, default=None, help="Optional CSV path to dump per-sample Voigt predictions.")
    # plotting controls
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_best", action="store_true")
    parser.add_argument("--plot_dir", default=os.path.join("results_equivariant_piezo", "test_plots"))
    parser.add_argument("--plot_predmax", action="store_true")
    parser.add_argument("--plot_voigt_comp", type=str, default=None)
    parser.add_argument("--plot_cart_comp", type=str, default=None)
    parser.add_argument("--nonzero_thresh", type=float, default=0.0)
    parser.add_argument("--filter_by", type=str, default="pred", choices=["pred", "true", "either"])
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})

    structure_key = args.structure_key or cfg.get("structure_key", "structure")
    piezo_key = args.piezo_key or cfg.get("piezo_key", "total")  # default to Cartesian key if present
    radial_cutoff = cfg.get("radial_cutoff", 5.0)
    num_elements = cfg.get("num_elements", 118)
    limit_n = args.limit_n if args.limit_n is not None else cfg.get("limit_n", None)
    output_mode = cfg.get("output_mode", "cartesian")
    shear_factor = cfg.get("shear_factor", 0.5)  # ★ keep consistent with training

    test_json = args.test_json or cfg.get("train_json")
    if not test_json:
        raise ValueError("Please specify --test_json or ensure checkpoint config contains 'train_json'.")

    # Load dataset (pass same shear_factor used during training)
    test_set = load_piezo_dataset(
        json_file_path=test_json,
        structure_key=structure_key,
        piezo_tensor_key=piezo_key,
        radial_cutoff=radial_cutoff,
        num_elements=num_elements,
        limit_n=limit_n,
        dtype=torch.float32,
    )
    if not test_set:
        raise RuntimeError("Test dataset loading failed or is empty.")

    # Restore scaler from checkpoint and normalize test y in-place (to mirror training)
    scaler = PiezoTensorScaler()
    saved_scaler = ckpt.get("scaler")
    if not saved_scaler or "mean" not in saved_scaler or "std" not in saved_scaler:
        raise RuntimeError("Checkpoint does not contain scaler statistics.")
    scaler.mean = saved_scaler["mean"].clone().float()
    scaler.std = saved_scaler["std"].clone().float()

    for data in test_set:
        # data.y: [1,3,3,3] Cartesian in original units -> normalize for consistency
        normalized = scaler.transform(data.y.squeeze(0))
        data.y = normalized.unsqueeze(0)

    # DataLoader
    try:
        from torch_geometric.loader import DataLoader as PyGDataLoader
    except ImportError as exc:
        raise RuntimeError("torch_geometric is required for loading datasets") from exc
    loader = PyGDataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    # Build and load model
    model = build_model(num_elements=num_elements, device=device, output_mode=output_mode)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    # Evaluate
    avg_loss, mae = evaluate(model, loader, scaler, device, output_mode, shear_factor)
    space = "voigt" if output_mode == "voigt" else "cartesian"
    print(f"Test Avg L1({space}): {avg_loss:.6f}")
    print(f"Test MAE({space}): {mae:.6f}")

    # ---------- Optional outputs / plotting ----------
    need_outputs = bool(
        args.save_predictions
        or args.plot
        or args.plot_best
        or args.plot_predmax
        or args.plot_voigt_comp
        or args.plot_cart_comp
    )

    true_tensor = None
    pred_tensor = None
    voigt_names_list = ["xx", "yy", "zz", "yz", "xz", "xy"]
    voigt_names_np = np.array(voigt_names_list)

    if need_outputs:
        all_true_voigt = []
        all_pred_voigt = []

        for batch in loader:
            batch = batch.to(device)
            batch = ensure_batch(batch)
            pred_any = model(batch)

            y_cart = scaler.inverse_transform(batch.y.squeeze(1))
            true_voigt = cartesian_to_voigt_torch(y_cart, shear_factor=shear_factor)
            pred_voigt = pred_any if output_mode == "voigt" else cartesian_to_voigt_torch(pred_any, shear_factor=shear_factor)

            all_true_voigt.append(true_voigt.detach().cpu())
            all_pred_voigt.append(pred_voigt.detach().cpu())

        true_tensor = torch.cat(all_true_voigt, dim=0)  # [N,3,6]
        pred_tensor = torch.cat(all_pred_voigt, dim=0)  # [N,3,6]

    if args.save_predictions:
        if pred_tensor is None or true_tensor is None:
            raise RuntimeError("Prediction tensors were not computed.")
        save_path = args.save_predictions
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_index", "i_index", "a_index", "a_name", "true_voigt", "pred_voigt"])
            N = pred_tensor.size(0)
            for sample_idx in range(N):
                for i in range(3):
                    for a in range(6):
                        writer.writerow([
                            sample_idx,
                            i,
                            a,
                            voigt_names_list[a],
                            float(true_tensor[sample_idx, i, a].item()),
                            float(pred_tensor[sample_idx, i, a].item()),
                        ])
        print(f"Saved predictions CSV: {os.path.abspath(save_path)}")

    if args.plot or args.plot_best or args.plot_predmax or args.plot_voigt_comp or args.plot_cart_comp:
        if pred_tensor is None or true_tensor is None:
            raise RuntimeError("Prediction tensors were not computed.")
        os.makedirs(args.plot_dir, exist_ok=True)

        # All-components scatter
        if args.plot:
            y_true = true_tensor.reshape(-1).numpy()
            y_pred = pred_tensor.reshape(-1).numpy()

            csv_path = os.path.join(args.plot_dir, "pred_vs_true_voigt_all.csv")
            np.savetxt(csv_path, np.column_stack([y_true, y_pred]), delimiter=",", header="true,pred", comments="")

            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 6), dpi=150)
                lim_min = float(min(y_true.min(), y_pred.min()))
                lim_max = float(max(y_true.max(), y_pred.max()))
                pad = 0.05 * (lim_max - lim_min + 1e-8)
                lo, hi = lim_min - pad, lim_max + pad
                plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
                plt.scatter(y_true, y_pred, s=6, alpha=0.3)
                plt.xlabel("True (Voigt components)")
                plt.ylabel("Pred (Voigt components)")
                plt.title("Pred vs True (Voigt 3x6, ALL components)")
                plt.xlim(lo, hi); plt.ylim(lo, hi); plt.grid(True, alpha=0.2); plt.legend(loc="best")
                out_png = os.path.join(args.plot_dir, "pred_vs_true_voigt_all.png")
                plt.tight_layout(); plt.savefig(out_png); plt.close()
                print(f"Saved scatter plot: {out_png}")
                print(f"Saved CSV: {csv_path}")
            except Exception as e:
                print(f"Plotting skipped: {e}")
                print(f"Saved CSV instead: {csv_path}")

        # Best-component scatter
        if args.plot_best:
            err = torch.mean(torch.abs(pred_tensor - true_tensor), dim=0)  # [3,6]
            err_np = err.numpy()
            idx_flat = int(err_np.argmin())
            i_best, a_best = divmod(idx_flat, 6)
            label = f"i={i_best}, a={voigt_names_list[a_best]}"
            comp_mae = float(err_np.min())

            y_true_c = true_tensor[:, i_best, a_best].numpy()
            y_pred_c = pred_tensor[:, i_best, a_best].numpy()

            csv_path = os.path.join(args.plot_dir, "pred_vs_true_voigt_best.csv")
            np.savetxt(csv_path, np.column_stack([y_true_c, y_pred_c]), delimiter=",", header="true,pred", comments="")

            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 6), dpi=150)
                lim_min = float(min(y_true_c.min(), y_pred_c.min())); lim_max = float(max(y_true_c.max(), y_pred_c.max()))
                pad = 0.05 * (lim_max - lim_min + 1e-8); lo, hi = lim_min - pad, lim_max + pad
                plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
                plt.scatter(y_true_c, y_pred_c, s=8, alpha=0.5)
                plt.xlabel(f"True (Voigt {label})"); plt.ylabel(f"Pred (Voigt {label})")
                plt.title(f"Best Component: {label}, MAE={comp_mae:.4f}")
                plt.xlim(lo, hi); plt.ylim(lo, hi); plt.grid(True, alpha=0.2)
                out_png = os.path.join(args.plot_dir, "pred_vs_true_voigt_best.png")
                plt.tight_layout(); plt.savefig(out_png); plt.close()
                print(f"Saved BEST-component scatter: {out_png}")
                print(f"Best component: {label}, MAE={comp_mae:.6f}")
                print(f"Saved CSV: {csv_path}")
            except Exception as e:
                print(f"Plotting skipped: {e}")
                print(f"Best component: {label}, MAE={comp_mae:.6f}")
                print(f"Saved CSV instead: {csv_path}")

        # Helper to parse Voigt spec
        def parse_voigt_spec(spec: str):
            spec = spec.strip()
            parts = [p.strip() for p in spec.replace("/", ",").split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError("--plot_voigt_comp format must be 'i,a'")
            i_s, a_s = parts
            i = int(i_s)
            name_map = {"xx":0, "yy":1, "zz":2, "yz":3, "xz":4, "xy":5}
            a = name_map.get(a_s.lower(), None)
            if a is None:
                a = int(a_s)
            if not (0 <= i <= 2 and 0 <= a <= 5):
                raise ValueError("i must be 0..2 and a must be 0..5")
            return i, a

        def jk_to_voigt_a(j: int, k: int) -> int:
            pair = tuple(sorted((j, k)))
            table = {(0,0):0, (1,1):1, (2,2):2, (1,2):3, (0,2):4, (0,1):5}
            if pair not in table:
                raise ValueError("j,k must be 0..2")
            return table[pair]

        # Plot a specific Voigt component
        if args.plot_voigt_comp:
            i_best, a_best = parse_voigt_spec(args.plot_voigt_comp)
            label = f"i={i_best}, a={voigt_names_list[a_best]}"
            y_true_c = true_tensor[:, i_best, a_best]
            y_pred_c = pred_tensor[:, i_best, a_best]

            if args.nonzero_thresh > 0:
                if args.filter_by == "pred":
                    mask = torch.abs(y_pred_c) > args.nonzero_thresh
                elif args.filter_by == "true":
                    mask = torch.abs(y_true_c) > args.nonzero_thresh
                else:
                    mask = (torch.abs(y_true_c) > args.nonzero_thresh) | (torch.abs(y_pred_c) > args.nonzero_thresh)
                y_true_c = y_true_c[mask]; y_pred_c = y_pred_c[mask]

            y_true_np = y_true_c.numpy(); y_pred_np = y_pred_c.numpy()
            csv_path = os.path.join(args.plot_dir, f"pred_vs_true_voigt_i{i_best}_a{voigt_names_list[a_best]}.csv")
            np.savetxt(csv_path, np.column_stack([y_true_np, y_pred_np]), delimiter=",", header="true,pred", comments="")

            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 6), dpi=150)
                lim_min = float(min(y_true_np.min(), y_pred_np.min())) if y_true_np.size and y_pred_np.size else -1.0
                lim_max = float(max(y_true_np.max(), y_pred_np.max())) if y_true_np.size and y_pred_np.size else 1.0
                pad = 0.05 * (lim_max - lim_min + 1e-8); lo, hi = lim_min - pad, lim_max + pad
                plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
                plt.scatter(y_true_np, y_pred_np, s=8, alpha=0.5)
                mae_comp = float(F.l1_loss(torch.tensor(y_true_np), torch.tensor(y_pred_np)).item()) if y_true_np.size else float('nan')
                plt.xlabel(f"True (Voigt {label})"); plt.ylabel(f"Pred (Voigt {label})")
                plt.title(f"Voigt component: {label}, MAE={mae_comp:.4f}")
                plt.xlim(lo, hi); plt.ylim(lo, hi); plt.grid(True, alpha=0.2)
                out_png = os.path.join(args.plot_dir, f"pred_vs_true_voigt_i{i_best}_a{voigt_names_list[a_best]}.png")
                plt.tight_layout(); plt.savefig(out_png); plt.close()
                print(f"Saved Voigt-component scatter: {out_png}")
                print(f"Saved CSV: {csv_path}")
            except Exception as e:
                print(f"Plotting specific Voigt comp skipped: {e}")
                print(f"Saved CSV instead: {csv_path}")

        # Plot a specific Cartesian e_ijk by mapping to Voigt index
        if args.plot_cart_comp:
            spec = [p.strip() for p in args.plot_cart_comp.replace("/", ",").split(",") if p.strip()]
            if len(spec) != 3:
                raise ValueError("--plot_cart_comp must be 'i,j,k'")
            i0, j0, k0 = map(int, spec)
            a0 = jk_to_voigt_a(j0, k0)
            label = f"e_{i0}{j0}{k0} -> Voigt a={voigt_names_list[a0]}"
            y_true_c = true_tensor[:, i0, a0]
            y_pred_c = pred_tensor[:, i0, a0]

            if args.nonzero_thresh > 0:
                if args.filter_by == "pred":
                    mask = torch.abs(y_pred_c) > args.nonzero_thresh
                elif args.filter_by == "true":
                    mask = torch.abs(y_true_c) > args.nonzero_thresh
                else:
                    mask = (torch.abs(y_true_c) > args.nonzero_thresh) | (torch.abs(y_pred_c) > args.nonzero_thresh)
                y_true_c = y_true_c[mask]; y_pred_c = y_pred_c[mask]

            y_true_np = y_true_c.numpy(); y_pred_np = y_pred_c.numpy()
            csv_path = os.path.join(args.plot_dir, f"pred_vs_true_cart_i{i0}_j{j0}_k{k0}.csv")
            np.savetxt(csv_path, np.column_stack([y_true_np, y_pred_np]), delimiter=",", header="true,pred", comments="")

            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 6), dpi=150)
                lim_min = float(min(y_true_np.min(), y_pred_np.min())) if y_true_np.size and y_pred_np.size else -1.0
                lim_max = float(max(y_true_np.max(), y_pred_np.max())) if y_true_np.size and y_pred_np.size else 1.0
                pad = 0.05 * (lim_max - lim_min + 1e-8); lo, hi = lim_min - pad, lim_max + pad
                plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
                plt.scatter(y_true_np, y_pred_np, s=8, alpha=0.5)
                mae_comp = float(F.l1_loss(torch.tensor(y_true_np), torch.tensor(y_pred_np)).item()) if y_true_np.size else float('nan')
                plt.xlabel(f"True (e_{i0}{j0}{k0})"); plt.ylabel(f"Pred (e_{i0}{j0}{k0})")
                plt.title(f"Component e_{i0}{j0}{k0} (Voigt {voigt_names_list[a0]}), MAE={mae_comp:.4f}")
                plt.xlim(lo, hi); plt.ylim(lo, hi); plt.grid(True, alpha=0.2)
                out_png = os.path.join(args.plot_dir, f"pred_vs_true_cart_i{i0}_j{j0}_k{k0}.png")
                plt.tight_layout(); plt.savefig(out_png); plt.close()
                print(f"Saved Cartesian-component scatter: {out_png}")
                print(f"Saved CSV: {csv_path}")
            except Exception as e:
                print(f"Plotting specific Cartesian comp skipped: {e}")
                print(f"Saved CSV instead: {csv_path}")

        # Distribution of per-sample maximum absolute predicted Voigt component
        if args.plot_predmax:
            N = pred_tensor.size(0)
            pred_flat = pred_tensor.reshape(N, -1)           # [N,18]
            true_flat = true_tensor.reshape(N, -1)           # [N,18]
            abs_flat = torch.abs(pred_flat)
            max_abs_vals, max_idx = torch.max(abs_flat, dim=1)  # [N]
            pred_max_signed = torch.gather(pred_flat, dim=1, index=max_idx.unsqueeze(1)).squeeze(1)
            true_at_predmax = torch.gather(true_flat, dim=1, index=max_idx.unsqueeze(1)).squeeze(1)

            comp_i = (max_idx // 6).cpu().numpy()
            comp_a = (max_idx % 6).cpu().numpy()
            a_names = voigt_names_np[comp_a]

            csv_path = os.path.join(args.plot_dir, "pred_max_distribution.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write("pred_max_abs,pred_max_signed,true_at_predmax,i_index,a_index,a_name\n")
                for v_abs, v_sig, v_true, ii, aa, an in zip(
                    max_abs_vals.cpu().numpy(),
                    pred_max_signed.cpu().numpy(),
                    true_at_predmax.cpu().numpy(),
                    comp_i,
                    comp_a,
                    a_names,
                ):
                    f.write(f"{float(v_abs):.8e},{float(v_sig):.8e},{float(v_true):.8e},{int(ii)},{int(aa)},{an}\n")

            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(7, 5), dpi=150)
                plt.hist(pred_max_signed.cpu().numpy(), bins=60, alpha=0.85)
                plt.xlabel("Predicted max Voigt component (signed)")
                plt.ylabel("Count")
                plt.title("Distribution of per-sample max predicted Voigt component (signed)")
                out_png = os.path.join(args.plot_dir, "pred_max_signed_hist.png")
                plt.tight_layout(); plt.savefig(out_png); plt.close()

                plt.figure(figsize=(7, 5), dpi=150)
                plt.hist(max_abs_vals.cpu().numpy(), bins=60, alpha=0.85)
                plt.xlabel("|Predicted| max Voigt component")
                plt.ylabel("Count")
                plt.title("Distribution of per-sample |max| predicted Voigt component")
                out_png_abs = os.path.join(args.plot_dir, "pred_max_abs_hist.png")
                plt.tight_layout(); plt.savefig(out_png_abs); plt.close()

                print(f"Saved pred-max histograms: {out_png}, {out_png_abs}")
                print(f"Saved CSV: {csv_path}")
            except Exception as e:
                print(f"Plotting pred-max skipped: {e}")
                print(f"Saved CSV instead: {csv_path}")

if __name__ == "__main__":
    main()
