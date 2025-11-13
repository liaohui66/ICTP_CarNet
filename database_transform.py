import json
import argparse
from pathlib import Path
from typing import List


def to_voigt_from_cartesian(e: List[List[List[float]]]) -> List[List[float]]:
    """
    Convert a single 3x3x3 Cartesian piezoelectric tensor e_{ijk}
    to a 3x6 Voigt tensor, following models.losses.voigt_to_cartesian_e mapping
    with shear_factor = 1.0:
      a=0->xx, 1->yy, 2->zz, 3->yz, 4->xz, 5->xy
    voigt[i,0]=e[i,0,0]; voigt[i,1]=e[i,1,1]; voigt[i,2]=e[i,2,2];
    voigt[i,3]=e[i,1,2]; voigt[i,4]=e[i,0,2]; voigt[i,5]=e[i,0,1].
    """
    if not (isinstance(e, list) and len(e) == 3 and all(isinstance(row, list) and len(row) == 3 for row in e)):
        raise ValueError("expected 3x3x3 list for a Cartesian tensor")
    v = [[0.0] * 6 for _ in range(3)]
    for i in range(3):
        v[i][0] = e[i][0][0]
        v[i][1] = e[i][1][1]
        v[i][2] = e[i][2][2]
        v[i][3] = e[i][1][2]
        v[i][4] = e[i][0][2]
        v[i][5] = e[i][0][1]
    return v


def is_record_list_schema(obj) -> bool:
    return isinstance(obj, list) and all(isinstance(r, dict) for r in obj[:3])


def main():
    parser = argparse.ArgumentParser(description="Convert dataset JSON to {structure, total_voigt_list} schema.")
    parser.add_argument("--src", type=str, default="4201352.json", help="Source JSON path")
    parser.add_argument("--dst", type=str, default="4201352_voigt.json", help="Destination JSON path")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    structures: List[dict] = []
    voigt_list: List[List[List[float]]] = []
    skipped = 0

    if is_record_list_schema(data):
        # Schema: list of records with keys: structure_dict, piezo_tensor_voigt (3x6)
        for i, rec in enumerate(data):
            try:
                s = rec["structure_dict"]
                e = rec["piezo_tensor_voigt"]

                # Accept 3x6 or flat 18
                if isinstance(e, list):
                    flat = []
                    for row in e:
                        if isinstance(row, list):
                            flat.extend(row)
                        else:
                            flat.append(row)
                    if len(flat) != 18:
                        raise ValueError(f"entry {i}: expected 18 elements, got {len(flat)}")
                    e = [flat[0:6], flat[6:12], flat[12:18]]
                else:
                    raise ValueError("piezo_tensor_voigt not a list")

                structures.append(s)
                voigt_list.append(e)
            except Exception as ex:
                skipped += 1
                print(f"Skipping index {i}: {ex}")
    elif isinstance(data, dict):
        # Schema: dict with 'structure' and 'total' (3x3x3 Cartesian list)
        if "structure" not in data or "total" not in data:
            raise KeyError("Expected keys 'structure' and 'total' in dict schema")
        structures = data["structure"]
        totals = data["total"]
        if len(structures) != len(totals):
            print(f"Warning: number of structures ({len(structures)}) != number of total tensors ({len(totals)})")
        N = min(len(structures), len(totals))
        for i in range(N):
            try:
                e_cart = totals[i]
                v = to_voigt_from_cartesian(e_cart)
                voigt_list.append(v)
            except Exception as ex:
                skipped += 1
                print(f"Skipping index {i}: {ex}")
        # Align structures length to converted tensors
        if len(voigt_list) != len(structures):
            structures = structures[:len(voigt_list)]
    else:
        raise ValueError("Unsupported source JSON schema: expected list of records or dict with 'structure' and 'total'")

    out = {"structure": structures, "total_voigt_list": voigt_list}
    with dst.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"Converted {len(voigt_list)} entries. Skipped: {skipped}.")
    print(f"Wrote: {dst.resolve()}")


if __name__ == "__main__":
    main()

