import argparse
import os
import sys
from typing import Dict, Tuple

import torch


def load_model_dict(pt_path: str) -> Dict:
    obj = torch.load(pt_path, map_location="cpu")
    # Megatron legacy ckpt structure
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def compare_models(a: Dict, b: Dict, rel_rms_thresh: float) -> Tuple[int, int, int]:
    missing_in_b = 0
    missing_in_a = 0
    mismatched = 0

    keys_a = set(a.keys())
    keys_b = set(b.keys())

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    if only_a:
        print(f"Keys only in A ({len(only_a)}):")
        for k in only_a[:20]:
            print(f"  {k}")
        if len(only_a) > 20:
            print(f"  ... {len(only_a) - 20} more")
        missing_in_b = len(only_a)

    if only_b:
        print(f"Keys only in B ({len(only_b)}):")
        for k in only_b[:20]:
            print(f"  {k}")
        if len(only_b) > 20:
            print(f"  ... {len(only_b) - 20} more")
        missing_in_a = len(only_b)

    common = sorted(keys_a & keys_b)
    for k in common:
        va = a[k]
        vb = b[k]
        if torch.is_tensor(va) and torch.is_tensor(vb):
            if va.shape != vb.shape:
                print(f"Shape mismatch for {k}: {tuple(va.shape)} vs {tuple(vb.shape)}")
                mismatched += 1
                continue
            diff = (va - vb).abs()
            max_diff = diff.max().item() if diff.numel() > 0 else 0.0
            rel = diff / (vb.abs() + 1e-12)
            max_rel = rel.max().item() if rel.numel() > 0 else 0.0
            # RMS and relative RMS
            if diff.numel() > 0:
                l2 = torch.linalg.norm(va.flatten() - vb.flatten(), ord=2).item()
                rms = l2 / (diff.numel() ** 0.5)
                vb_l2 = torch.linalg.norm(vb.flatten(), ord=2).item()
                rel_rms = rms / ((vb_l2 / (diff.numel() ** 0.5)) + 1e-12)
            else:
                rms = 0.0
                rel_rms = 0.0

            if rel_rms > rel_rms_thresh:
                print(
                    f"Value mismatch for {k}: max_abs={max_diff:.3e}, max_rel={max_rel:.3e}, rms={rms:.3e}, rel_rms={rel_rms:.3e}"
                )
                mismatched += 1
        else:
            continue

    return missing_in_b, missing_in_a, mismatched


def get_all_files(root_dir):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            rel_path = os.path.relpath(os.path.join(root, name), root_dir)
            file_list.append(rel_path)
    return sorted(file_list)


def compare_directories(dir_a, dir_b, thresh):
    all_files_a = get_all_files(dir_a)
    all_files_b = get_all_files(dir_b)

    files_a = {f for f in all_files_a if f.endswith("model_optim_rng.pt")}
    files_b = {f for f in all_files_b if f.endswith("model_optim_rng.pt")}

    if not files_a and all_files_a:
        print(
            f"Warning: No 'model_optim_rng.pt' found in A. Available: {all_files_a[:5]}..."
        )
    if not files_b and all_files_b:
        print(
            f"Warning: No 'model_optim_rng.pt' found in B. Available: {all_files_b[:5]}..."
        )

    only_a = files_a - files_b
    only_b = files_b - files_a
    common = files_a & files_b

    if only_a:
        print(f"Files only in A (model_optim_rng.pt): {only_a}")
    if only_b:
        print(f"Files only in B (model_optim_rng.pt): {only_b}")

    if only_a or only_b:
        print("Directory structure mismatch (based on model_optim_rng.pt)!")
        if not common:
            print("Error: No common 'model_optim_rng.pt' files found to compare.")
            return True

    total_mismatches = 0
    failure = False

    for f in sorted(list(common)):
        print(f"\nComparing file: {f}")
        path_a = os.path.join(dir_a, f)
        path_b = os.path.join(dir_b, f)

        try:
            a = load_model_dict(path_a)
            b = load_model_dict(path_b)
            missing_in_b, missing_in_a, mismatched = compare_models(
                a, b, rel_rms_thresh=thresh
            )
            if missing_in_b > 0 or missing_in_a > 0 or mismatched > 0:
                print(f"  -> Mismatch found in {f}")
                total_mismatches += 1
                failure = True
            else:
                print("  -> Match")
        except Exception as e:
            print(f"  -> Error comparing {f}: {e}")
            failure = True

    return failure


def main():
    parser = argparse.ArgumentParser(
        description="Compare two Megatron checkpoints (directory or legacy .pt)"
    )
    parser.add_argument(
        "ckpt_a",
        type=str,
        help="Path to A: directory or .pt file",
    )
    parser.add_argument(
        "ckpt_b",
        type=str,
        help="Path to B: directory or .pt file",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=1e-3,
        help="Threshold on relative RMS; tensors above are reported",
    )
    args = parser.parse_args()

    if os.path.isdir(args.ckpt_a) and os.path.isdir(args.ckpt_b):
        print(f"Comparing Directory A: {args.ckpt_a}")
        print(f"Comparing Directory B: {args.ckpt_b}")
        if compare_directories(args.ckpt_a, args.ckpt_b, args.thresh):
            sys.exit(1)
        else:
            print("\nAll compared files match within tolerance.")
            sys.exit(0)

    elif os.path.isfile(args.ckpt_a) and os.path.isfile(args.ckpt_b):
        print(f"Loading A: {args.ckpt_a}")
        a = load_model_dict(args.ckpt_a)
        print(f"Loading B: {args.ckpt_b}")
        b = load_model_dict(args.ckpt_b)

        missing_in_b, missing_in_a, mismatched = compare_models(
            a, b, rel_rms_thresh=args.thresh
        )
        print("\nSummary:")
        print(f"  Only in A: {missing_in_b}")
        print(f"  Only in B: {missing_in_a}")
        print(f"  Value mismatches: {mismatched}")

        if missing_in_a == 0 and missing_in_b == 0 and mismatched == 0:
            print("Checkpoints match within tolerance.")
            sys.exit(0)
        else:
            sys.exit(1)

    else:
        print("Error: Inputs must be both files or both directories.")
        sys.exit(2)


if __name__ == "__main__":
    main()
