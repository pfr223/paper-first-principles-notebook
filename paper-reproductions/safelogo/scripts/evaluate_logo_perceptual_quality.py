#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_index(sample_id: str) -> str:
    m = re.search(r"(\d+)$", sample_id)
    return m.group(1) if m else sample_id


def _to_float_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(20.0 * np.log10(1.0) - 10.0 * np.log10(mse))


def _ssim_global(x: np.ndarray, y: np.ndarray) -> float:
    # Global SSIM proxy on luminance channel; stable and dependency-light.
    wx = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    wy = 0.299 * y[:, :, 0] + 0.587 * y[:, :, 1] + 0.114 * y[:, :, 2]

    mu_x = float(np.mean(wx))
    mu_y = float(np.mean(wy))
    var_x = float(np.var(wx))
    var_y = float(np.var(wy))
    cov_xy = float(np.mean((wx - mu_x) * (wy - mu_y)))

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if den <= 1e-12:
        return 1.0
    return float(num / den)


def _l2_proxy(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _maybe_build_lpips(
    device: str,
    net: str,
) -> Optional[Callable[[np.ndarray, np.ndarray], float]]:
    try:
        import lpips
        import torch
    except Exception:
        return None

    model = lpips.LPIPS(net=net)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model = model.to(device)
    model.eval()

    def _fn(a: np.ndarray, b: np.ndarray) -> float:
        ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        # LPIPS expects [-1, 1]
        ta = ta * 2.0 - 1.0
        tb = tb * 2.0 - 1.0
        with torch.no_grad():
            val = model(ta, tb)
        return float(val.item())

    return _fn


def _aggregate(values: Sequence[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _build_pairs(rows: Sequence[dict], orig_setting: str, logo_setting: str) -> List[dict]:
    grouped: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        setting = str(row.get("setting", "")).lower()
        split = str(row.get("split", "id")).lower()
        sample_id = str(row.get("sample_id", ""))
        image_path = str(row.get("image_path", ""))
        idx = _extract_index(sample_id)
        key = (split, idx)
        grouped.setdefault(key, {})
        grouped[key][setting] = image_path

    pairs: List[dict] = []
    for (split, idx), item in sorted(grouped.items()):
        if orig_setting not in item or logo_setting not in item:
            continue
        pairs.append(
            {
                "split": split,
                "index": idx,
                "orig_path": item[orig_setting],
                "logo_path": item[logo_setting],
            }
        )
    return pairs


def _resolve_local_path(
    path_str: str,
    dataset_root: Path,
    path_prefix_from: str,
    path_prefix_to: str,
) -> str:
    path = Path(path_str)
    if path.exists():
        return str(path)

    if path_prefix_from and path_prefix_to and path_str.startswith(path_prefix_from):
        mapped = Path(path_str.replace(path_prefix_from, path_prefix_to, 1))
        if mapped.exists():
            return str(mapped)

    marker = "/images/"
    if marker in path_str:
        tail = path_str.split(marker, 1)[1]
        candidate = dataset_root / "images" / tail
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Cannot resolve image path: {path_str}. "
        f"Try --path-prefix-from/--path-prefix-to for cross-machine datasets."
    )


def _write_report(path: Path, summary: dict) -> None:
    lines: List[str] = []
    lines.append("# Perceptual Quality Report")
    lines.append("")
    lines.append("## Meta")
    lines.append("")
    meta = summary["meta"]
    lines.append(f"- dataset_jsonl: `{meta['dataset_jsonl']}`")
    lines.append(f"- orig_setting: `{meta['orig_setting']}`")
    lines.append(f"- logo_setting: `{meta['logo_setting']}`")
    lines.append(f"- num_pairs: `{meta['num_pairs']}`")
    lines.append(f"- lpips_enabled: `{meta['lpips_enabled']}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    overall = summary["overall"]
    lines.append("| metric | mean | std | min | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for metric in ("PSNR", "SSIM", "L2Proxy", "LPIPS"):
        if metric not in overall:
            continue
        m = overall[metric]
        lines.append(
            f"| {metric} | {float(m['mean']):.4f} | {float(m['std']):.4f} | {float(m['min']):.4f} | {float(m['max']):.4f} |"
        )
    lines.append("")
    lines.append("| pass_rate |")
    lines.append("| ---: |")
    lines.append(f"| {float(overall['QualityPassRate']):.2f} |")
    lines.append("")
    lines.append("## By Split")
    lines.append("")
    lines.append("| split | PSNR mean | SSIM mean | L2 mean | LPIPS mean | pass_rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for split, rec in summary["by_split"].items():
        lpips_mean = "N/A"
        if "LPIPS" in rec:
            lpips_mean = f"{float(rec['LPIPS']['mean']):.4f}"
        lines.append(
            f"| {split} | {float(rec['PSNR']['mean']):.4f} | {float(rec['SSIM']['mean']):.4f} | "
            f"{float(rec['L2Proxy']['mean']):.4f} | {lpips_mean} | {float(rec['QualityPassRate']):.2f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate perceptual quality between orig and logo images.")
    parser.add_argument("--dataset-jsonl", required=True, help="Path to dataset_realtest.jsonl.")
    parser.add_argument("--work-dir", required=True, help="Output directory.")
    parser.add_argument("--orig-setting", default="no_defense", help="Baseline image setting.")
    parser.add_argument("--logo-setting", default="safelogo_only", help="Logo image setting.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Max pairs to evaluate (0 means all).")
    parser.add_argument("--psnr-min", type=float, default=28.0, help="Quality pass threshold for PSNR.")
    parser.add_argument("--ssim-min", type=float, default=0.90, help="Quality pass threshold for SSIM.")
    parser.add_argument("--lpips-max", type=float, default=0.20, help="Quality pass threshold for LPIPS.")
    parser.add_argument("--lpips-device", default="cpu", help="LPIPS device, e.g., cpu/cuda:0.")
    parser.add_argument("--lpips-net", default="alex", help="LPIPS backbone net.")
    parser.add_argument("--path-prefix-from", default="", help="Optional path prefix to replace from.")
    parser.add_argument("--path-prefix-to", default="", help="Optional path prefix to replace to.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    dataset_jsonl = Path(args.dataset_jsonl)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_jsonl}")
    dataset_root = dataset_jsonl.parent.resolve()

    rows = _load_jsonl(dataset_jsonl)
    pairs = _build_pairs(
        rows=rows,
        orig_setting=str(args.orig_setting).lower(),
        logo_setting=str(args.logo_setting).lower(),
    )
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]
    if not pairs:
        raise ValueError("No image pairs found. Check settings or dataset content.")

    lpips_fn = _maybe_build_lpips(device=str(args.lpips_device), net=str(args.lpips_net))
    lpips_enabled = lpips_fn is not None

    pair_rows: List[dict] = []
    for i, pair in enumerate(pairs, 1):
        orig_path = _resolve_local_path(
            pair["orig_path"],
            dataset_root=dataset_root,
            path_prefix_from=str(args.path_prefix_from),
            path_prefix_to=str(args.path_prefix_to),
        )
        logo_path = _resolve_local_path(
            pair["logo_path"],
            dataset_root=dataset_root,
            path_prefix_from=str(args.path_prefix_from),
            path_prefix_to=str(args.path_prefix_to),
        )

        x = _to_float_rgb(orig_path)
        y = _to_float_rgb(logo_path)
        if x.shape != y.shape:
            yy = Image.fromarray((y * 255.0).astype(np.uint8)).resize(
                (x.shape[1], x.shape[0]), resample=Image.Resampling.BICUBIC
            )
            y = np.asarray(yy, dtype=np.float32) / 255.0

        psnr = _psnr(x, y)
        ssim = _ssim_global(x, y)
        l2 = _l2_proxy(x, y)
        lpips_val = lpips_fn(x, y) if lpips_fn is not None else None
        pass_ok = (psnr >= float(args.psnr_min)) and (ssim >= float(args.ssim_min))
        if lpips_val is not None:
            pass_ok = pass_ok and (lpips_val <= float(args.lpips_max))

        row = {
            "split": pair["split"],
            "index": pair["index"],
            "orig_path": orig_path,
            "logo_path": logo_path,
            "PSNR": psnr,
            "SSIM": ssim,
            "L2Proxy": l2,
            "LPIPS": lpips_val if lpips_val is not None else "",
            "QualityPass": 1 if pass_ok else 0,
        }
        pair_rows.append(row)
        if i % 50 == 0:
            print(f"[progress] pairs {i}/{len(pairs)}")

    def _split_rows(split: str) -> List[dict]:
        return [r for r in pair_rows if str(r["split"]) == split]

    def _summary_for_rows(sub: Sequence[dict]) -> dict:
        out = {
            "PSNR": _aggregate([float(r["PSNR"]) for r in sub]),
            "SSIM": _aggregate([float(r["SSIM"]) for r in sub]),
            "L2Proxy": _aggregate([float(r["L2Proxy"]) for r in sub]),
            "QualityPassRate": float(sum(int(r["QualityPass"]) for r in sub) * 100.0 / max(1, len(sub))),
        }
        lpips_values = [float(r["LPIPS"]) for r in sub if str(r["LPIPS"]) != ""]
        if lpips_values:
            out["LPIPS"] = _aggregate(lpips_values)
        return out

    by_split: Dict[str, dict] = {}
    for split in sorted({str(r["split"]) for r in pair_rows}):
        by_split[split] = _summary_for_rows(_split_rows(split))
    overall = _summary_for_rows(pair_rows)

    per_pair_csv = work_dir / "perceptual_quality_per_pair.csv"
    summary_json = work_dir / "perceptual_quality_summary.json"
    report_md = work_dir / "perceptual_quality_report.md"

    _write_csv(per_pair_csv, pair_rows)
    summary = {
        "meta": {
            "dataset_jsonl": str(dataset_jsonl),
            "orig_setting": str(args.orig_setting).lower(),
            "logo_setting": str(args.logo_setting).lower(),
            "num_pairs": len(pair_rows),
            "lpips_enabled": lpips_enabled,
            "thresholds": {
                "psnr_min": float(args.psnr_min),
                "ssim_min": float(args.ssim_min),
                "lpips_max": float(args.lpips_max),
            },
            "elapsed_sec": time.time() - t0,
        },
        "overall": overall,
        "by_split": by_split,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(report_md, summary)

    print(f"saved: {per_pair_csv}")
    print(f"saved: {summary_json}")
    print(f"saved: {report_md}")


if __name__ == "__main__":
    main()
