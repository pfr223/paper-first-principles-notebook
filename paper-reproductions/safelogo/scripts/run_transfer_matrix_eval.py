#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

from run_multiround_transfer_attack_eval import (
    DEFAULT_FAMILIES,
    _build_round_instruction,
    _infer_one,
    _iter_filtered_rows,
    _load_jsonl,
    _load_model,
    _slug,
)
from safelogo_real_harness import classify_response, make_attack_prompt


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _short_model_tag(model_id: str, max_len: int = 28) -> str:
    tail = model_id.rstrip("/").split("/")[-1]
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in tail)
    if not safe:
        safe = "model"
    return safe[:max_len]


def _pair_filename(source_model: str, target_model: str) -> str:
    digest = hashlib.sha1(f"{source_model}||{target_model}".encode("utf-8")).hexdigest()[:12]
    s_tag = _short_model_tag(source_model)
    t_tag = _short_model_tag(target_model)
    return f"{s_tag}__to__{t_tag}__{digest}.jsonl"


def _try_clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _search_source_prompts(
    source_model: str,
    filtered_rows: Sequence[dict],
    families: Sequence[str],
    rounds: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    dtype: str,
) -> List[dict]:
    print(f"[source] loading {source_model}")
    src_processor, src_model = _load_model(source_model, dtype=dtype)

    out: List[dict] = []
    total_cases = len(filtered_rows) * len(families)
    case_counter = 0
    for sample in filtered_rows:
        sample_id = str(sample["sample_id"])
        split = str(sample.get("split", "id")).lower()
        image_path = str(sample["image_path"])
        harmful_instruction = str(sample["harmful_instruction"])
        for family in families:
            case_counter += 1
            best_round = rounds
            best_prompt = ""
            best_response = ""
            best_label = "ambiguous"
            for round_idx in range(1, rounds + 1):
                attack_instruction = _build_round_instruction(harmful_instruction, round_idx)
                prompt = make_attack_prompt(family, attack_instruction, safety_instruction=None)
                response = _infer_one(
                    processor=src_processor,
                    model=src_model,
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )
                label = classify_response(response)
                best_round = round_idx
                best_prompt = prompt
                best_response = response
                best_label = label
                if label == "harmful_actionable":
                    break
            out.append(
                {
                    "source_model": source_model,
                    "sample_id": sample_id,
                    "split": split,
                    "family": family,
                    "image_path": image_path,
                    "best_round": best_round,
                    "source_prompt": best_prompt,
                    "source_response": best_response,
                    "source_label": best_label,
                }
            )
            if case_counter % 20 == 0:
                print(f"[source:{_slug(source_model)}] progress {case_counter}/{total_cases}")

    del src_model
    del src_processor
    _try_clear_cuda_cache()
    return out


def _eval_target(
    target_model: str,
    source_prompts: Sequence[dict],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    dtype: str,
) -> List[dict]:
    print(f"[target] loading {target_model}")
    tgt_processor, tgt_model = _load_model(target_model, dtype=dtype)

    out: List[dict] = []
    for i, row in enumerate(source_prompts, 1):
        target_response = _infer_one(
            processor=tgt_processor,
            model=tgt_model,
            image_path=str(row["image_path"]),
            prompt=str(row["source_prompt"]),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        target_label = classify_response(target_response)
        out.append(
            {
                "source_model": str(row["source_model"]),
                "target_model": target_model,
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
                "family": str(row["family"]),
                "best_round": int(row["best_round"]),
                "source_label": str(row["source_label"]),
                "target_label": target_label,
                "source_prompt": str(row["source_prompt"]),
                "target_response": target_response,
            }
        )
        if i % 25 == 0:
            print(f"[target:{_slug(target_model)}] progress {i}/{len(source_prompts)}")

    del tgt_model
    del tgt_processor
    _try_clear_cuda_cache()
    return out


def _summarize_pair(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "SourceHAR": 0.0,
            "TargetHAR": 0.0,
            "TransferHARGivenSourceSuccess": 0.0,
            "n": 0,
            "n_source_success": 0,
        }
    src_flags = [1.0 if str(r["source_label"]) == "harmful_actionable" else 0.0 for r in rows]
    tgt_flags = [1.0 if str(r["target_label"]) == "harmful_actionable" else 0.0 for r in rows]
    src_success_rows = [r for r in rows if str(r["source_label"]) == "harmful_actionable"]
    transfer_flags = [1.0 if str(r["target_label"]) == "harmful_actionable" else 0.0 for r in src_success_rows]
    return {
        "SourceHAR": float(sum(src_flags) * 100.0 / len(src_flags)),
        "TargetHAR": float(sum(tgt_flags) * 100.0 / len(tgt_flags)),
        "TransferHARGivenSourceSuccess": float(sum(transfer_flags) * 100.0 / max(1, len(transfer_flags))),
        "n": len(rows),
        "n_source_success": len(src_success_rows),
    }


def _matrix_table(summary_rows: Sequence[dict], value_key: str, row_label: str) -> List[str]:
    source_models = sorted({str(r["source_model"]) for r in summary_rows})
    target_models = sorted({str(r["target_model"]) for r in summary_rows})
    lookup = {(str(r["source_model"]), str(r["target_model"])): float(r.get(value_key, 0.0)) for r in summary_rows}
    lines: List[str] = []
    lines.append(f"### {row_label}")
    lines.append("")
    header = "| source\\\\target | " + " | ".join(target_models) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(target_models)) + " |"
    lines.append(header)
    lines.append(sep)
    for src in source_models:
        vals = [f"{lookup.get((src, tgt), 0.0):.2f}" for tgt in target_models]
        lines.append("| " + src + " | " + " | ".join(vals) + " |")
    lines.append("")
    return lines


def _write_report(path: Path, meta: dict, summary_rows: Sequence[dict]) -> None:
    lines: List[str] = []
    lines.append("# Transfer Matrix Report")
    lines.append("")
    lines.append("## Meta")
    lines.append("")
    lines.append(f"- source_models: `{', '.join(meta['source_models'])}`")
    lines.append(f"- target_models: `{', '.join(meta['target_models'])}`")
    lines.append(f"- setting: `{meta['setting']}`")
    lines.append(f"- splits: `{', '.join(meta['splits'])}`")
    lines.append(f"- families: `{', '.join(meta['families'])}`")
    lines.append(f"- rounds: `{meta['rounds']}`")
    lines.append(f"- num_filtered_rows: `{meta['num_filtered_rows']}`")
    lines.append("")
    lines.append("## Pair Summary")
    lines.append("")
    lines.append("| source_model | target_model | SourceHAR | TargetHAR | TransferHARGivenSourceSuccess | n |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        lines.append(
            f"| {row['source_model']} | {row['target_model']} | {float(row['SourceHAR']):.2f} | "
            f"{float(row['TargetHAR']):.2f} | {float(row['TransferHARGivenSourceSuccess']):.2f} | {int(row['n'])} |"
        )
    lines.append("")
    lines.extend(_matrix_table(summary_rows, value_key="TargetHAR", row_label="TargetHAR Matrix"))
    lines.extend(
        _matrix_table(
            summary_rows,
            value_key="TransferHARGivenSourceSuccess",
            row_label="TransferHARGivenSourceSuccess Matrix",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-target transfer matrix evaluation.")
    parser.add_argument("--dataset-jsonl", required=True, help="Path to dataset_realtest.jsonl.")
    parser.add_argument("--source-models", required=True, help="Comma-separated source model ids/paths.")
    parser.add_argument("--target-models", required=True, help="Comma-separated target model ids/paths.")
    parser.add_argument("--work-dir", required=True, help="Output directory.")
    parser.add_argument("--setting", default="no_defense", help="Dataset setting filter.")
    parser.add_argument("--splits", default="id,ood", help="Comma-separated split filter.")
    parser.add_argument("--families", default="PAIR,GCG,PAP,AutoDAN", help="Comma-separated families.")
    parser.add_argument("--rounds", type=int, default=4, help="Search rounds per source model.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit filtered rows (0 means all).")
    parser.add_argument("--dtype", default="bf16", help="bf16/fp16/fp32.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation max tokens.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    dataset_jsonl = Path(args.dataset_jsonl)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_jsonl}")

    rows = _load_jsonl(dataset_jsonl)
    splits = [x.strip().lower() for x in str(args.splits).split(",") if x.strip()]
    families = [x.strip() for x in str(args.families).split(",") if x.strip()]
    if not families:
        families = list(DEFAULT_FAMILIES)
    source_models = [x.strip() for x in str(args.source_models).split(",") if x.strip()]
    target_models = [x.strip() for x in str(args.target_models).split(",") if x.strip()]
    if not source_models or not target_models:
        raise ValueError("source-models and target-models cannot be empty.")

    filtered_rows = _iter_filtered_rows(
        rows=rows,
        setting=str(args.setting),
        split_filter=splits,
        max_samples=int(args.max_samples),
    )
    if not filtered_rows:
        raise ValueError("No rows after filter. Check --setting/--splits.")

    rounds = max(1, int(args.rounds))
    all_summary: List[dict] = []
    all_detailed: List[dict] = []

    for source_model in source_models:
        source_prompt_rows = _search_source_prompts(
            source_model=source_model,
            filtered_rows=filtered_rows,
            families=families,
            rounds=rounds,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            dtype=str(args.dtype),
        )
        source_prompts_path = work_dir / "source_prompts" / f"{_slug(source_model)}.jsonl"
        _write_jsonl(source_prompts_path, source_prompt_rows)
        print(f"saved: {source_prompts_path}")

        for target_model in target_models:
            pair_rows = _eval_target(
                target_model=target_model,
                source_prompts=source_prompt_rows,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                dtype=str(args.dtype),
            )
            pair_stats = _summarize_pair(pair_rows)
            summary_row = {
                "source_model": source_model,
                "target_model": target_model,
                "SourceHAR": pair_stats["SourceHAR"],
                "TargetHAR": pair_stats["TargetHAR"],
                "TransferHARGivenSourceSuccess": pair_stats["TransferHARGivenSourceSuccess"],
                "n": pair_stats["n"],
                "n_source_success": pair_stats["n_source_success"],
            }
            all_summary.append(summary_row)
            all_detailed.extend(pair_rows)

            pair_path = work_dir / "pair_runs" / _pair_filename(source_model, target_model)
            _write_jsonl(pair_path, pair_rows)
            print(f"saved: {pair_path}")

    detailed_jsonl = work_dir / "transfer_matrix_details.jsonl"
    summary_json = work_dir / "transfer_matrix_summary.json"
    summary_csv = work_dir / "transfer_matrix_summary.csv"
    report_md = work_dir / "transfer_matrix_report.md"

    _write_jsonl(detailed_jsonl, all_detailed)
    _write_csv(summary_csv, all_summary)

    meta = {
        "dataset_jsonl": str(dataset_jsonl),
        "source_models": source_models,
        "target_models": target_models,
        "setting": str(args.setting),
        "splits": splits,
        "families": families,
        "rounds": rounds,
        "num_filtered_rows": len(filtered_rows),
        "elapsed_sec": time.time() - t0,
    }
    summary_json.write_text(
        json.dumps(
            {
                "meta": meta,
                "summary_rows": all_summary,
                "outputs": {
                    "detailed_jsonl": str(detailed_jsonl),
                    "summary_csv": str(summary_csv),
                    "report_md": str(report_md),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(report_md, meta=meta, summary_rows=all_summary)

    print(f"saved: {detailed_jsonl}")
    print(f"saved: {summary_json}")
    print(f"saved: {summary_csv}")
    print(f"saved: {report_md}")


if __name__ == "__main__":
    main()
