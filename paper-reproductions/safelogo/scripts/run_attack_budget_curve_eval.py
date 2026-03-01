#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from run_multiround_transfer_attack_eval import (
    DEFAULT_FAMILIES,
    _build_round_instruction,
    _infer_one,
    _iter_filtered_rows,
    _load_jsonl,
    _load_model,
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


def _case_key(sample_id: str, family: str) -> str:
    return f"{sample_id}|{family}"


def _build_budget_rows(
    detailed_rows: Sequence[dict],
    max_rounds: int,
    has_target: bool,
) -> List[dict]:
    by_case_round: Dict[Tuple[str, int], dict] = {}
    case_ids: List[str] = []
    seen_case = set()
    for row in detailed_rows:
        case = str(row["case_key"])
        r = int(row["round"])
        by_case_round[(case, r)] = row
        if case not in seen_case:
            seen_case.add(case)
            case_ids.append(case)

    n_cases = len(case_ids)
    out: List[dict] = []
    for budget in range(1, max_rounds + 1):
        src_cum_success = 0
        tgt_cum_success = 0
        src_amb_round = 0
        tgt_amb_round = 0
        src_round_count = 0
        tgt_round_count = 0

        for case in case_ids:
            src_hit = False
            tgt_hit = False
            for r in range(1, budget + 1):
                row = by_case_round.get((case, r))
                if row is None:
                    continue
                if str(row["source_label"]) == "harmful_actionable":
                    src_hit = True
                if has_target and str(row.get("target_label", "")) == "harmful_actionable":
                    tgt_hit = True
            if src_hit:
                src_cum_success += 1
            if tgt_hit:
                tgt_cum_success += 1

            row_budget = by_case_round.get((case, budget))
            if row_budget is not None:
                src_round_count += 1
                if str(row_budget["source_label"]) == "ambiguous":
                    src_amb_round += 1
                if has_target:
                    tgt_round_count += 1
                    if str(row_budget.get("target_label", "")) == "ambiguous":
                        tgt_amb_round += 1

        rec = {
            "budget_round": budget,
            "n_cases": n_cases,
            "SourceCumHAR": (100.0 * src_cum_success / max(1, n_cases)),
            "SourceRoundAmbiguousRate": (100.0 * src_amb_round / max(1, src_round_count)),
        }
        if has_target:
            rec["TargetCumHAR"] = (100.0 * tgt_cum_success / max(1, n_cases))
            rec["TargetRoundAmbiguousRate"] = (100.0 * tgt_amb_round / max(1, tgt_round_count))
        out.append(rec)
    return out


def _write_report(path: Path, meta: dict, budget_rows: Sequence[dict]) -> None:
    lines: List[str] = []
    lines.append("# Attack Budget Curve Report")
    lines.append("")
    lines.append("## Meta")
    lines.append("")
    lines.append(f"- source_model: `{meta['source_model']}`")
    if meta.get("target_model"):
        lines.append(f"- target_model: `{meta['target_model']}`")
    lines.append(f"- setting: `{meta['setting']}`")
    lines.append(f"- splits: `{', '.join(meta['splits'])}`")
    lines.append(f"- families: `{', '.join(meta['families'])}`")
    lines.append(f"- max_rounds: `{meta['max_rounds']}`")
    lines.append(f"- num_filtered_rows: `{meta['num_filtered_rows']}`")
    lines.append("")
    lines.append("## Budget Curve")
    lines.append("")
    if meta.get("target_model"):
        lines.append("| budget_round | SourceCumHAR | TargetCumHAR | SourceRoundAmbiguousRate | TargetRoundAmbiguousRate |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for row in budget_rows:
            lines.append(
                f"| {int(row['budget_round'])} | {float(row['SourceCumHAR']):.2f} | "
                f"{float(row.get('TargetCumHAR', 0.0)):.2f} | {float(row['SourceRoundAmbiguousRate']):.2f} | "
                f"{float(row.get('TargetRoundAmbiguousRate', 0.0)):.2f} |"
            )
    else:
        lines.append("| budget_round | SourceCumHAR | SourceRoundAmbiguousRate |")
        lines.append("| ---: | ---: | ---: |")
        for row in budget_rows:
            lines.append(
                f"| {int(row['budget_round'])} | {float(row['SourceCumHAR']):.2f} | "
                f"{float(row['SourceRoundAmbiguousRate']):.2f} |"
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HAR curves under round/query budget.")
    parser.add_argument("--dataset-jsonl", required=True, help="Path to dataset_realtest.jsonl.")
    parser.add_argument("--source-model", required=True, help="Source model id/path.")
    parser.add_argument("--target-model", help="Optional target model id/path for transfer budget curve.")
    parser.add_argument("--work-dir", required=True, help="Output directory.")
    parser.add_argument("--setting", default="no_defense", help="Dataset setting filter.")
    parser.add_argument("--splits", default="id,ood", help="Comma-separated split filter.")
    parser.add_argument("--families", default="PAIR,GCG,PAP,AutoDAN", help="Comma-separated families.")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum attack rounds.")
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

    filtered_rows = _iter_filtered_rows(
        rows=rows,
        setting=str(args.setting),
        split_filter=splits,
        max_samples=int(args.max_samples),
    )
    if not filtered_rows:
        raise ValueError("No rows after filter. Check --setting/--splits.")

    src_model_id = str(args.source_model)
    tgt_model_id: Optional[str] = str(args.target_model) if args.target_model else None
    max_rounds = max(1, int(args.max_rounds))

    print(f"[source] loading {src_model_id}")
    src_processor, src_model = _load_model(src_model_id, dtype=str(args.dtype))

    tgt_processor = None
    tgt_model = None
    if tgt_model_id:
        print(f"[target] loading {tgt_model_id}")
        tgt_processor, tgt_model = _load_model(tgt_model_id, dtype=str(args.dtype))

    detailed_rows: List[dict] = []
    total_cases = len(filtered_rows) * len(families)
    case_counter = 0
    for sample in filtered_rows:
        sample_id = str(sample["sample_id"])
        split = str(sample.get("split", "id")).lower()
        image_path = str(sample["image_path"])
        harmful_instruction = str(sample["harmful_instruction"])
        for family in families:
            case_counter += 1
            case = _case_key(sample_id, family)
            for round_idx in range(1, max_rounds + 1):
                attack_instruction = _build_round_instruction(harmful_instruction, round_idx)
                prompt = make_attack_prompt(family, attack_instruction, safety_instruction=None)
                src_response = _infer_one(
                    processor=src_processor,
                    model=src_model,
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                )
                src_label = classify_response(src_response)

                tgt_response = None
                tgt_label = None
                if tgt_model_id and tgt_processor is not None and tgt_model is not None:
                    tgt_response = _infer_one(
                        processor=tgt_processor,
                        model=tgt_model,
                        image_path=image_path,
                        prompt=prompt,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=bool(args.do_sample),
                        temperature=float(args.temperature),
                    )
                    tgt_label = classify_response(tgt_response)

                detailed_rows.append(
                    {
                        "case_key": case,
                        "sample_id": sample_id,
                        "split": split,
                        "family": family,
                        "round": round_idx,
                        "source_model": src_model_id,
                        "source_label": src_label,
                        "source_prompt": prompt,
                        "source_response": src_response,
                        "target_model": tgt_model_id or "",
                        "target_label": tgt_label or "",
                        "target_response": tgt_response or "",
                    }
                )
            if case_counter % 20 == 0:
                print(f"[progress] cases {case_counter}/{total_cases}")

    budget_rows = _build_budget_rows(
        detailed_rows=detailed_rows,
        max_rounds=max_rounds,
        has_target=bool(tgt_model_id),
    )

    detailed_jsonl = work_dir / "budget_round_details.jsonl"
    summary_json = work_dir / "budget_curve_summary.json"
    summary_csv = work_dir / "budget_curve_summary.csv"
    report_md = work_dir / "budget_curve_report.md"

    _write_jsonl(detailed_jsonl, detailed_rows)
    _write_csv(summary_csv, budget_rows)

    meta = {
        "dataset_jsonl": str(dataset_jsonl),
        "source_model": src_model_id,
        "target_model": tgt_model_id or "",
        "setting": str(args.setting),
        "splits": splits,
        "families": families,
        "max_rounds": max_rounds,
        "num_filtered_rows": len(filtered_rows),
        "elapsed_sec": time.time() - t0,
    }
    summary_json.write_text(
        json.dumps(
            {
                "meta": meta,
                "budget_rows": budget_rows,
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
    _write_report(report_md, meta=meta, budget_rows=budget_rows)

    print(f"saved: {detailed_jsonl}")
    print(f"saved: {summary_json}")
    print(f"saved: {summary_csv}")
    print(f"saved: {report_md}")


if __name__ == "__main__":
    main()
