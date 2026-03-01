#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from safelogo_real_harness import is_unsafe_response


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect_metrics(summary: dict) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in summary.get("summary_rows", []):
        model_id = str(row["model_id"])
        out.setdefault(model_id, {"mean_asr_avg": 0.0, "brr_avg": 0.0, "n": 0.0})
        out[model_id]["mean_asr_avg"] += float(row.get("Mean", 0.0))
        out[model_id]["brr_avg"] += float(row.get("BenignRefusalRate", 0.0))
        out[model_id]["n"] += 1.0
    for model_id, rec in out.items():
        n = max(1.0, rec["n"])
        rec["mean_asr_avg"] /= n
        rec["brr_avg"] /= n
    return out


def _collect_failures(cache_jsonl: Path, top_k: int = 5) -> List[dict]:
    rows = _load_jsonl(cache_jsonl)
    harmful_rows = [r for r in rows if str(r.get("family", "")).upper() != "BENIGN"]
    unsafe_rows = [r for r in harmful_rows if is_unsafe_response(str(r.get("response", "")))]
    unsafe_rows.sort(key=lambda r: len(str(r.get("response", ""))), reverse=True)
    return unsafe_rows[:top_k]


def _build_markdown(
    summary: dict,
    model_metrics: Dict[str, Dict[str, float]],
    failures: Dict[str, List[dict]],
) -> str:
    lines: List[str] = []
    lines.append("# SafeLogo Replay Analysis")
    lines.append("")
    lines.append("## Cross-Model Averages")
    lines.append("")
    lines.append("| model_id | Avg Mean ASR | Avg BenignRefusalRate |")
    lines.append("| --- | ---: | ---: |")
    for model_id, rec in model_metrics.items():
        lines.append(
            f"| {model_id} | {rec['mean_asr_avg']:.2f} | {rec['brr_avg']:.2f} |"
        )
    lines.append("")
    lines.append("## Detailed Rows")
    lines.append("")
    lines.append("| model_id | setting | split | Mean | BenignRefusalRate |")
    lines.append("| --- | --- | --- | ---: | ---: |")
    for row in summary.get("summary_rows", []):
        lines.append(
            f"| {row['model_id']} | {row['setting']} | {row['split']} | "
            f"{float(row.get('Mean', 0.0)):.2f} | {float(row.get('BenignRefusalRate', 0.0)):.2f} |"
        )
    lines.append("")
    lines.append("## Unsafe Failure Samples (Harmful Prompt仍被回答)")
    lines.append("")
    empty_all = True
    for model_id, rows in failures.items():
        lines.append(f"### {model_id}")
        if not rows:
            lines.append("- 未发现 unsafe failure 样本。")
        else:
            empty_all = False
            for i, row in enumerate(rows, 1):
                prompt = str(row.get("prompt", "")).replace("\n", " ")[:220]
                resp = str(row.get("response", "")).replace("\n", " ")[:220]
                lines.append(f"{i}. key=`{row.get('key','')}`")
                lines.append(f"   prompt: `{prompt}`")
                lines.append(f"   response: `{resp}`")
        lines.append("")
    if empty_all:
        lines.append("- 当前数据/模型组合下未观察到 ASR > 0 的 harmful 输出。")
        lines.append("")

    lines.append("## Delta vs 论文设定")
    lines.append("")
    lines.append("- 本分析为 replay 离线评测，不是论文中的完整在线训练+大规模 benchmark。")
    lines.append("- 图像数据是合成压力测试集，目的是对比 setting 差异与鲁棒趋势。")
    lines.append("- 指标解释保留论文习惯（ASR/BenignRefusalRate），但绝对数值不直接等同论文主表。")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze multi-model replay results.")
    parser.add_argument("--work-dir", required=True, help="Directory containing multi_model_summary.json")
    parser.add_argument("--out-md", required=True, help="Output markdown report.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    summary_path = work_dir / "multi_model_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary not found: {summary_path}")
    summary = _load_json(summary_path)
    metrics = _collect_metrics(summary)

    failures: Dict[str, List[dict]] = {}
    for run in summary.get("runs", []):
        model_id = str(run["model_id"])
        cache_path = Path(str(run["filled_cache_jsonl"]))
        if cache_path.exists():
            failures[model_id] = _collect_failures(cache_path, top_k=5)
        else:
            failures[model_id] = []

    report = _build_markdown(summary, metrics, failures)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report, encoding="utf-8")
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()
