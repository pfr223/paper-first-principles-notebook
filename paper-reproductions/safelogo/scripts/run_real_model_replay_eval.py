#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from safelogo_real_harness import ReplayAdapter, make_attack_prompt, make_benign_prompt, run_replay_suite, setting_to_safety_instruction

try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover
    AutoModelForImageTextToText = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen2_5_VLForConditionalGeneration = None


FAMILIES_BY_SPLIT = {
    "id": ["PAIR", "GCG", "PAP"],
    "ood": ["PAIR", "GCG", "PAP", "AutoDAN"],
}


@dataclass
class EvalRunResult:
    model_id: str
    filled_cache_jsonl: Path
    replay_eval_json: Path
    replay_result: Dict[str, object]
    elapsed_sec: float


def _slug(model_id: str) -> str:
    out = model_id.replace("/", "__")
    out = out.replace(":", "_").replace(" ", "_")
    return out


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_template_rows(dataset_rows: Sequence[dict]) -> List[dict]:
    out: List[dict] = []
    for row in dataset_rows:
        sample_id = str(row["sample_id"])
        setting = str(row.get("setting", "no_defense")).lower()
        split = str(row.get("split", "id")).lower()
        harmful_instruction = str(row["harmful_instruction"])
        benign_instruction = str(row.get("benign_instruction", "Describe the image safely."))

        safety_instruction = setting_to_safety_instruction(setting)
        families = FAMILIES_BY_SPLIT.get(split, FAMILIES_BY_SPLIT["id"])
        for fam in families:
            prompt = make_attack_prompt(
                fam,
                harmful_instruction,
                safety_instruction=safety_instruction,
            )
            out.append(
                {
                    "key": ReplayAdapter.make_key(sample_id, fam, prompt),
                    "sample_id": sample_id,
                    "setting": setting,
                    "split": split,
                    "family": fam,
                    "prompt": prompt,
                    "response": "",
                }
            )

        benign_prompt = make_benign_prompt(
            benign_instruction,
            safety_instruction=safety_instruction,
        )
        out.append(
            {
                "key": ReplayAdapter.make_key(sample_id, "BENIGN", benign_prompt),
                "sample_id": sample_id,
                "setting": setting,
                "split": split,
                "family": "BENIGN",
                "prompt": benign_prompt,
                "response": "",
            }
        )
    return out


def _index_image_paths(dataset_rows: Sequence[dict]) -> Dict[Tuple[str, str, str], str]:
    idx: Dict[Tuple[str, str, str], str] = {}
    for row in dataset_rows:
        key = (
            str(row["sample_id"]),
            str(row.get("setting", "no_defense")).lower(),
            str(row.get("split", "id")).lower(),
        )
        idx[key] = str(row["image_path"])
    return idx


def _load_model(model_id: str, dtype: str) -> Tuple[Any, object]:
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype.lower(), torch.bfloat16)
    kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if AutoModelForImageTextToText is not None:
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
            model.eval()
            return processor, model
        except Exception:
            pass
    if Qwen2_5_VLForConditionalGeneration is not None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        model.eval()
        return processor, model
    raise RuntimeError(
        "No compatible VLM loader found. Install transformers with Qwen2.5-VL support."
    )


def _to_device(inputs: Dict[str, object], model: object) -> Dict[str, object]:
    try:
        device = next(model.parameters()).device
    except Exception:  # pragma: no cover
        return inputs
    out: Dict[str, object] = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _infer_one(
    processor: Any,
    model: object,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    import torch

    try:
        from qwen_vl_utils import process_vision_info
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "qwen_vl_utils is required for image-text preprocessing. "
            "Please install it in your environment."
        ) from exc
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = _to_device(inputs, model)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    out = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return out.strip()


def _run_one_model(
    model_id: str,
    template_rows: Sequence[dict],
    image_index: Dict[Tuple[str, str, str], str],
    dataset_jsonl: Path,
    out_dir: Path,
    dim: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    dtype: str,
    save_every: int,
    resume: bool,
) -> EvalRunResult:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    filled_path = out_dir / "replay_cache_filled.jsonl"
    eval_path = out_dir / "replay_eval_results.json"

    rows = [dict(x) for x in template_rows]
    restored = 0
    if resume and filled_path.exists():
        old_rows = _load_jsonl(filled_path)
        old_map = {str(r["key"]): str(r.get("response", "")) for r in old_rows if r.get("response")}
        for row in rows:
            key = str(row["key"])
            if key in old_map:
                row["response"] = old_map[key]
                restored += 1
        print(f"[{model_id}] resume: restored {restored} entries from existing cache")

    pending_idx = [i for i, r in enumerate(rows) if not str(r.get("response", "")).strip()]
    print(f"[{model_id}] pending: {len(pending_idx)} / {len(rows)}")

    processor, model = _load_model(model_id, dtype=dtype)
    print(f"[{model_id}] model loaded")

    for step, i in enumerate(pending_idx, 1):
        row = rows[i]
        img_key = (
            str(row["sample_id"]),
            str(row.get("setting", "no_defense")).lower(),
            str(row.get("split", "id")).lower(),
        )
        image_path = image_index.get(img_key)
        if image_path is None:
            raise KeyError(f"Cannot find image path for key={img_key}")

        row["response"] = _infer_one(
            processor=processor,
            model=model,
            image_path=image_path,
            prompt=str(row["prompt"]),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        if (step % save_every) == 0:
            _write_jsonl(filled_path, rows)
            print(f"[{model_id}] progress {step}/{len(pending_idx)} (autosaved)")

    _write_jsonl(filled_path, rows)
    print(f"[{model_id}] saved filled cache: {filled_path}")

    replay_res = run_replay_suite(
        dataset_jsonl=str(dataset_jsonl),
        replay_cache_jsonl=str(filled_path),
        dim=dim,
        strict=True,
    )
    replay_res["model_id"] = model_id
    replay_res["generated_at"] = int(time.time())
    eval_path.write_text(json.dumps(replay_res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{model_id}] saved replay eval: {eval_path}")

    elapsed = time.time() - t0
    return EvalRunResult(
        model_id=model_id,
        filled_cache_jsonl=filled_path,
        replay_eval_json=eval_path,
        replay_result=replay_res,
        elapsed_sec=elapsed,
    )


def _flatten_summary(run_results: Sequence[EvalRunResult]) -> List[dict]:
    rows: List[dict] = []
    for run in run_results:
        replay = run.replay_result.get("replay_results", {})
        for setting, split_map in replay.items():
            for split, metrics in split_map.items():
                mean_ci = metrics.get("Mean_CI95", [0.0, 0.0])
                brr_ci = metrics.get("BenignRefusalRate_CI95", [0.0, 0.0])
                rec = {
                    "model_id": run.model_id,
                    "setting": setting,
                    "split": split,
                    "Mean": float(metrics.get("Mean", 0.0)),
                    "HAR_Mean": float(metrics.get("HAR_Mean", metrics.get("Mean", 0.0))),
                    "HAR_CI95_Low": float(mean_ci[0]) if isinstance(mean_ci, list) and len(mean_ci) > 1 else 0.0,
                    "HAR_CI95_High": float(mean_ci[1]) if isinstance(mean_ci, list) and len(mean_ci) > 1 else 0.0,
                    "BenignRefusalRate": float(metrics.get("BenignRefusalRate", 0.0)),
                    "ORR_CI95_Low": float(brr_ci[0]) if isinstance(brr_ci, list) and len(brr_ci) > 1 else 0.0,
                    "ORR_CI95_High": float(brr_ci[1]) if isinstance(brr_ci, list) and len(brr_ci) > 1 else 0.0,
                    "AmbiguousRate": float(metrics.get("AmbiguousRate", 0.0)),
                    "BenignAmbiguousRate": float(metrics.get("BenignAmbiguousRate", 0.0)),
                }
                for fam in ("PAIR", "GCG", "PAP", "AutoDAN"):
                    if fam in metrics:
                        rec[fam] = float(metrics[fam])
                rows.append(rec)
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


def _write_markdown_report(path: Path, rows: Sequence[dict], run_results: Sequence[EvalRunResult]) -> None:
    lines: List[str] = []
    lines.append("# SafeLogo Real Replay Multi-Model Report")
    lines.append("")
    lines.append("## Run Meta")
    lines.append("")
    for run in run_results:
        lines.append(
            f"- `{run.model_id}`: elapsed `{run.elapsed_sec:.1f}s`, eval `{run.replay_eval_json}`"
        )
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| model_id | setting | split | HAR Mean | HAR CI95 | ORR | ORR CI95 | Ambiguous |")
    lines.append("| --- | --- | --- | ---: | --- | ---: | --- | ---: |")
    for row in rows:
        lines.append(
            f"| {row['model_id']} | {row['setting']} | {row['split']} | "
            f"{row.get('HAR_Mean', row.get('Mean', 0.0)):.2f} | "
            f"[{row.get('HAR_CI95_Low', 0.0):.2f}, {row.get('HAR_CI95_High', 0.0):.2f}] | "
            f"{row.get('BenignRefusalRate', 0.0):.2f} | "
            f"[{row.get('ORR_CI95_Low', 0.0):.2f}, {row.get('ORR_CI95_High', 0.0):.2f}] | "
            f"{row.get('AmbiguousRate', 0.0):.2f} |"
        )
    lines.append("")

    # Focused observations from the generated metrics.
    lines.append("## Key Findings")
    lines.append("")
    if not rows:
        lines.append("- No rows generated.")
    else:
        by_model: Dict[str, List[dict]] = {}
        for row in rows:
            by_model.setdefault(str(row["model_id"]), []).append(row)
        for model_id, model_rows in by_model.items():
            mean_har = np_mean([r.get("HAR_Mean", r.get("Mean", 0.0)) for r in model_rows])
            mean_orr = np_mean([r.get("BenignRefusalRate", 0.0) for r in model_rows])
            mean_amb = np_mean([r.get("AmbiguousRate", 0.0) for r in model_rows])
            lines.append(
                f"- `{model_id}` average HAR=`{mean_har:.2f}`, average ORR=`{mean_orr:.2f}`, average Ambiguous=`{mean_amb:.2f}`."
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def np_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real VLM replay evaluation for one or multiple models."
    )
    parser.add_argument("--dataset-jsonl", required=True, help="Dataset jsonl path.")
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model IDs, e.g. Qwen/Qwen2.5-VL-3B-Instruct,Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Output directory for template/cache/results.",
    )
    parser.add_argument("--dim", type=int, default=256, help="Feature dim for replay evaluator.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Generation max_new_tokens.")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling generation.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--dtype", default="bf16", help="Model dtype: bf16/fp16/fp32.")
    parser.add_argument("--save-every", type=int, default=20, help="Autosave interval.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from existing cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_jsonl = Path(args.dataset_jsonl)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"dataset-jsonl not found: {dataset_jsonl}")

    model_ids = [x.strip() for x in args.models.split(",") if x.strip()]
    if not model_ids:
        raise ValueError("No model ids provided.")

    dataset_rows = _load_jsonl(dataset_jsonl)
    template_rows = _build_template_rows(dataset_rows)
    image_index = _index_image_paths(dataset_rows)

    template_path = work_dir / "replay_cache_template.jsonl"
    _write_jsonl(template_path, template_rows)
    print(f"Template rows: {len(template_rows)} -> {template_path}")

    run_results: List[EvalRunResult] = []
    for model_id in model_ids:
        model_out = work_dir / "model_runs" / _slug(model_id)
        run = _run_one_model(
            model_id=model_id,
            template_rows=template_rows,
            image_index=image_index,
            dataset_jsonl=dataset_jsonl,
            out_dir=model_out,
            dim=args.dim,
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            dtype=str(args.dtype),
            save_every=max(1, int(args.save_every)),
            resume=not bool(args.no_resume),
        )
        run_results.append(run)

    summary_rows = _flatten_summary(run_results)
    summary_json = work_dir / "multi_model_summary.json"
    summary_csv = work_dir / "multi_model_summary.csv"
    report_md = work_dir / "multi_model_report.md"

    summary_json.write_text(
        json.dumps(
            {
                "models": [r.model_id for r in run_results],
                "runs": [
                    {
                        "model_id": r.model_id,
                        "filled_cache_jsonl": str(r.filled_cache_jsonl),
                        "replay_eval_json": str(r.replay_eval_json),
                        "elapsed_sec": r.elapsed_sec,
                    }
                    for r in run_results
                ],
                "summary_rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(summary_csv, summary_rows)
    _write_markdown_report(report_md, summary_rows, run_results)

    print(f"saved: {summary_json}")
    print(f"saved: {summary_csv}")
    print(f"saved: {report_md}")


if __name__ == "__main__":
    main()
