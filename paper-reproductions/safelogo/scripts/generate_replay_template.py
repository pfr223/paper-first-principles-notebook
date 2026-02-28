#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from safelogo_real_harness import (
    ReplayAdapter,
    make_attack_prompt,
    make_benign_prompt,
    setting_to_safety_instruction,
)


FAM_MAP: Dict[str, List[str]] = {
    "id": ["PAIR", "GCG", "PAP"],
    "ood": ["PAIR", "GCG", "PAP", "AutoDAN"],
}


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate replay_cache template keys from replay dataset jsonl."
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset jsonl.")
    parser.add_argument("--out", required=True, help="Output replay cache template jsonl.")
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            sample_id = str(row["sample_id"])
            setting = str(row.get("setting", "no_defense")).lower()
            split = str(row.get("split", "id")).lower()
            harmful_instruction = row["harmful_instruction"]
            benign_instruction = row.get("benign_instruction", "Describe the image safely.")

            safety_instruction = setting_to_safety_instruction(setting)
            families = FAM_MAP.get(split, FAM_MAP["id"])

            for fam in families:
                prompt = make_attack_prompt(
                    fam, harmful_instruction, safety_instruction=safety_instruction
                )
                key = ReplayAdapter.make_key(sample_id, fam, prompt)
                item = {
                    "key": key,
                    "sample_id": sample_id,
                    "setting": setting,
                    "split": split,
                    "family": fam,
                    "prompt": prompt,
                    "response": "",
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

            benign_prompt = make_benign_prompt(
                benign_instruction, safety_instruction=safety_instruction
            )
            benign_key = ReplayAdapter.make_key(sample_id, "BENIGN", benign_prompt)
            benign_item = {
                "key": benign_key,
                "sample_id": sample_id,
                "setting": setting,
                "split": split,
                "family": "BENIGN",
                "prompt": benign_prompt,
                "response": "",
            }
            f.write(json.dumps(benign_item, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} template entries to: {out_path}")


if __name__ == "__main__":
    main()

