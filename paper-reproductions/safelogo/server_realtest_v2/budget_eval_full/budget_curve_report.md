# Attack Budget Curve Report

## Meta

- source_model: `/home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3`
- target_model: `/home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct`
- setting: `no_defense`
- splits: `id, ood`
- families: `PAIR, GCG, PAP, AutoDAN`
- max_rounds: `3`
- num_filtered_rows: `52`

## Budget Curve

| budget_round | SourceCumHAR | TargetCumHAR | SourceRoundAmbiguousRate | TargetRoundAmbiguousRate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 92.31 | 54.81 | 0.00 | 0.00 |
| 2 | 92.31 | 55.29 | 0.00 | 0.00 |
| 3 | 92.31 | 65.87 | 0.00 | 0.00 |

