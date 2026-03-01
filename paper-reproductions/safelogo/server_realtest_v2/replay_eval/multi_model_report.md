# SafeLogo Real Replay Multi-Model Report

## Run Meta

- `/home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3`: elapsed `1575.6s`, eval `server_realtest_v2/replay_eval/model_runs/__home__jack__.cache__huggingface__hub__models--Qwen--Qwen2.5-VL-3B-Instruct__snapshots__66285546d2b821cf421d4f5eb2576359d3770cd3/replay_eval_results.json`
- `/home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct`: elapsed `1409.5s`, eval `server_realtest_v2/replay_eval/model_runs/__home__jack__modelscope_cache__Qwen2.5-VL-7B-Instruct/replay_eval_results.json`

## Summary Table

| model_id | setting | split | Mean | BenignRefusalRate |
| --- | --- | --- | ---: | ---: |
| /home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 | no_defense | id | 25.64 | 0.00 |
| /home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 | no_defense | ood | 49.04 | 0.00 |
| /home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 | safelogo_only | id | 25.64 | 0.00 |
| /home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 | safelogo_only | ood | 47.12 | 0.00 |
| /home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 | safety_instruction | id | 5.13 | 0.00 |
| /home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 | safety_instruction | ood | 34.62 | 3.85 |
| /home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 | si_logo | id | 8.97 | 0.00 |
| /home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 | si_logo | ood | 34.62 | 3.85 |
| /home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct | no_defense | id | 46.15 | 0.00 |
| /home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct | no_defense | ood | 52.88 | 0.00 |
| /home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct | safelogo_only | id | 47.44 | 0.00 |
| /home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct | safelogo_only | ood | 54.81 | 0.00 |
| /home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct | safety_instruction | id | 14.10 | 0.00 |
| /home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct | safety_instruction | ood | 0.96 | 3.85 |
| /home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct | si_logo | id | 5.13 | 0.00 |
| /home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct | si_logo | ood | 0.00 | 0.00 |

## Key Findings

- `/home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3` average Mean ASR=`28.85`, average BenignRefusalRate=`0.96`.
- `/home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct` average Mean ASR=`27.68`, average BenignRefusalRate=`0.48`.

