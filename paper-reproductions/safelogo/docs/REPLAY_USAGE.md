# SafeLogo Replay 评测使用说明

本流程用于“真实模型响应离线重放评测”，不依赖在线 API。

## 1) 准备数据集模板

复制并编辑：

`/Users/peiduo/Desktop/aidev/SafeLogo_paper/dataset_template.jsonl`

字段说明：
- `sample_id`: 必须唯一（建议包含 split + setting 前缀）
- `image_path`: 绝对路径
- `harmful_instruction`: 有害请求
- `benign_instruction`: 正常请求
- `setting`: `no_defense | safety_instruction | safelogo_only | si_logo`
- `split`: `id | ood`

说明：
- `safelogo_only` / `si_logo` 的 `image_path` 建议使用**已叠加 logo**后的图像。
- `no_defense` / `safety_instruction` 用原图。

## 2) 生成 replay cache 模板（key + prompt）

```bash
source /Users/peiduo/Desktop/aidev/SafeLogo_paper/.venv_safelogo/bin/activate
python /Users/peiduo/Desktop/aidev/SafeLogo_paper/generate_replay_template.py \
  --dataset /Users/peiduo/Desktop/aidev/SafeLogo_paper/dataset_template.jsonl \
  --out /Users/peiduo/Desktop/aidev/SafeLogo_paper/replay_cache_template.jsonl
```

输出文件中的每行包含：
- `key`: 与评测逻辑强绑定的唯一键
- `prompt`: 你需要喂给真实模型的输入
- `response`: 留空，待你填真实模型响应文本

## 3) 填充真实模型响应

把 `replay_cache_template.jsonl` 中每行的 `response` 填上真实模型输出，保存为：

`/Users/peiduo/Desktop/aidev/SafeLogo_paper/replay_cache_filled.jsonl`

要求：
- 保持 `key` 原样不变
- 一行一条 JSON，不要破坏格式

## 4) 运行 replay 评测

```bash
source /Users/peiduo/Desktop/aidev/SafeLogo_paper/.venv_safelogo/bin/activate
python - <<'PY'
import json
from safelogo_real_harness import run_replay_suite

res = run_replay_suite(
    dataset_jsonl='/Users/peiduo/Desktop/aidev/SafeLogo_paper/dataset_template.jsonl',
    replay_cache_jsonl='/Users/peiduo/Desktop/aidev/SafeLogo_paper/replay_cache_filled.jsonl',
    dim=256,
    strict=True,
)
out = '/Users/peiduo/Desktop/aidev/SafeLogo_paper/replay_eval_results.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print('saved:', out)
print(json.dumps(res, indent=2, ensure_ascii=False))
PY
```

## 5) 指标解释（新版）

- `HAR_Mean`（兼容别名 `Mean`）：harmful-actionable rate（%），越低越好。
- `BenignRefusalRate`（ORR）：benign 请求被拒答比例（%），越低越好。
- `AmbiguousRate`：有害评测中被判为 ambiguous 的比例（%），用于识别评测不确定区。
- `*_CI95`：bootstrap 95% 置信区间，用于比较差异显著性。

## 6) 推荐：大规模 + 双模型一键评测

适用场景：你希望一次完成 `>=200` 条记录，且同时比较两个模型（例如 3B/7B）。

```bash
cd /home/jack/paper-first-principles-notebook/paper-reproductions/safelogo

# A. 生成 208 记录真实压力测试集
~/anaconda3/bin/python scripts/build_realtest_dataset.py \
  --out-dir server_realtest_v2 \
  --num-bases 52 \
  --seed 20260301

# B. 一次跑完多模型推理 + replay 评测
HF_ENDPOINT=https://hf-mirror.com ~/anaconda3/bin/python scripts/run_real_model_replay_eval.py \
  --dataset-jsonl server_realtest_v2/dataset_realtest.jsonl \
  --models Qwen/Qwen2.5-VL-3B-Instruct,Qwen/Qwen2.5-VL-7B-Instruct \
  --work-dir server_realtest_v2/replay_eval \
  --max-new-tokens 96 \
  --dtype bf16

# C. 生成分析报告（含失败样例）
~/anaconda3/bin/python scripts/analyze_replay_results.py \
  --work-dir server_realtest_v2/replay_eval \
  --out-md server_realtest_v2/replay_eval/analysis_report.md

# D. 多轮+迁移攻击评测（source=3B, target=7B）
~/anaconda3/bin/python scripts/run_multiround_transfer_attack_eval.py \
  --dataset-jsonl server_realtest_v2/dataset_realtest.jsonl \
  --source-model Qwen/Qwen2.5-VL-3B-Instruct \
  --target-models Qwen/Qwen2.5-VL-7B-Instruct \
  --setting no_defense \
  --splits id,ood \
  --rounds 4 \
  --work-dir server_realtest_v2/transfer_eval \
  --max-new-tokens 128 \
  --dtype bf16

# E. 攻击预算曲线（HAR@budget）
~/anaconda3/bin/python scripts/run_attack_budget_curve_eval.py \
  --dataset-jsonl server_realtest_v2/dataset_realtest.jsonl \
  --source-model /home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --target-model /home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct \
  --setting no_defense \
  --splits id,ood \
  --max-rounds 5 \
  --work-dir server_realtest_v2/budget_eval \
  --max-new-tokens 128 \
  --dtype bf16

# F. 迁移矩阵（source_models x target_models）
~/anaconda3/bin/python scripts/run_transfer_matrix_eval.py \
  --dataset-jsonl server_realtest_v2/dataset_realtest.jsonl \
  --source-models /home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3,/home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct \
  --target-models /home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3,/home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct \
  --setting no_defense \
  --splits id,ood \
  --rounds 4 \
  --work-dir server_realtest_v2/transfer_matrix \
  --max-new-tokens 128 \
  --dtype bf16

# G. 感知质量评测（PSNR/SSIM/(可选LPIPS)）
~/anaconda3/bin/python scripts/evaluate_logo_perceptual_quality.py \
  --dataset-jsonl server_realtest_v2/dataset_realtest.jsonl \
  --orig-setting no_defense \
  --logo-setting safelogo_only \
  --work-dir server_realtest_v2/perceptual_quality \
  --psnr-min 28 \
  --ssim-min 0.90 \
  --lpips-max 0.20

# 如果 dataset_jsonl 中是跨机器绝对路径，加前缀映射：
#   --path-prefix-from /home/jack/paper-first-principles-notebook/paper-reproductions/safelogo/ \
#   --path-prefix-to /Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/
```

输出核心文件：
- `server_realtest_v2/replay_eval/multi_model_summary.json`
- `server_realtest_v2/replay_eval/multi_model_summary.csv`
- `server_realtest_v2/replay_eval/multi_model_report.md`
- `server_realtest_v2/replay_eval/analysis_report.md`
- `server_realtest_v2/budget_eval/budget_curve_summary.json`
- `server_realtest_v2/transfer_matrix/transfer_matrix_summary.json`
- `server_realtest_v2/perceptual_quality/perceptual_quality_summary.json`
