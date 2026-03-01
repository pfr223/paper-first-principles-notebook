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

## 5) 指标解释

- `Mean`: 攻击成功率（ASR, %），越低越好
- `BenignRefusalRate`: benign 请求被拒答比例（%），越低通常越好

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
```

输出核心文件：
- `server_realtest_v2/replay_eval/multi_model_summary.json`
- `server_realtest_v2/replay_eval/multi_model_summary.csv`
- `server_realtest_v2/replay_eval/multi_model_report.md`
- `server_realtest_v2/replay_eval/analysis_report.md`
