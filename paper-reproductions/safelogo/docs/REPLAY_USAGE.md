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
