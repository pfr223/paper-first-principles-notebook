# SafeLogo 论文复现

## 论文信息
- 标题：**SafeLogo: Turning Your Logos into Jailbreak Shields via Micro-Regional Adversarial Training**
- 文件来源：`42174_SafeLogo_Turning_Your_Lo.pdf`（本仓库不含原始 PDF）
- 任务类型：VLM 安全防御 / jailbreak defense
- 复现方式：Codex 原生分析与实现（不调用 Gemini 或其他外部 LLM API）

## 第一性原理摘要
- `objective`：将防御建模为双层 min-max，外层优化局部 logo 扰动，内层选择最强 jailbreak。
- `assumptions`：
  1. 局部 logo 与 safety instruction 联合可激活模型拒答倾向。
  2. 小覆盖率扰动可在可用性损失可控时提升防御鲁棒性。
  3. 动态 strongest-attack 训练可提高泛化。
- `state variables`：`x, t_jail, M, δ, λ, ρ, ε_def, P_refuse`。
- `mechanism chain`：inner 选最强攻击 -> outer 更新 logo -> 安全/语义损失平衡 -> ASR 降低并保持 benign 能力。

## 已完成内容
- 机制级复现 notebook（含主实验、消融、鲁棒性、失败样例分析）。
- 真实接口版 replay harness（支持离线缓存真实模型响应后重放评测）。
- 数据模板、replay cache 模板生成器、评测使用说明。

## 目录结构
- `scripts/`
  - `build_safelogo_notebook.py`
  - `build_safelogo_real_harness_notebook.py`
  - `safelogo_real_harness.py`
  - `generate_replay_template.py`
  - `build_realtest_dataset.py`（生成 >=200 记录真实评测集）
  - `run_real_model_replay_eval.py`（多模型自动推理 + replay 评测）
  - `analyze_replay_results.py`（汇总对比报告 + 失败样例）
- `notebooks/`
  - `safelogo_first_principles_reproduction.ipynb`
  - `safelogo_first_principles_reproduction.executed.ipynb`
  - `safelogo_real_harness_notebook.ipynb`
  - `safelogo_real_harness_notebook.executed.ipynb`
- `artifacts/`
  - `safelogo_repro_report.json`
  - `safelogo_real_harness_results.json`
- `data/`
  - `dataset_template.jsonl`
  - `replay_cache_template.jsonl`
- `docs/`
  - `REPLAY_USAGE.md`
  - `safelogo_paper.txt`（PDF 提取文本）

## 快速运行
```bash
source /Users/peiduo/Desktop/aidev/SafeLogo_paper/.venv_safelogo/bin/activate
python /Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/scripts/build_safelogo_notebook.py
python /Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/scripts/build_safelogo_real_harness_notebook.py
```

## Replay 评测（真实模型）
1. 填写 `data/dataset_template.jsonl` 的真实路径与文本。
2. 生成 replay 模板：
```bash
source /Users/peiduo/Desktop/aidev/SafeLogo_paper/.venv_safelogo/bin/activate
python /Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/scripts/generate_replay_template.py \
  --dataset /Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/data/dataset_template.jsonl \
  --out /Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/data/replay_cache_template.jsonl
```
3. 填充真实响应后，按 `docs/REPLAY_USAGE.md` 运行评测。

## 服务器真实测试（2x A800 推荐流程）
```bash
cd /home/jack/paper-first-principles-notebook/paper-reproductions/safelogo

# 1) 生成 208 条记录（52 base x 4 settings）
~/anaconda3/bin/python scripts/build_realtest_dataset.py \
  --out-dir server_realtest_v2 \
  --num-bases 52 \
  --seed 20260301

# 2) 双模型自动推理 + replay 评测
HF_ENDPOINT=https://hf-mirror.com ~/anaconda3/bin/python scripts/run_real_model_replay_eval.py \
  --dataset-jsonl server_realtest_v2/dataset_realtest.jsonl \
  --models Qwen/Qwen2.5-VL-3B-Instruct,Qwen/Qwen2.5-VL-7B-Instruct \
  --work-dir server_realtest_v2/replay_eval \
  --max-new-tokens 96 \
  --dtype bf16

# 3) 生成对比分析报告
~/anaconda3/bin/python scripts/analyze_replay_results.py \
  --work-dir server_realtest_v2/replay_eval \
  --out-md server_realtest_v2/replay_eval/analysis_report.md
```

## 本轮新增三项改进（已实现）
1. 可扩展压力测试数据集生成：支持自动构造 ID/OOD 图像分布、logo/no-logo 对照与 4 个 setting。
2. 多模型自动评测流水线：支持 3B/7B 一次跑完、断点续跑、自动导出 JSON/CSV/Markdown 汇总。
3. 结果分析自动化：新增失败样例抓取、跨模型对比与论文设定差异说明，便于直接汇报。

## 真实测试结论（V2）
- 深度报告：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/docs/REALTEST_V2_RESULTS.md`
- 原始汇总：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/server_realtest_v2/replay_eval/multi_model_summary.json`
- 失败样例分析：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/server_realtest_v2/replay_eval/analysis_report.md`

## 说明
- 当前结果分两层：
  - 机制级（surrogate）验证 SafeLogo 的核心因果机制和参数趋势。
  - 接口级（replay）用于接入真实模型离线响应做可重复评测。
- 若要严格对齐论文 benchmark，需要补真实数据与真实 VLM 推理响应缓存。
