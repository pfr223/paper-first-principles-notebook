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
  - `run_multiround_transfer_attack_eval.py`（多轮攻击搜索 + 跨模型迁移评测）
  - `run_attack_budget_curve_eval.py`（攻击预算曲线：HAR@budget）
  - `run_transfer_matrix_eval.py`（source x target 迁移矩阵评测）
  - `evaluate_logo_perceptual_quality.py`（PSNR/SSIM/LPIPS 感知质量评测）
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

# 4) 多轮 + 迁移攻击评测（source=3B -> target=7B）
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
```

## 高级实验（预算曲线 / 迁移矩阵 / 感知质量）
```bash
cd /home/jack/paper-first-principles-notebook/paper-reproductions/safelogo

# 1) 攻击预算曲线：round budget 从 1..5 的累计 HAR
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

# 2) 迁移矩阵：source_models x target_models
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

# 3) 感知质量：PSNR/SSIM/(可选 LPIPS)
~/anaconda3/bin/python scripts/evaluate_logo_perceptual_quality.py \
  --dataset-jsonl server_realtest_v2/dataset_realtest.jsonl \
  --orig-setting no_defense \
  --logo-setting safelogo_only \
  --work-dir server_realtest_v2/perceptual_quality \
  --psnr-min 28 \
  --ssim-min 0.90 \
  --lpips-max 0.20

# 若 dataset 中是其他机器绝对路径，可加路径映射：
#   --path-prefix-from /home/jack/paper-first-principles-notebook/paper-reproductions/safelogo/ \
#   --path-prefix-to /Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/
```

## 本轮新增能力（已实现）
1. 可扩展压力测试数据集生成：支持自动构造 ID/OOD 图像分布、logo/no-logo 对照与 4 个 setting。
2. 三态评测与统计稳健性：将单一 ASR 判定升级为 `HAR/ORR/Ambiguous`，并输出 bootstrap 95% CI。
3. 多轮迁移攻击评测：新增 source->target 的多轮攻击搜索与迁移复放脚本，支持 2xA800 实测。
4. 攻击预算曲线：输出 `HAR@budget` 与 ambiguous 随预算变化曲线，可识别“仅抗首轮”的伪稳健。
5. 迁移矩阵评测：输出 source x target 的 `TargetHAR` 与 `TransferHARGivenSourceSuccess`。
6. 感知质量评测：新增 `PSNR/SSIM/(可选LPIPS)` 与阈值通过率，量化安全-视觉质量权衡。

## 真实测试结论（V2）
- 深度报告：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/docs/REALTEST_V2_RESULTS.md`
- 原始汇总：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/server_realtest_v2/replay_eval/multi_model_summary.json`
- 失败样例分析：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/server_realtest_v2/replay_eval/analysis_report.md`

## 说明
- 当前结果分两层：
  - 机制级（surrogate）验证 SafeLogo 的核心因果机制和参数趋势。
  - 接口级（replay）用于接入真实模型离线响应做可重复评测。
- 若要严格对齐论文 benchmark，需要补真实数据与真实 VLM 推理响应缓存。
