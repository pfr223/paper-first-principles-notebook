# SafeLogo Advanced Eval (Small-Batch Real Test)

## Run Scope
- Date: 2026-03-01
- Dataset: `server_realtest_v2/dataset_realtest.jsonl`
- Setting: `no_defense`
- Splits: `id, ood`
- Families: `PAIR, GCG, PAP, AutoDAN`
- Models:
  - 3B: `/home/jack/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3`
  - 7B: `/home/jack/modelscope_cache/Qwen2.5-VL-7B-Instruct`

## 1) Attack Budget Curve (source=3B, target=7B)
- Config: `max_rounds=3`, `max_samples=8` (32 cases)
- Summary:
  - Budget=1: SourceCumHAR=96.88, TargetCumHAR=53.12
  - Budget=2: SourceCumHAR=96.88, TargetCumHAR=53.12
  - Budget=3: SourceCumHAR=96.88, TargetCumHAR=71.88
- Interpretation:
  - 3B 生成的攻击在 source 侧几乎首轮即成功（高 SourceCumHAR）。
  - 迁移到 7B 的成功率随预算增加明显提升（53.12 -> 71.88），体现 query budget 风险。

## 2) Transfer Matrix (source x target)
- Config: `rounds=3`, `max_samples=8` (32 cases each pair)
- Results:
  - 3B -> 3B: SourceHAR=96.88, TargetHAR=96.88, TransferHARGivenSourceSuccess=100.00
  - 3B -> 7B: SourceHAR=96.88, TargetHAR=50.00, TransferHARGivenSourceSuccess=51.61
  - 7B -> 3B: SourceHAR=71.88, TargetHAR=56.25, TransferHARGivenSourceSuccess=73.91
  - 7B -> 7B: SourceHAR=71.88, TargetHAR=71.88, TransferHARGivenSourceSuccess=100.00
- Interpretation:
  - 同模迁移对角线接近 100%（预期）。
  - 非对角线迁移存在显著不对称：3B->7B 低于 7B->3B。

## 3) Perceptual Quality (no_defense vs safelogo_only)
- Config: `num_pairs=52`, thresholds: `PSNR>=28`, `SSIM>=0.90` (LPIPS unavailable)
- Overall:
  - PSNR mean=24.86, SSIM mean=0.9538, L2 mean=0.0584
  - QualityPassRate=3.85%
- By split:
  - ID pass rate=0.00%
  - OOD pass rate=7.69%
- Interpretation:
  - 当前 logo 叠加强度在 SSIM 维度较好，但 PSNR 达标率偏低。
  - 后续需要减弱视觉扰动（缩小 patch、调整 alpha、加入质量约束优化）。

## Next Actions
1. 扩到 full-sample（max_samples=0）并分两段执行（先 budget/matrix 再 quality）。
2. 增加质量约束搜索：logo `size/alpha/position` 网格，目标提升 pass rate。
3. 输出同一报告中的 HAR-ORR-Quality 三目标帕累托前沿。

## Artifacts
- `server_realtest_v2/budget_eval_small/budget_curve_summary.json`
- `server_realtest_v2/transfer_matrix_small/transfer_matrix_summary.json`
- `server_realtest_v2/perceptual_quality_small/perceptual_quality_summary.json`
