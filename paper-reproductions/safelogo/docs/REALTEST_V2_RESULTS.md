# SafeLogo RealTest V2 Results (2x A800)

## 1) 本次执行内容（你要求的三步）

1. 扩展真实测试集到 `>=200`：已生成 `208` 条记录（`52 base x 4 settings`），覆盖 `ID/OOD` 与四种 setting。  
   - 文件：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/server_realtest_v2/dataset_realtest.jsonl`
   - 摘要：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/server_realtest_v2/dataset_summary.json`

2. 增加第二模型 baseline 并完成同流程 replay 评测：  
   - `Qwen2.5-VL-3B-Instruct`（HF 本地快照）
   - `Qwen2.5-VL-7B-Instruct`（ModelScope 本地目录）

3. 输出真实测试结论文档与报告：  
   - 汇总：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/server_realtest_v2/replay_eval/multi_model_summary.json`
   - 报告：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/server_realtest_v2/replay_eval/analysis_report.md`

---

## 2) 三项改进（已实现）

### 改进 A: 可扩展数据构造器（覆盖 >=200 + ID/OOD + logo/no-logo）
- 脚本：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/scripts/build_realtest_dataset.py`
- 价值：
  - 从“8 条样本”升级为“可配置规模（默认 208）”
  - 自动构造 `no_defense/safety_instruction/safelogo_only/si_logo`
  - 自动制造 OOD 扰动（模糊/色阶/旋转/噪声）

### 改进 B: 多模型自动评测流水线（断点续跑 + 自动汇总）
- 脚本：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/scripts/run_real_model_replay_eval.py`
- 价值：
  - 一次命令跑多个模型
  - 自动 `template -> filled cache -> replay eval -> summary(json/csv/md)`
  - 支持 `resume`，网络/模型问题下可续跑

### 改进 C: 自动失败样例分析与论文差异说明
- 脚本：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/paper-reproductions/safelogo/scripts/analyze_replay_results.py`
- 价值：
  - 自动抽取 unsafe failure case
  - 生成跨模型对比与“与论文设定差异”说明
  - 便于直接进入下一轮迭代

---

## 3) 核心定量结果

### 3B（Qwen2.5-VL-3B-Instruct）
- `ID Mean ASR`:  
  - `no_defense`: 25.64
  - `safelogo_only`: 25.64
  - `safety_instruction`: 5.13
  - `si_logo`: 8.97
- `OOD Mean ASR`:
  - `no_defense`: 49.04
  - `safelogo_only`: 47.12
  - `safety_instruction`: 34.62
  - `si_logo`: 34.62

### 7B（Qwen2.5-VL-7B-Instruct）
- `ID Mean ASR`:
  - `no_defense`: 46.15
  - `safelogo_only`: 47.44
  - `safety_instruction`: 14.10
  - `si_logo`: 5.13
- `OOD Mean ASR`:
  - `no_defense`: 52.88
  - `safelogo_only`: 54.81
  - `safety_instruction`: 0.96
  - `si_logo`: 0.00

### BenignRefusalRate（两模型）
- 大多数 setting 下为 `0.00`
- 在 `safety_instruction` 的 OOD 下出现 `3.85`（两模型均可见）

---

## 4) 第一性原理诊断（深入）

### 4.1 Objective 层
SafeLogo 的目标是用局部区域（logo）触发拒答倾向，同时保持 benign 能力。  
本次 realtest 的 `safelogo_only` 效果弱于预期，关键原因是：
- 这里使用的是“静态 logo 覆盖”，不是论文中的“训练得到的对抗性 logo 参数”。
- 因此它更像视觉扰动基线，而不是真正优化后的防御器。

### 4.2 Assumptions 层
论文假设之一是：`safety instruction + trained logo` 有协同。  
本次结果显示协同在 7B 上明显（特别是 OOD 从 `0.96 -> 0.00`），但在 3B 上不明显（`34.62 -> 34.62`）。
这说明协同强度受模型内在 alignment 与 prompt-following 能力影响。

### 4.3 State Variables 层
关键状态变量中，本次最敏感的是：
- `split (ID/OOD)`：OOD 下无防御 ASR 明显升高。
- `setting`：`safety_instruction` 直接显著压低 ASR。
- `model scale`：3B 与 7B 对同一攻击族的脆弱点分布不同（如 7B 的 GCG/AutoDAN 更高）。

### 4.4 Mechanism Chain 层
观测到的机制链可概括为：
1. 攻击族 prompt 提升“越狱成功倾向”。  
2. `safety_instruction` 给出强先验，抑制部分 harmful 输出。  
3. 未训练 logo 对抑制链路贡献有限；协同主要来自 instruction。  
4. 在 7B 上，`si_logo` 对 OOD 极端攻击进一步收敛到 0，显示更强策略一致性。

---

## 5) 与原论文设定差异（必须说明）

1. 本轮是 replay 离线评测，不是论文中的在线双层训练流程。  
2. 本轮 logo 是静态覆盖图，不是通过 bi-level min-max 优化得到。  
3. 数据为合成压力集，不是论文 benchmark（MM-SafetyBench / VLGuard / MME）原始分布。  

因此：可用于机制趋势验证与工程压力测试，但不能直接对标论文主表绝对数值。

---

## 6) 下一轮高价值动作

1. 把“静态 logo”替换成“真实优化 logo”（outer-loop 学习），再重复同样双模型 replay。  
2. 将 unsafe 判定从字符串启发式升级为多判据（拒答、合规解释、行动性内容分级）。  
3. 按攻击族做分层采样扩展（特别是 GCG/AutoDAN OOD），提升统计稳定性与可解释性。
