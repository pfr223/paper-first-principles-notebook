# paper-first-principles-notebook

使用 Codex 对论文进行第一性原理解释，并生成可运行复现 Notebook 的技能。

## 目标
- 从论文原文出发，给出可验证的“原理 -> 机制 -> 代码 -> 指标”闭环。
- 复现尽可能完整，不强制固定章节数，按论文结构自适应组织。
- 默认包含消融实验、鲁棒性分析、失败案例与限制说明。

## 默认原则
- 使用 Codex 自身进行分析与生成。
- 默认不调用 Gemini 或其他外部 LLM API（除非你明确要求）。
- 不伪造实验结果；无法复现时明确写出原因和边界。

## 触发方式
在请求中显式写：

```text
使用 $paper-first-principles-notebook ...
```

## 常用调用示例

### 1) 只做论文解释
```text
使用 $paper-first-principles-notebook，基于这篇论文做第一性原理解释，重点讲 objective、assumptions、state variables、mechanism chain 和 failure modes。
```

### 2) 生成可运行复现 Notebook
```text
使用 $paper-first-principles-notebook，把这篇论文复现成可运行 notebook，尽可能详细覆盖论文全貌，并包含定量评测。
```

### 3) 强化消融与鲁棒性
```text
使用 $paper-first-principles-notebook，生成高深度复现 notebook，必须包含 ablation matrix、噪声/OOD/超参敏感性鲁棒性实验，以及失败样例分析。
```

## 最小可运行示例（arXiv URL -> notebook）

把下面这段直接发给 Codex 即可：

```text
使用 $paper-first-principles-notebook，分析并复现这篇论文：
https://arxiv.org/abs/1706.03762

要求：
1) 用第一性原理解释核心机制（objective、assumptions、state variables、mechanism chain）。
2) 生成可运行 notebook，结构按论文内容自适应，不强制固定章节数。
3) 复现尽可能详细覆盖论文全貌，包含主实验、消融实验、鲁棒性分析（噪声/OOD/超参敏感性至少两项）。
4) 输出定量结果、图表、失败样例分析，并说明与原论文设定差异。
5) 默认使用 Codex 推理，不调用 Gemini 或其他外部模型 API。
```

期望输出：
- 第一性原理拆解摘要
- 可运行 `.ipynb` 复现文件
- 消融与鲁棒性结果汇总
- 复现边界与未覆盖项说明

## 输入与输出

### 输入支持
- PDF 文件
- arXiv 链接
- 论文文本片段

### 输出内容
- 第一性原理拆解（结构见 `references/first_principles_template.md`）
- 原理到实现映射（方程/规则 -> 代码模块 -> 可观测指标）
- 自适应结构的复现 notebook（要求见 `references/notebook_structure.md`）
- 执行状态与复现边界说明

## 标准流程
1. 锁定输入类型与交付目标（解释/复现/两者）。
2. 完成第一性原理拆解（目标、假设、变量、机制链、失效模式）。
3. 建立“原理 -> 公式 -> 代码 -> 观测”的映射。
4. 生成复现 notebook（结构自适应，覆盖必要模块）。
5. 运行校验并修复（无未定义变量、无占位代码、可执行）。
6. 汇总结论：复现成功项、失败项、与原论文差异。

## 运行校验（可复制）
```bash
python -m pip install nbclient nbformat jupyter ipykernel
python - <<'PY'
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('generated_notebook.ipynb', as_version=4)
NotebookClient(nb, timeout=120, kernel_name='python3', allow_errors=False).execute()
print('Notebook execution validation passed')
PY
```

## 质量要求
- 必须有定量结果，不只 narrative 说明。
- 必须有消融与鲁棒性实验（若不可行需明确说明阻塞原因）。
- 必须包含错误分析（典型失败样例 + 可能原因）。
- 必须说明 toy 复现与论文原始设定的差异。

## 相关文件
- 技能定义：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/SKILL.md`
- 第一性原理模板：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/references/first_principles_template.md`
- Notebook 结构与验收：`/Users/peiduo/.codex/skills/paper-first-principles-notebook/references/notebook_structure.md`
