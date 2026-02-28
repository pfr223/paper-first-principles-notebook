from __future__ import annotations

from pathlib import Path
import textwrap
import nbformat as nbf

out_path = Path('/Users/peiduo/Desktop/aidev/SafeLogo_paper/safelogo_real_harness_notebook.ipynb')

cells = []

cells.append(nbf.v4.new_markdown_cell(textwrap.dedent('''
# SafeLogo 真实接口版复现 Harness（不依赖 Gemini）

本 notebook 使用 `safelogo_real_harness.py`：
- 默认 `MockAdapter` 可直接运行（无外部 API）
- 可替换为 `ReplayAdapter` 接入你真实模型的离线响应缓存
- 覆盖主实验、消融（coverage/epsilon）、鲁棒性（噪声）

> 目的：把上一版机制复现升级为“可对接真实评测数据/模型接口”的工程化入口。
''')))

cells.append(nbf.v4.new_code_cell(textwrap.dedent('''
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/peiduo/Desktop/aidev/SafeLogo_paper')
from safelogo_real_harness import run_full_suite

plt.style.use('seaborn-v0_8-whitegrid')
''')))

cells.append(nbf.v4.new_markdown_cell(textwrap.dedent('''
## 第一性原理与接口映射

- objective: `min_phi max_t L_defense(phi, x, t)`
- assumptions: 局部 logo + 安全指令联合触发拒答机制
- state variables: `delta, mask, coverage, eps, safety instruction`
- mechanism chain: inner 选最强攻击 -> outer 更新 logo -> ASR 降低 + benign 尽量保持

代码映射：
- 训练：`train_patch_surrogate`
- 评测：`evaluate_asr`, `benign_retention`
- 一键套件：`run_full_suite`
''')))

cells.append(nbf.v4.new_code_cell(textwrap.dedent('''
results = run_full_suite(seed=42)
print('keys:', list(results.keys()))
''')))

cells.append(nbf.v4.new_code_cell(textwrap.dedent('''
main = results['main_results']
for k, v in main.items():
    print()
    print(k)
    print('  ID  ASR Mean:', round(v['ID_ASR']['Mean'], 2))
    print('  OOD ASR Mean:', round(v['OOD_ASR']['Mean'], 2))
    print('  Benign ID Total:', round(v['ID_Benign_Total'], 2))
''')))

cells.append(nbf.v4.new_code_cell(textwrap.dedent('''
names = list(main.keys())
id_asr = [main[n]['ID_ASR']['Mean'] for n in names]
ood_asr = [main[n]['OOD_ASR']['Mean'] for n in names]
ben = [main[n]['ID_Benign_Total'] for n in names]

x = np.arange(len(names))
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(x - 0.18, id_asr, width=0.36, label='ID ASR')
axes[0].bar(x + 0.18, ood_asr, width=0.36, label='OOD ASR')
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, rotation=12)
axes[0].set_ylabel('ASR (%) ↓')
axes[0].set_title('Main Defense Effectiveness')
axes[0].legend()

axes[1].bar(x, ben, color='tab:green')
axes[1].set_xticks(x)
axes[1].set_xticklabels(names, rotation=12)
axes[1].set_ylabel('Benign Retention (%) ↑')
axes[1].set_title('Benign Utility (ID)')

plt.tight_layout()
plt.show()
''')))

cells.append(nbf.v4.new_markdown_cell(textwrap.dedent('''
## 消融：Coverage / Epsilon
''')))

cells.append(nbf.v4.new_code_cell(textwrap.dedent('''
cov = results['ablation_coverage']
eps = results['ablation_epsilon']

cov_x = [r[0] * 100 for r in cov]
cov_asr = [r[1] for r in cov]
cov_ben = [r[2] for r in cov]

eps_x = [r[0] * 255 for r in eps]
eps_asr = [r[1] for r in eps]
eps_ben = [r[2] for r in eps]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(cov_x, cov_asr, marker='o', label='ASR ↓')
axes[0].plot(cov_x, cov_ben, marker='s', label='Benign ↑')
axes[0].set_title('Coverage Ablation')
axes[0].set_xlabel('Coverage (%)')
axes[0].legend()

axes[1].plot(eps_x, eps_asr, marker='o', label='ASR ↓')
axes[1].plot(eps_x, eps_ben, marker='s', label='Benign ↑')
axes[1].set_title('Epsilon Ablation')
axes[1].set_xlabel('Epsilon (pixel scale /255)')
axes[1].legend()

plt.tight_layout()
plt.show()
''')))

cells.append(nbf.v4.new_markdown_cell(textwrap.dedent('''
## 鲁棒性：噪声敏感性
''')))

cells.append(nbf.v4.new_code_cell(textwrap.dedent('''
noise = results['noise_robustness']

plt.figure(figsize=(7.5, 4))
plt.plot(noise['noise_grid'], noise['si_only'], marker='o', label='SI only')
plt.plot(noise['noise_grid'], noise['si_logo'], marker='s', label='SI + SafeLogo')
plt.xlabel('Noise std')
plt.ylabel('ASR (%) ↓')
plt.title('Noise Robustness')
plt.legend()
plt.tight_layout()
plt.show()
''')))

cells.append(nbf.v4.new_markdown_cell(textwrap.dedent('''
## 输出结果文件
''')))

cells.append(nbf.v4.new_code_cell(textwrap.dedent('''
out = Path('/Users/peiduo/Desktop/aidev/SafeLogo_paper/safelogo_real_harness_results.json')
out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
print('saved:', out)
''')))

cells.append(nbf.v4.new_markdown_cell(textwrap.dedent('''
## 对接真实模型/数据（下一步）

1. 把真实样本整理为 JSONL（`sample_id,image_path,harmful_instruction,benign_instruction`）。
2. 对每个 `(sample, attack_family, prompt)` 先离线跑模型，缓存响应到 JSONL。
3. 使用 `ReplayAdapter` 读取缓存重放评测，即可保持完全可重复。

这样你可以在不改主评测逻辑的情况下，平滑切到真实 VLM 结果。
''')))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.x'}
}

with out_path.open('w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Notebook written to:', out_path)
