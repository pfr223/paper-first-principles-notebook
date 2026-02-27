---
name: paper-first-principles-notebook
description: Use when the user asks to explain an academic paper from first principles and/or generate a runnable reproduction notebook (.ipynb) from a PDF, arXiv URL, or paper text. Prefer Codex-native reasoning and local code execution; do not use Gemini or other external LLM APIs unless the user explicitly asks for them.
---

# Paper First Principles Notebook

Deliver paper understanding and runnable reproduction with a strict first-principles workflow.

## Non-negotiables
- Use Codex itself for explanation, decomposition, and notebook drafting.
- Do not call Gemini or any other external model API by default.
- Tie claims to paper evidence (equations, algorithm steps, section-level statements).
- Do not fake results; if a metric is not reproducible under toy scale, state that explicitly.

## Workflow
1. Lock input and output contract.
Identify source type: PDF, arXiv URL, or pasted text.
Identify target output: explanation only, notebook only, or both.

2. Build first-principles decomposition.
For the core method, extract:
- objective (what is optimized and why)
- assumptions (what must hold)
- state variables (symbols/tensors and meanings)
- mechanism chain (cause -> effect steps)
- failure modes (where/why it breaks)
Use `references/first_principles_template.md` as the canonical schema.

3. Map principle to implementation.
Create explicit mapping:
- principle/equation -> code module/function
- algorithm step -> tensor operation/update
- hypothesis -> observable metric/log
Keep this map before writing notebook cells.

4. Generate notebook with adaptive but complete structure.
Mirror the paper's actual structure and depth instead of forcing a fixed section count.
Use `references/notebook_structure.md` to ensure required modules are covered.
Require markdown explanation around each major code block:
objective -> assumptions -> variables -> mechanism -> expected behavior.

5. Validate runtime and coherence.
Execute notebook top-to-bottom when environment allows.
Minimum checks:
- no undefined variables/import gaps
- no placeholder code (`TODO`, `pass`, `...`)
- at least one quantitative evaluation table/summary
- plots/logs support the claimed behavior
- include robustness checks (noise/shift/sensitivity) and ablation experiments where feasible
If execution is unavailable, explicitly report the gap and provide exact local run commands.

6. Deliver concise outputs.
Provide:
- first-principles summary
- reproduction assumptions/simplifications
- execution status (passed/failed/not-run)
- ablation and robustness findings (what changed, what stayed stable, what failed)
- key deltas vs original paper setup

## Commands (copy-paste)
Use these when notebook execution validation is required:

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

## Reference map
- `references/first_principles_template.md`: canonical decomposition template.
- `references/notebook_structure.md`: adaptive notebook structure contract and depth acceptance checks.
