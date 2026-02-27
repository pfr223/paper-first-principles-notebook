# Notebook Structure (Adaptive, Not Fixed 12 Sections)

Do not force a fixed section count. Organize sections to mirror the target paper's
actual narrative and technical depth.

## Required modules (must all appear)
1. Problem framing and objective
2. Method derivation and principle explanation
3. Data setup and preprocessing/tokenization
4. Model and algorithm implementation
5. Training/inference procedures
6. Main evaluation (paper-aligned metrics)
7. Ablation experiments
8. Robustness and failure analysis
9. Conclusion and limitations

## Per-section acceptance rules
- Include first-principles explanation for the section:
  objective -> assumptions -> variables -> mechanism -> expected behavior.
- Ensure equations/algorithm steps map to specific code blocks.
- Keep cells top-to-bottom runnable with no hidden state dependency.

## Runtime acceptance rules
- Notebook executes end-to-end without exceptions.
- No placeholders (`TODO`, `pass`, `NotImplementedError`, `...`).
- Include quantitative comparison between baseline and main method.
- Include plots/tables directly tied to paper claims.
- Include ablation matrix over key axes (for example: architecture, loss terms,
  training strategy, data size/noise), or explicitly state why a specific axis
  cannot be reproduced.
- Include robustness checks (for example: perturbation noise, OOD split,
  hyperparameter sensitivity, seed variance) with quantitative summary.
- Include error analysis: representative failures and likely causes.

## If runtime fails
- Report failing cell index and traceback summary.
- Apply minimal fix first (imports, variable scope, shape mismatch, device mismatch).
- Re-run full notebook after fix; do not mark complete before pass.
