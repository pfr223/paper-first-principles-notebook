# First-Principles Template

Use this template before writing notebook code.

## 1) Problem and Objective
- Task definition:
- Optimization target (formula if available):
- Why this objective should solve the task:

## 2) Assumptions
- Data assumptions:
- Modeling assumptions:
- Optimization assumptions:

## 3) Core Variables and Symbols
- Symbol/tensor:
- Meaning:
- Shape/range:
- Where it appears in the algorithm:

## 4) Mechanism Chain (Cause -> Effect)
1. Step:
Cause:
Effect:
2. Step:
Cause:
Effect:
3. Step:
Cause:
Effect:

## 5) Principle -> Equation -> Code -> Observable
| Principle | Equation/Rule | Code location | Observable check |
| --- | --- | --- | --- |
| | | | |

## 6) Failure Modes
- Failure mode:
- Why it happens:
- How to detect:
- Mitigation at toy scale:

## 7) Minimal Reproduction Claim
- Claim to reproduce:
- What is considered "successful" under toy setup:
- What is intentionally not reproduced (and why):

## 8) Ablation Plan
- Ablation axis:
- Why this axis matters to the mechanism:
- Variants to test:
- Expected trend if the paper's claim is true:

## 9) Robustness Plan
- Robustness dimension (noise/OOD/shift/sensitivity):
- Test protocol:
- Metrics to track:
- Pass/fail criteria:

## 10) Depth and Coverage Checklist
- Which paper sections are fully reproduced:
- Which are partially reproduced:
- Missing pieces and technical blockers:
- Confidence level of reproduction (high/medium/low) with reason:
