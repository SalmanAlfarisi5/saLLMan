# Temperature sampling (top-k / top-p)

**Cluster:** [[Phase 5 - Evaluation]]

**Intuition.** **Temperature** rescales logits before softmax (`softmax(z/T)`): T<1 sharpens (more
greedy), T>1 flattens (more diverse). **top-k** samples from the k most probable tokens; **top-p
(nucleus)** samples from the smallest token set whose cumulative probability exceeds p, truncating the
unreliable tail.

**Effect on [[pass@k]].** Low T (~0 / greedy) is best for **pass@1**; higher T is best for **pass@k at
large k** because diversity raises coverage (Codex used T=0 for pass@1, T~=0.8-1.0 for pass@100, with
optimal T rising monotonically with k).

**In saLLMan.** Drives both Phase 4 rollouts ([[On-policy vs off-policy]]) and Phase 5 scoring.

## Reference
- Nucleus sampling: "The Curious Case of Neural Text Degeneration," Holtzman et al., 2019 - arXiv:1904.09751.

**Connects to:** [[pass@k]] | [[On-policy vs off-policy]] | [[Greedy vs sampling]]
