# Reward hacking (and reward shaping)

**Cluster:** [[Phase 4 - GRPO reinforcement learning]]

**Intuition.** Agents exploit loopholes in the reward (e.g. printing expected outputs, passing weak
tests without solving the problem, padding format tokens). Mitigations: strong/hidden test sets,
format rewards, and the [[KL regularization]] penalty.

**In saLLMan.** Directly relevant to the [[Code-execution reward]] - private tests and decontamination
([[Decontamination]]) reduce the gaming surface.

**Connects to:** [[Code-execution reward]] | [[KL regularization]] | [[Decontamination]]
