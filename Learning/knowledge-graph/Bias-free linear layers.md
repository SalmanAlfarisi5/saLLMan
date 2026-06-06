# Bias-free linear layers

**Cluster:** [[Phase 2 - LLaMA-class modernized decoder]]

**Intuition.** Drop bias terms from linear/attention projections; in large LMs they cost
parameters and memory with negligible quality benefit, and improve stability slightly.

**In saLLMan.** Phase 2 sets `bias=False` everywhere (LLaMA convention), pairing naturally with
[[RMSNorm]] (which also drops its bias).

**Connects to:** [[RMSNorm]] | [[LLaMA architecture]]
