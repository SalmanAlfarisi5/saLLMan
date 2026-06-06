# AdamW

**Cluster:** [[Phase 2 - LLaMA-class modernized decoder]]

**Intuition.** Decouple weight decay from the adaptive gradient update (apply decay directly to
the weights), fixing Adam's broken L2 regularization and improving generalization.

**Selective decay.** Apply decay only to matmul weights - **not** to norms, biases, or embeddings
(the LLaMA convention). Implemented as two parameter groups.

**In saLLMan.** Phase 2 onward, with betas (0.9, 0.95) (the LLaMA/Chinchilla setting vs Vaswani's
0.9/0.98). Phase 3 uses the [[TF32 and fused AdamW]] fused kernel.

## Reference
- "Decoupled Weight Decay Regularization," Loshchilov & Hutter, 2017 - arXiv:1711.05101.

**Connects to:** [[Adam optimizer]] | [[TF32 and fused AdamW]] | [[Cosine schedule]]
