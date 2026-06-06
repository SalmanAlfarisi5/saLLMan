# Masked loss (response-only loss)

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** Compute loss only on the *response* tokens, masking the prompt with `ignore_index=-100`
so the model learns to *answer*, not to reproduce the prompt.

**Mechanism (saLLMan).** Tokenize prompt and response separately for a bit-exact boundary:
`input_ids = [BOS] + prompt + response + [EOS]`, `loss_mask = [0] + [0]*Lp + [1]*Lr + [1]`. The
label at position `t` is gated by `loss_mask[t+1]` (the shift-by-one of [[Teacher forcing]]).

**Connects to:** [[Supervised fine-tuning]] | [[Dynamic padding]] | [[Teacher forcing]]
