# Dynamic padding (key-padding masks)

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** Pad each batch only to *its* longest sequence (not a global max) and supply an
attention key-padding mask so attention ignores pad tokens - saves compute on variable-length code.

**In saLLMan.** The SFT collate function pads to the batch max and builds the label tensor with
`-100` on pad/prompt positions ([[Masked loss]]). Contrast with Phase 1/3 pretraining, where
[[Block packing]] means there is no padding at all.

**Connects to:** [[Causal and padding masking]] | [[Masked loss]] | [[Block packing]]
