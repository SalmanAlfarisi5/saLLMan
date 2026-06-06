# Decoder-only GPT architecture

**Cluster:** [[Phase 1 - Decoder-only GPT]]

**Intuition.** Drop the encoder and [[Encoder vs decoder cross-attention]]; a single stack of
causally-masked decoder blocks predicts the next token. Each block now has two sub-layers
(masked self-attention + FFN) instead of three.

**In saLLMan.** `GPT` / `GPTBlock` in Phase 1, reusing Phase 0's attention/FFN/masks unchanged -
the point is to *isolate* the decoder-only transition. Single [[Tokenization thread]], single
sequence, single mask.

## References
- GPT-1 "Improving Language Understanding by Generative Pre-Training," Radford et al., 2018 (OpenAI tech report; **no arXiv ID**).
- GPT-2 "Language Models are Unsupervised Multitask Learners," Radford et al., 2019 (OpenAI tech report; **no arXiv ID**).
- GPT-3 "Language Models are Few-Shot Learners," Brown et al., 2020 - arXiv:2005.14165.

**Connects to:** [[Next-token prediction]] | [[Encoder vs decoder cross-attention]] | [[Causal and padding masking]] | [[LLaMA architecture]]
