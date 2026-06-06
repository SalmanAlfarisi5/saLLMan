# Byte-level BPE

**Cluster:** [[Phase 1 - Decoder-only GPT]]

**Intuition.** BPE iteratively merges the most frequent adjacent symbol pair to build a subword
vocabulary. **Byte-level** BPE uses the 256 bytes as the base alphabet, so every string is
representable and no `<unk>` ever fires - critical for code (Phase 3), where whitespace and odd
characters matter. GPT-2 used 50,000 merges + 256 byte tokens + 1 = 50,257 vocab.

**In saLLMan.** Trained via HuggingFace `tokenizers` (`ByteLevel(add_prefix_space=False)` to keep
Python indentation literal). The same vocab + special-token IDs thread through every phase
(the [[Tokenization thread]]).

## References
- BPE: "Neural Machine Translation of Rare Words with Subword Units," Sennrich, Haddow & Birch, 2016 - arXiv:1508.07909.
- Byte-level variant: GPT-2 (Radford et al., 2019).

**Connects to:** [[Subword tokenization]] | [[Tokenization thread]] | [[Block packing]]
