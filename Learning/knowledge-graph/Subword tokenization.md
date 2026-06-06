# Subword tokenization

**Cluster:** [[Phase 1 - Decoder-only GPT]]

**Intuition.** Tokens between characters and words balance vocabulary size against sequence
length and gracefully handle rare/unseen words by splitting them into known pieces
(`"unbelievably" -> ["un", "believ", "ably"]`). Byte-level avoids `<unk>` entirely.

**In saLLMan.** Replaces Phase 0's regex word-tokenizer, which exploded the vocab on raw text.

**Connects to:** [[Byte-level BPE]] | [[Tokenization thread]]
