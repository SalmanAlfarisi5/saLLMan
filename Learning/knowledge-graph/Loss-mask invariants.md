# Loss-mask invariants

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** [[Masked loss]] correctness is silent: a wrong mask trains a working-looking model
on the wrong thing. Add invariant assertions over every record before any GPU time is spent.

**Two invariants saLLMan asserts** (in `verify_finetune_data.py`, standalone, no torch):

1. **Length match.** `len(input_ids) == len(loss_mask)` for every record. A drift here means
   the tokenizer was called with different settings between the prompt encode and the response
   encode.

2. **Single 0->1 flip.** `loss_mask` is structurally `[0]*L_p + [1]*L_r` (any non-negative
   prefix of zeros followed by any non-negative suffix of ones). A `1` followed by a `0` means
   the mask was constructed wrong - e.g. the response_ids were prepended instead of appended,
   or the mask was inverted. The script's `find_mask_flip_violation` walks once and reports the
   first violating index with a context window.

**Visual confirmation.** Beyond the assertions, decode the **mask==1 tokens only** for the first
few records and eyeball:
- Must NOT contain `<problem>`, `</problem>`, or the *opening* `<reasoning>` tag.
- Must contain `</reasoning>\n<code>\n` near the middle and end at `</code>`.
- The BPE merge across the prompt/response join may absorb a leading space; that's fine.

If the mask==1 decode shows any prompt content, the boundary is off by N tokens - the gradient
signal is being computed on what the model should have *conditioned on*, not on what it should
have *generated*.

**Anti-pattern.** Building the mask by searching for a delimiter token in the encoded sequence
(e.g. "find `<reasoning>` in `input_ids` and start the mask after it"). BPE may merge the tag
with neighboring whitespace, producing different token IDs at the boundary than expected.
**Encode prompt and response separately and concatenate** for a bit-exact boundary.

**Where to put the check.** Standalone script (no torch), runs in seconds. Run it once after
data prep; run it again after any change to the tokenizer, the prompt template, or the
collate function.

**Connects to:** [[Masked loss]] | [[Supervised fine-tuning]] | [[Dynamic padding]] | [[Teacher forcing]]
