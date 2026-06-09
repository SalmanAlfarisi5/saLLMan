# Residual stream

**Cluster:** cross-cutting hub · part of [[Home]]

## Intuition
The backbone of every Transformer block. Each sub-layer (attention, FFN) *reads from* and *writes back to* a shared residual vector rather than replacing it:

```
x = x + Sublayer(x)
```

## Why it matters
- Normalization *placement* on this stream is the whole Post-LN vs Pre-LN story: [[Post-LN]] normalizes the belt after each add; [[Pre-LN]] normalizes only the *input to* each sub-layer and never touches the belt itself.
- [[GPT-2 initialization]]'s `1/√(2N)` scaling on residual-feeding projections exists specifically to keep the belt's variance from blowing up as depth N grows.

## Connects to
[[LayerNorm]] · [[Pre-LN]] · [[Post-LN]] · [[GPT-2 initialization]] · [[Architecture progression]]

Status: Done