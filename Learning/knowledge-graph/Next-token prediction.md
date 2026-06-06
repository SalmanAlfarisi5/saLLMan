# Next-token prediction

**Cluster:** cross-cutting hub · part of [[saLLMan MOC]]

## Intuition
The single objective underneath every phase. Model the joint probability of a sequence autoregressively — each token predicted from everything before it:

```
p(x) = Π_t  p(x_t | x_<t)
```

Trained by cross-entropy on the sequence shifted by one position. This never changes across saLLMan; what changes around it is the architecture and the data.

## How it threads through the phases
- **Phase 0/1:** trained with [[Teacher forcing]] under a [[Causal masking]] mask so position *t* only sees tokens `< t`.
- **Phase 3 SFT:** still next-token, but with [[Masked loss]] so only *response* tokens contribute.
- **Phase 4 GRPO:** the policy is still a next-token model; RL just reshapes *which* sequences get reinforced (see [[GRPO]]).

## Connects to
[[Teacher forcing]] · [[Causal masking]] · [[Decoder-only GPT architecture]] · [[Masked loss]] · [[Architecture progression]]
