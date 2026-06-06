# Xavier initialization

**Cluster:** [[Phase 0 - Vanilla Transformer]]

**Aliases:** Glorot initialization

## Intuition
Initialise weights so activation and gradient variance stay roughly constant across layers, sampling from variance `2 / (fan_in + fan_out)`. Prevents signals from vanishing or exploding at the start of training.

## Connects to
[[GPT-2 initialization]] — the Phase-2 successor (N(0, 0.02) with residual-projection scaling).

## Reference
"Understanding the difficulty of training deep feedforward neural networks," Glorot & Bengio, AISTATS 2010.
