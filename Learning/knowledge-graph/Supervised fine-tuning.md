# Supervised fine-tuning (SFT)

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** Continue training the pretrained base on (instruction, response) pairs to teach the
desired format - here, "given a problem, produce reasoning + code". The pretrained model knows
Python syntax but not this *distribution*.

**Recipe (saLLMan).** LR one-to-two orders below pretrain (2e-5 vs 3e-4: weights are already in a
good basin), short warmup, few epochs (small set -> overfitting risk), [[Masked loss]] on response
tokens only. It is also stage 1 of the full [[RLHF]] pipeline that Phase 4 continues.

**Connects to:** [[Masked loss]] | [[RLHF]] | [[Chain-of-thought]] | [[Next-token prediction]]
