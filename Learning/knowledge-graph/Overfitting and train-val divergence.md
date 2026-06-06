# Overfitting and train-val divergence

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** When training loss keeps falling but *validation* loss rises, the model is
memorizing rather than generalizing. With a small corpus, many epochs guarantee this.

**In saLLMan.** The Phase 3 v1 failure mode (train PPL 1.34 vs val 68 after ~28 epochs of a 22M-token
corpus) - the direct motivation for the v2 redo guided by [[Chinchilla scaling laws]]. Healthy
training keeps train/val within ~0.5 with val still falling at the end.

**Connects to:** [[Chinchilla scaling laws]] | [[Decontamination]]
