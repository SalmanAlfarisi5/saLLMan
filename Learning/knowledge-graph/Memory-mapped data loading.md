# Memory-mapped data loading

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** Store the tokenized corpus as a flat `uint16` array on disk and access it via
`np.memmap`. The OS pages in only the slices touched - instant startup, ~0 RAM, scales to billions
of tokens (the nanoGPT `.bin` pattern). `uint16` suffices for vocab < 65,536.

**In saLLMan.** `MemmapTokenDataset` samples random `block_size+1` windows (an implicit infinite
shuffle), extending [[Block packing]] to a corpus that can't fit in RAM.

**Connects to:** [[Block packing]] | [[Memory budget]]
