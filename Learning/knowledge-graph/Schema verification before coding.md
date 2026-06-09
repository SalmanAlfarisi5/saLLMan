# Schema verification before coding

**Cluster:** [[Phase 3 - Production-scale code pretraining and SFT]]

**Intuition.** When loading an external dataset (HF, S3, an API), inspect the actual schema by
sampling a few rows before writing the extractor. Use the verified schema as ground truth - do
NOT write probe-and-fallback patterns like `next((k for k in candidates if k in cols), None)`.

**Why it matters.** Defensive probing looks robust but actively *hides* upstream changes. If the
schema shifts (a column rename, a field disappears), a probe-and-fallback silently selects the
wrong column and produces garbage examples that train the model on noise. A hardcoded reference
to the verified schema would error loudly and force a fix.

**Failure modes this prevents (saLLMan).**
- v1 SFT plan named `open-r1/codeforces` as the source. Inspection showed it had no reasoning
  traces - only problems and submissions. The correct source is `open-r1/codeforces-cots`,
  config `solutions_py_decontaminated`. Caught before any code was written.
- Initial extractor used `<think>...</think>` from the assistant output as the reasoning source.
  A length distribution check showed ~15k-token median - far past the 2048 context. Pivoted to
  the dataset's separate `editorial` field (see [[Context length from data]]).
- `bigcode/the-stack-smol` was assumed to expose `data/python/*.parquet` shards. Inspection
  showed a single `data/python/data.json` file. The data_prep.py `hf_hub_download` ->
  `load_dataset("json", ...)` pattern is built against the verified layout.

**Recipe.** Before coding:
1. `load_dataset(..., streaming=True)`, grab a few rows.
2. Check `ds.column_names`, then `print(row)` for the first 2-3 rows.
3. For text fields, sample lengths (`len(row["field"]) for row in islice(ds, 100)`) to confirm
   they fit your downstream budget.
4. Hardcode the column names + nested-path access against what you verified.
5. If a row violates the verified schema at runtime, *skip and count* it (visible) rather than
   silently substituting an alternative column.

**Anti-pattern.** Probing for "common" field names (`next((k for k in ('problem', 'description',
'task') if k in cols), None)`) hides the schema and breaks silently the moment one of those
names becomes correct for the *wrong* reason.

**Connects to:** [[codeforces-cots]] | [[The Stack dataset]] | [[Supervised fine-tuning]]
