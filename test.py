# verify_python_json.py
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import os

local = hf_hub_download(
    "bigcode/the-stack-smol",
    filename="data/python/data.json",
    repo_type="dataset",
)
print(f"local path: {local}")
print(f"file size: {os.path.getsize(local)/1e6:.1f} MB")

# load_dataset("json", ...) handles both JSON array and JSONL transparently.
ds = load_dataset("json", data_files=local, split="train")
print(f"rows: {len(ds)}")
print(f"columns: {ds.column_names}")

sample = ds[0]
print("\n-- field types & sizes --")
for k, v in sample.items():
    s = str(v)
    print(f"  {k!r}: type={type(v).__name__}, len={len(s)}")

# Show the code field's first 300 chars to sanity-check it's actually Python.
for candidate in ("content", "code", "text"):
    if candidate in sample:
        print(f"\n-- {candidate!r}[:300] --")
        print(sample[candidate][:300])
        break
else:
    print("\nWARN: no obvious code field; full first row:")
    print({k: str(v)[:120] for k, v in sample.items()})

# Rough total-character signal across the whole dataset (first 100 rows).
import statistics
sizes = [len(str(ds[i].get("content") or ds[i].get("code") or ds[i].get("text") or ""))
         for i in range(min(100, len(ds)))]
print(f"\nfirst-100 code-field char stats: "
      f"mean={statistics.mean(sizes):.0f}, "
      f"median={statistics.median(sizes):.0f}, "
      f"max={max(sizes)}")