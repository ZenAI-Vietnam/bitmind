from datasets import load_dataset

ds = load_dataset("BianYx/VAP-Data", split='train', streaming=True)

print(next(iter(ds)).keys())