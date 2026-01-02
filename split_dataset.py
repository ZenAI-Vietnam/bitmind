import os
import random
import orjson
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================= CONFIG =================
ROOT_DIR = "video"
NUM_VAL = 500

OUT_ALL = "all_first_16_3_classes_overfit.jsonl"
OUT_VAL = "val_first_16_3_classes_overfit.jsonl"

LABELS = ["synthetic", "semisynthetic", "real"]

# ==========================================


def get_int_label(label, dataset_name):
    # if dataset_name in ["FF23-real", "UADFV-real", "dfd-real"]:
    #     return 3
    # elif dataset_name in ["FF23-fake", "UADFV-fake", "dfd-fake", "semisynthetic-video"]:
    #     return 4
    if label == "synthetic":
        return 1
    elif label == "semisynthetic":
        return 2
    return 0


def process_dataset(args):
    label, dataset_name = args
    dataset_path = os.path.join(ROOT_DIR, label, dataset_name)
    jsonl_path = os.path.join(dataset_path, "frame_paths_first_16.jsonl")

    if not os.path.exists(jsonl_path):
        return [], []
    print(f"Processing {dataset_name} with {label} label")

    int_label = get_int_label(label, dataset_name)

    grouped = defaultdict(list)
    all_lines = []
    val_lines = []

    with open(jsonl_path, "r") as f:
        for line in f:
            sample = orjson.loads(line)
            sample["label"] = int_label

            b = orjson.dumps(sample)
            all_lines.append(b)

            grouped[sample["original_video_path"]].append(b)

    # sample validation
    keys = list(grouped.keys())
    selected = random.sample(keys, min(NUM_VAL, len(keys)))

    for k in selected:
        frames = grouped[k]
        # n = min(len(frames), 4)
        # n -= n % 2
        n = 1
        if n > 0:
            val_lines.extend(random.sample(frames, n))

    return all_lines, val_lines


def main():
    # 🔥 XOÁ FILE CŨ
    for path in [OUT_ALL, OUT_VAL]:
        if os.path.exists(path):
            os.remove(path)

    # tạo job list
    jobs = []
    for label in LABELS:
        label_dir = os.path.join(ROOT_DIR, label)
        if not os.path.exists(label_dir):
            continue
        for dataset_name in os.listdir(label_dir):
            # if dataset_name not in ["dfd-real", "dfd-fake", "FF23-real", "FF23-fake", "UADFV-real", "UADFV-fake"]:
            #     continue
            jobs.append((label, dataset_name))

    print(f"🚀 Processing {len(jobs)} datasets using {cpu_count()} CPU cores")

    # multiprocessing
    with Pool(cpu_count()) as pool, \
         open(OUT_ALL, "wb") as f_all, \
         open(OUT_VAL, "wb") as f_val:

        for all_lines, val_lines in tqdm(
            pool.imap_unordered(process_dataset, jobs),
            total=len(jobs),
            desc="Datasets"
        ):
            for l in all_lines:
                f_all.write(l + b"\n")
            for l in val_lines:
                f_val.write(l + b"\n")


if __name__ == "__main__":
    main()
