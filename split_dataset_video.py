import os
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import orjson
from tqdm import tqdm

# ================= CONFIG =================
ROOT_DIR = "video"
NUM_VAL = 200  # số video để lấy vào val cho mỗi dataset

OUT_ALL = "all_video.jsonl"
OUT_VAL = "val_video.jsonl"

LABELS = ["real", "semisynthetic", "synthetic"]
# ==========================================


def get_int_label(label: str, dataset_name: str) -> int:
    """Giữ đúng mapping label như bên split_dataset.py."""
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
    """Xử lý 1 dataset (1 file jsonl) – chạy song song bằng multiprocessing."""
    label, dataset_name = args

    # hiện tại code cũ chỉ dùng week_dirs = [""]
    dataset_path = os.path.join(ROOT_DIR, label, dataset_name)
    jsonl_path = os.path.join(dataset_path, "frame_paths_first_16.jsonl")

    if not os.path.exists(jsonl_path):
        print(f"Dataset {dataset_name} not found")
        return [], []

    print(f"Processing {dataset_name} with label={label}")

    int_label = get_int_label(label, dataset_name)

    grouped = defaultdict(list)  # original_video_path -> list[bytes]
    all_lines = []
    val_lines = []

    with open(jsonl_path, "r") as f:
        for line in f:
            sample = orjson.loads(line)
            sample["label"] = int_label

            b = orjson.dumps(sample)
            all_lines.append(b)

            grouped[sample["original_video_path"]].append(b)

    # chọn video để vào val (giống ý tưởng code cũ: sample theo original_video_path)
    keys = list(grouped.keys())
    selected = random.sample(keys, min(NUM_VAL, len(keys)))

    for k in selected:
        # lấy toàn bộ frame của video đó cho val (đúng như code cũ)
        val_lines.extend(grouped[k])

    return all_lines, val_lines


def main():
    # xóa file cũ nếu tồn tại
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
            jobs.append((label, dataset_name))

    print(f"🚀 Processing {len(jobs)} video datasets using {cpu_count()} CPU cores")

    # multiprocessing + ghi streaming ra file giống split_dataset.py
    with Pool(cpu_count()) as pool, \
            open(OUT_ALL, "wb") as f_all, \
            open(OUT_VAL, "wb") as f_val:

        for all_lines, val_lines in tqdm(
            pool.imap_unordered(process_dataset, jobs),
            total=len(jobs),
            desc="Video datasets",
        ):
            for b in all_lines:
                f_all.write(b + b"\n")
            for b in val_lines:
                f_val.write(b + b"\n")


if __name__ == "__main__":
    main()