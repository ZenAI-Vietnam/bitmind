import os
import json
import torch
import numpy as np
import hashlib
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import List, Tuple, Optional
from tqdm import tqdm


ROOT_DIR = "/mnt/video/video/datasets"
CACHE_DIR = "/mnt/video/cache/224"

label_to_int = {
    "real": 0,
    "synthetic": 1,
    "semisynthetic": 1,
}


def get_hash_path(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()


class VideoCacheDataset(Dataset):
    def __init__(self, root_dir: str, cache_dir: str):
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.samples: List[Tuple[str, int]] = []

        self._collect_samples()

    def _collect_samples(self):
        """
        Scan all metadata once, build index
        """
        for dataset_name in os.listdir(self.root_dir):
            if dataset_name != "veo3-preferences":
                continue
            dataset_path = os.path.join(self.root_dir, dataset_name)

            week_dirs = (
                os.listdir(dataset_path)
                if dataset_name == "gasstation-generated-videos"
                else [""]
            )

            for week_dir in week_dirs:
                meta_file = os.path.join(
                    dataset_path, week_dir, "sample_metadata.json"
                )
                if not os.path.exists(meta_file):
                    continue

                with open(meta_file, "r") as f:
                    data = json.load(f)

                samples_dir = os.path.join(dataset_path, week_dir, "samples")

                for file, info in data.items():
                    video_path = os.path.join(samples_dir, file)
                    source_file = info["source_file"]
                    member_path = info["member_path"]
                    label = label_to_int[info["media_type"]]
                    # if label != 0:
                    #     continue

                    hash_path = get_hash_path(video_path)
                    cache_path = os.path.join(
                        self.cache_dir, f"{hash_path}.pt"
                    )

                    if os.path.exists(cache_path):
                        self.samples.append((video_path, source_file, member_path, cache_path, label))

        print(f"[INFO] Collected {len(self.samples)} cached samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, source_file, member_path, cache_path, label = self.samples[idx]

        with torch.no_grad():
            raw_feat = torch.load(cache_path, map_location="cpu")
        return raw_feat, label, source_file, member_path

def main():
    dataset = VideoCacheDataset(
        root_dir=ROOT_DIR,
        cache_dir=CACHE_DIR,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        persistent_workers=True,
    )

    label_dict = {0: [], 1: []}

    for feats, labels, source_file, member_path in tqdm(dataloader):
        if source_file[0] != "0030_ray2_1_68a96774.mp4":
            continue
        print(feats.shape)
        print(feats.numpy())
        feats = feats.mean(dim=1)
        outputs = F.avg_pool2d(feats, kernel_size=(7, 7), stride=(7, 7), padding=(0, 0))
        outputs = outputs.view(outputs.size(0), -1)
        print(outputs.shape)
        print(outputs.numpy())
        for f, l in zip(outputs, labels):
            label_dict[int(l)].append(f)

    # for label, feats in label_dict.items():
    #     if len(feats) == 0:
    #         continue
    #     feats = torch.stack(feats)
    #     # print(f"Label {label}: {feats.shape}")
    #     with open(f"index_mean_pool_7x7_flattened_label_{label}.pt", "wb") as f:
    #         torch.save(feats, f)
    #     print(f"Label {label}: {feats.shape}")


if __name__ == "__main__":
    main()
    