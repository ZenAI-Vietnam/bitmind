# dataset_video_val.py
import os
import json
import random
from typing import Dict, List, Optional
from collections import Counter

import torch
import numpy as np
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms

from transforms import apply_random_augmentations

DATA_DIR = "/mnt/bitmind/datasets/datasets"

label_to_int = {"real": 0, "synthetic": 1, "semisynthetic": 1}

DEFAULT_EXCLUDE = {
    "celeb-df-v2",
    "celeb-df-v1",
    "rtfs-10k-uniface",
    "UADFV-fake",
    "UADFV-real",
    "evalcrafter-t2v",
    "text-2-video-human-preferences-moonvalley-marey",
    "lovora-real",
}


def _apply_augmentations(video_thwc: np.ndarray, level_probs: Dict[int, float], mask: np.ndarray = None) -> np.ndarray:
    video_thwc, mask, _, _ = apply_random_augmentations(video_thwc, mask=mask, level_probs=level_probs)
    return video_thwc


def val_transforms_lvl0(video_thwc: np.ndarray) -> np.ndarray:
    # deterministic/no heavy random augs
    return _apply_augmentations(video_thwc, level_probs={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0})


class ValDataset(Dataset):
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        limit_per_dataset: int = 100,
        num_frames: int = 16,
        height: int = 224,
        width: int = 224,
        seed: int = 42,
        exclude: Optional[List[str]] = None,
        verbose: bool = True,
        group_by_dataset: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.limit_per_dataset = limit_per_dataset
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.rng = random.Random(seed)
        self.exclude = set(exclude) if exclude is not None else set(DEFAULT_EXCLUDE)
        self.verbose = verbose
        self.group_by_dataset = group_by_dataset

        # preprocess giống code train/val của bạn
        self.custom_transforms = val_transforms_lvl0
        self.aug_list = transforms.Compose([
            transforms.ToTensor(),  # HWC uint8 -> CHW float [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.data = self._load_data()

        # Group theo dataset để eval chạy theo từng dataset (block)
        if self.group_by_dataset:
            self.data.sort(key=lambda x: (x["dataset_name"], x["video_path"]))

            if self.verbose:
                c = Counter(d["dataset_name"] for d in self.data)
                print("[ValDataset] grouped order summary:")
                for k in sorted(c.keys()):
                    print(f"  - {k}: {c[k]} samples")
                print(f"[ValDataset] total indexed samples: {len(self.data)}")

    def _load_data(self) -> List[dict]:
        data: List[dict] = []

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"DATA_DIR not found: {self.data_dir}")

        # stable order
        dataset_names = [
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]
        dataset_names = sorted(dataset_names)

        for dataset_name in dataset_names:
            if dataset_name in self.exclude:
                if self.verbose:
                    print(f"[ValDataset] skip excluded dataset: {dataset_name}")
                continue

            dataset_path = os.path.join(self.data_dir, dataset_name)
            dataset_samples: List[dict] = []

            week_dirs = sorted(os.listdir(dataset_path)) if dataset_name == "gasstation-generated-videos" else [""]

            if self.verbose:
                if dataset_name == "gasstation-generated-videos":
                    print(f"[ValDataset] indexing dataset: {dataset_name} (weeks={len(week_dirs)})")
                else:
                    print(f"[ValDataset] indexing dataset: {dataset_name}")

            for week_dir in week_dirs:
                meta_path = os.path.join(dataset_path, week_dir, "sample_metadata.json")
                if not os.path.exists(meta_path):
                    continue

                with open(meta_path, "r") as f:
                    metadata = json.load(f)

                for video_name, info in metadata.items():
                    media_type = info.get("media_type", None)
                    if media_type not in label_to_int:
                        continue
                    dataset_samples.append({
                        "video_path": os.path.join(dataset_path, week_dir, "samples", video_name),
                        "label": media_type,
                        "dataset_name": dataset_name,
                    })

            before = len(dataset_samples)
            if self.limit_per_dataset is not None and before > self.limit_per_dataset:
                dataset_samples = self.rng.sample(dataset_samples, self.limit_per_dataset)
            after = len(dataset_samples)

            if self.verbose:
                print(f"[ValDataset] -> collected {before} candidates, keep {after} samples for {dataset_name}")

            data.extend(dataset_samples)

        if self.verbose:
            print(f"[ValDataset] total indexed samples (pre-group): {len(data)}")

        return data

    def __len__(self) -> int:
        return len(self.data)

    def _dummy(self):
        x = torch.zeros((self.num_frames, 3, self.height, self.width), dtype=torch.float32)
        y = torch.tensor(0, dtype=torch.long)
        dataset_name = "dummy"
        w = torch.tensor(0.0, dtype=torch.float32)  # ignored
        return x, y, dataset_name, w

    def _load_video_thwc(self, video_path: str) -> np.ndarray:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total = len(vr)
        if total <= 0:
            raise ValueError("No frames")

        T = self.num_frames
        frames = []
        for i in range(min(T, total)):
            frame = vr[i].asnumpy()  # HWC RGB
            if frame is None or frame.size == 0:
                raise ValueError(f"Invalid frame {i}")
            frames.append(frame)

        if len(frames) < T:
            frames.extend([frames[-1]] * (T - len(frames)))

        return np.array(frames, dtype=np.uint8)

    def __getitem__(self, idx: int):
        max_tries = 8
        last_err = None

        for _ in range(max_tries):
            item = self.data[idx]
            vp = item["video_path"]
            try:
                video = self._load_video_thwc(vp)     # (T,H,W,C) uint8
                video = self.custom_transforms(video) # (T,H,W,C)

                frames = []
                for t in range(video.shape[0]):
                    frame = video[t]
                    # force 3 channels
                    if frame.shape[-1] > 3:
                        frame = frame[..., :3]
                    elif frame.shape[-1] == 1:
                        frame = np.repeat(frame, 3, axis=-1)

                    ft = self.aug_list(frame)  # (C,H,W) float normalized
                    frames.append(ft)

                x = torch.stack(frames, dim=0)  # (T,C,H,W)
                y = torch.tensor(label_to_int[item["label"]], dtype=torch.long)
                dataset_name = item["dataset_name"]
                w = torch.tensor(1.0, dtype=torch.float32)
                return x, y, dataset_name, w

            except Exception as e:
                last_err = e
                idx = self.rng.randrange(len(self.data))

        # không raise, return dummy w=0
        # if self.verbose:
        #     print(f"[ValDataset] failed after {max_tries}, last_err={last_err}")
        return self._dummy()