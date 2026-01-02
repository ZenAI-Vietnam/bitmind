import os
import json
import random
import re
import hashlib
import tempfile
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from transforms import apply_random_augmentations


# ================== CONFIG ==================
ROOT_DIR = "/mnt/video/video/datasets"
CACHE_DIR = "cache/224"

label_to_int = {
    "real": 0,
    "synthetic": 1,
    "semisynthetic": 1
}


# ================== UTILS ==================
def extract_number(filepath):
    match = re.match(r"(\d+)", filepath)
    return int(match.group(1)) if match else filepath


def hash_path(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()


def _apply_augmentations(image: np.ndarray, mask=None):
    image, _, _, _ = apply_random_augmentations(image, target_size=(224, 224), mask=mask, level=0)
    return image


def load_first_frame(meta):
    if "video_frames" in meta:
        frames = sorted(meta["video_frames"], key=extract_number)
        img = Image.open(
            os.path.join(meta["video_path"], frames[0])
        ).convert("RGB")
        return np.array(img, dtype=np.uint8)
    else:
        video_path = meta["video_path"]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(meta["video_path"])
        return frame


# ================== BASE DATASET ==================
class BaseDataset(Dataset):
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _cache_path(self, video_path: str):
        h = hash_path(video_path)
        return os.path.join(CACHE_DIR, f"{h}.pt")


# ================== TRAIN DATASET ==================
class TrainDataset(BaseDataset):
    def __init__(self, use_cache=True):
        super().__init__(use_cache)
        self._load_data()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        try:
            meta = self.metadata[idx]
            video_path = meta["video_path"]
            label = meta["label"]

            cache_path = self._cache_path(video_path)

            # ===== LOAD IMAGE FROM CACHE =====
            if self.use_cache and os.path.exists(cache_path):
                image_tensor = torch.load(cache_path)
                return image_tensor, torch.tensor(label)

            # ===== LOAD RAW IMAGE =====
            frame = load_first_frame(meta)
            video_array = np.array([frame], dtype=np.uint8)

            # ===== AUGMENT + TO TENSOR =====
            aug = _apply_augmentations(video_array)
            aug = np.transpose(aug, (0, 3, 1, 2)) / 255.0
            image_tensor = torch.from_numpy(aug[0]).float()

            # ===== SAVE IMAGE ONLY =====
            if self.use_cache:
                torch.save(image_tensor, cache_path)

            return image_tensor, torch.tensor(label)
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return self.__getitem__(idx-1)

    def _load_data(self):
        metadata = []
        for dataset_name in os.listdir(ROOT_DIR):
            dataset_items = []
            dataset_path = os.path.join(ROOT_DIR, dataset_name)
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
                for file in data:
                    label = label_to_int[data[file]["media_type"]]
                    video_path = os.path.join(samples_dir, file)
                    item = {
                        "video_path": video_path,
                        "label": label,
                    }
                    if os.path.isdir(video_path):
                        item["video_frames"] = data[file]["video_frames"]
                    dataset_items.append(item)

            if len(dataset_items) > 0 and len(dataset_items) < 5000:
                # duplicate dataset_items to reach 5000 items
                dataset_items = dataset_items * (5000 // len(dataset_items))
            metadata.extend(dataset_items)

        self.metadata = metadata


# ================== VAL DATASET ==================
class ValDataset(BaseDataset):
    def __init__(self, use_cache=True, max_per_dataset=200):
        super().__init__(use_cache)
        self.max_per_dataset = max_per_dataset
        self._load_data()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        try:
            meta = self.metadata[idx]
            video_path = meta["video_path"]
            label = meta["label"]
            dataset_name = meta["dataset_name"]

            cache_path = self._cache_path(video_path)

            # ===== LOAD IMAGE FROM CACHE =====
            if self.use_cache and os.path.exists(cache_path):
                image_tensor = torch.load(cache_path)
                return image_tensor, torch.tensor(label), dataset_name

            # ===== LOAD RAW IMAGE =====
            frame = load_first_frame(meta)

            # ===== AUGMENT + TO TENSOR =====
            aug = _apply_augmentations(frame[None, ...])
            aug = np.transpose(aug, (0, 3, 1, 2)) / 255.0
            image_tensor = torch.from_numpy(aug[0]).float()

            # ===== SAVE IMAGE ONLY =====
            if self.use_cache:
                torch.save(image_tensor, cache_path)

            return image_tensor, torch.tensor(label), dataset_name
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return self.__getitem__(idx-1)

    def _load_data(self):
        metadata = []

        for dataset_name in os.listdir(ROOT_DIR):
            dataset_path = os.path.join(ROOT_DIR, dataset_name)
            dataset_items = []

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
                for file in data:
                    label = label_to_int[data[file]["media_type"]]
                    video_path = os.path.join(samples_dir, file)
                    item = {
                        "dataset_name": dataset_name,
                        "video_path": video_path,
                        "label": label,
                    }
                    if os.path.isdir(video_path):
                        item["video_frames"] = data[file]["video_frames"]
                    dataset_items.append(item)

            # dataset_items = random.sample(
            #     dataset_items,
            #     min(len(dataset_items), self.max_per_dataset)
            # )
            metadata.extend(dataset_items)

        self.metadata = metadata


# ================== BUILD CACHE ==================
if __name__ == "__main__":
    # print("Building shared image cache...")
    # for _ in tqdm(TrainDataset()):
    #     pass
    # print("✅ Done!")

    # dataloader
    from torch.utils.data import DataLoader
    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=16)
    
    for images, labels in tqdm(train_dataloader):
        pass