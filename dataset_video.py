import os
import json
import random
from tqdm import tqdm
from typing import Dict, Tuple, Callable, List

import torch
import numpy as np
import cv2
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from transforms import apply_random_augmentations
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


def _apply_augmentations(image: np.ndarray, level_probs: Dict[int, float], 
                         mask: np.ndarray = None) -> bytes:
    """Apply random augmentations and convert to bytes."""
    image, mask, _, _ = apply_random_augmentations(image, mask=mask, level_probs=level_probs)
    return image


def train_transforms(image: np.ndarray, mask: np.ndarray = None) -> bytes:
    return _apply_augmentations(image, mask=mask, 
                                level_probs={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0})


def val_transforms(image: np.ndarray) -> bytes:
    return _apply_augmentations(image, level_probs={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

def val_transforms_lvl0(image: np.ndarray) -> bytes:
    return _apply_augmentations(image, level_probs={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0})

def val_transforms_lvl1(image: np.ndarray) -> bytes:
    return _apply_augmentations(image, level_probs={0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0})

def val_transforms_lvl2(image: np.ndarray) -> bytes:
    return _apply_augmentations(image, level_probs={0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0})

def val_transforms_lvl3(image: np.ndarray) -> bytes:
    return _apply_augmentations(image, level_probs={0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0})



main_data = []
main_dataset_dict = {}
with open("all_video.jsonl", 'r') as f:
    for line in tqdm(f, desc="Loading dataset"):
        d = json.loads(line)
        dataset = d["dataset"]
        video = d["original_video_path"]
        if dataset not in main_dataset_dict:
            main_dataset_dict[dataset] = {}
        if video not in main_dataset_dict[dataset]:
            main_dataset_dict[dataset][video] = []
        main_dataset_dict[dataset][video].append(d)
for dataset_name in main_dataset_dict.keys():
    print(f"{dataset_name}: {len(main_dataset_dict[dataset_name])}")


class CustomDataset(Dataset):    
    def __init__(self, dataset_file: str, is_training: bool = False, label_map: Dict[int, int] = None, weights: Dict[str, float] = None):
        self.dataset_file = dataset_file
        self.is_training = is_training
        self.label_map = label_map
        self._load_data(weights)
        self.custom_transforms = train_transforms if is_training else val_transforms

    def _load_data(self, weights: Dict[str, float]):
        data = []
    
        for dataset_name, video_dict in main_dataset_dict.items():
            loop = int(weights.get(dataset_name, 1))
            unique_videos = list(video_dict.keys())
            limit_per_dataset = self.limit_per_dataset if self.limit_per_dataset != -1 else len(unique_videos)
            for _ in range(loop):
                sampled_videos = random.sample(unique_videos, min(limit_per_dataset, len(unique_videos)))

                for video in sampled_videos:
                    frames = video_dict[video]
                    if len(frames) < 4:
                        continue
                    # sort frames by frame_path
                    frames.sort(key=lambda x: x['frame_path'])
                    data.append(frames[:4])
        self.data = data

    def load_image(self, item: dict):
        # image = Image.open(item['frame_path']).convert("RGB")
        image = cv2.imread(item['frame_path'], cv2.IMREAD_COLOR)
        return image
    
    def __getitem__(self, idx: int):
        item = self.data[idx]
        image = self.load_image(item)
        
        image = image.copy()  # Fix: Make a copy to avoid negative strides (in case of future transforms)
        image = torch.from_numpy(image)
        
        label = item['label']
        if self.label_map is not None:
            label = self.label_map[label]
        label = torch.tensor(label, dtype=torch.long)

        source = item["dataset"]
        return image, label, source
    
    def __len__(self) -> int:
        return len(self.data)


class TrainDataset(CustomDataset):
    def __init__(self, dataset_file: str, label_map: Dict[int, int] = None, weights: Dict[str, float] = {}, limit_per_dataset=-1):
        self.limit_per_dataset = limit_per_dataset
        super().__init__(dataset_file, is_training=True, label_map=label_map, weights=weights)

    def __getitem__(self, idx: int):
        try:
            item = self.data[idx]
            frames = []
            for frame in item:
                image = self.load_image(frame)
                frames.append(image)
                video_array = np.array(frames, dtype=np.uint8)
        except Exception as e:
            item = self.data[idx - 1] if idx - 1 >= 0 else self.data[idx + 1]
            frames = []
            for frame in item:
                image = self.load_image(frame)
                frames.append(image)
                video_array = np.array(frames, dtype=np.uint8)
        
        aug_thwc = self.custom_transforms(video_array)
        video_array = np.transpose(aug_thwc / 255.0, (0, 3, 1, 2))
        video_array = torch.from_numpy(video_array.copy()).float()
        # Normalize with ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        video_array = (video_array - mean) / std
        if video_array.shape[0] < 8:
            video_array = torch.cat([video_array, torch.zeros((8 - video_array.shape[0], video_array.shape[1], video_array.shape[2], video_array.shape[3]))], dim=0)
        label = item[0]['label']
        if self.label_map is not None:
            label = self.label_map[label]
        label = torch.tensor(label, dtype=torch.long)

        return video_array, label

    def _load_data(self, weights: Dict[str, float]):
        data = []
    
        for dataset_name, video_dict in main_dataset_dict.items():
            loop = int(weights.get(dataset_name, 1))
            unique_videos = list(video_dict.keys())
            limit_per_dataset = self.limit_per_dataset if self.limit_per_dataset != -1 else len(unique_videos)
            for _ in range(loop):
                if limit_per_dataset >= len(unique_videos):
                    sampled_videos = unique_videos
                else:
                    sampled_videos = random.sample(unique_videos, min(limit_per_dataset, len(unique_videos)))

                for video in sampled_videos:
                    frames = video_dict[video]
                    frames.sort(key=lambda x: x['frame_path'])
                    data.append(frames[:8])
        self.data = data


class ValDataset(CustomDataset):
    def __init__(self, dataset_file: str, label_map: Dict[int, int] = None):
        self.custom_transforms_list = [
            val_transforms_lvl0,
            # val_transforms_lvl1,
            # val_transforms_lvl2,
            # val_transforms_lvl3,
        ]
        super().__init__(dataset_file, is_training=False, label_map=label_map, weights={})

    def __getitem__(self, idx: int):
        item = self.data[idx]
        level = idx % len(self.custom_transforms_list)
        frames = []
        for frame in item:
            image = self.load_image(frame)
            frames.append(image)
        video_array = np.array(frames, dtype=np.uint8)
        aug_thwc = self.custom_transforms_list[level](video_array)
        video_array = np.transpose(aug_thwc / 255.0, (0, 3, 1, 2))
        video_array = torch.from_numpy(video_array.copy()).float()
        if video_array.shape[0] < 8:
            video_array = torch.cat([video_array, torch.zeros((8 - video_array.shape[0], video_array.shape[1], video_array.shape[2], video_array.shape[3]))], dim=0)
        # Normalize with ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        video_array = (video_array - mean) / std
        label = item[0]['label']
        if self.label_map is not None:
            label = self.label_map[label]
        label = torch.tensor(label, dtype=torch.long)
        source = item[0]['dataset']

        return video_array, label, source, level

fix_val_dataset_dict = {}
with open("val_video.jsonl", 'r') as f:
    for line in f:
        d = json.loads(line)
        dataset = d["dataset"]
        video = d["original_video_path"]
        if dataset not in fix_val_dataset_dict:
            fix_val_dataset_dict[dataset] = {}
        if video not in fix_val_dataset_dict[dataset]:
            fix_val_dataset_dict[dataset][video] = []
        fix_val_dataset_dict[dataset][video].append(d)

class FixValDataset(ValDataset):
    def __init__(self, dataset_file: str, label_map: Dict[int, int] = None):
        super().__init__(dataset_file, label_map=label_map)

    def _load_data(self, weights: Dict[str, float]):
        data = []

        for dataset_name, video_dict in fix_val_dataset_dict.items():
            for video in video_dict.keys():
                frames = video_dict[video]
                frames.sort(key=lambda x: x['frame_path'])
                data.append(frames[:8])
        self.data = data


class RandomValDataset(ValDataset):
    def __init__(self, dataset_file: str, label_map: Dict[int, int] = None):
        super().__init__(dataset_file, label_map=label_map)

    def _load_data(self, weights: Dict[str, float]):
        data = []
    
        for dataset_name, video_dict in main_dataset_dict.items(): 
            unique_videos = list(video_dict.keys())
            sampled_videos = random.sample(unique_videos, min(200, len(unique_videos)))
    
            for video in sampled_videos:
                frames = video_dict[video]
                frames.sort(key=lambda x: x['frame_path'])
                data.append(frames[:8])
        self.data = data


# if __name__ == "__main__":
#     dataset = TrainDataset(dataset_file="all_first_16_3_classes_overfit_expended.jsonl")
#     print(len(dataset))
#     print(dataset[0])