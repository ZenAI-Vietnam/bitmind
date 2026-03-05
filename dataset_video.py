import os
import json
import random
from tqdm import tqdm
from typing import Dict, Tuple, Callable, List

import torch
import numpy as np
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from transforms import apply_random_augmentations
from torchvision import transforms


def _apply_augmentations(image: np.ndarray, level_probs: Dict[int, float], 
                         mask: np.ndarray = None) -> bytes:
    """Apply random augmentations and convert to bytes."""
    image, mask, _, _ = apply_random_augmentations(image, mask=mask, level_probs=level_probs)
    return image


def train_transforms(image: np.ndarray, mask: np.ndarray = None) -> bytes:
    return _apply_augmentations(image, mask=mask, 
                                level_probs={0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0})

def val_transforms_lvl0(image: np.ndarray) -> bytes:
    return _apply_augmentations(image, level_probs={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0})


DATA_DIR = "/mnt/bitmind/datasets/datasets"
TEST_DATA_DIR = "/mnt/bitmind/test/datasets"


label_to_int = {
    "real": 0,
    "synthetic": 1,
    "semisynthetic": 1
}

SKIP_DATASETS = {
    "celeb-df-v2",
    "celeb-df-v1",
    # "rtfs-10k-uniface",
    "UADFV-fake",
    "UADFV-real",
    # "evalcrafter-t2v",
    # "lovora-real",
    # Thêm các dataset sai nhãn
    "Ivy-Fake-Youku_1M_10s-real",
    # "Ivy-Fake-ZeroScope-fake",
    # "Ivy-Fake-Kinetics-400-real",
    # "Ivy-Fake-Kinetics-400-val-real",
    "Ivy-Fake-SEINE-fake",
    "Ivy-Fake-OpenSora-fake",
    "Ivy-Fake-DynamicCrafter-fake"
    "video_archives",
}

def load_all_samples(data_dir: str = DATA_DIR, skip_datasets: set = SKIP_DATASETS):
    """Load all samples from data_dir, return dict[dataset_name] -> list of sample dicts."""
    all_data = {}
    for dataset_name in os.listdir(data_dir):
        if dataset_name in skip_datasets:
            continue
        dataset_path = os.path.join(data_dir, dataset_name)
        if dataset_name == "gasstation-generated-videos":
            week_dirs = os.listdir(dataset_path)
        else:
            week_dirs = [""]

        dataset_samples = []
        for week_dir in week_dirs:
            meta_path = os.path.join(dataset_path, week_dir, "sample_metadata.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            for video_name in metadata:
                dataset_samples.append({
                    "video_path": os.path.join(dataset_path, week_dir, "samples", video_name),
                    "label": metadata[video_name]["media_type"],
                    "dataset_name": dataset_name,
                })
        if dataset_samples:
            all_data[dataset_name] = dataset_samples
    return all_data

def split_train_val(data_dir: str = DATA_DIR,
                    train_limit: int = 2000,
                    real_bonus: int = 500,
                    val_limit: int = 100,
                    val_ratio: float = 0.05,
                    seed: int = 42):

    all_data = load_all_samples(data_dir)
    rng = random.Random(seed)

    train_samples = []
    val_samples = []

    for dataset_name, samples in all_data.items():
        shuffled = samples.copy()
        rng.shuffle(shuffled)

        # Số lượng val: lấy min giữa val_limit và tỷ lệ val_ratio
        val_count = min(val_limit, max(1, int(len(shuffled) * val_ratio)))
        
        is_real_dataset = all(s["label"] == "real" for s in samples)
        effective_limit = train_limit + real_bonus if is_real_dataset else train_limit
        
        val_split = shuffled[:val_count]
        train_pool = shuffled[val_count:]

        # Giới hạn train
        if effective_limit is not None:
            train_split = train_pool[:effective_limit]
        else:
            train_split = train_pool

        val_samples.extend(val_split)
        train_samples.extend(train_split)

    return train_samples, val_samples

class CustomDataset(Dataset):    
    def __init__(self, data: list, is_training: bool = False,
                 num_frames=16, height=224, width=224):
        self.is_training = is_training
        self.data = data
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.custom_transforms = train_transforms if is_training else val_transforms_lvl0

        self.aug_list = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    # Add this dummy method to prevent AttributeError when loading test dataset without calling _load_data
    def _dummy_video(self):
        # (T, C, H, W) float tensor of zeros
        x = torch.zeros((self.num_frames, 3, self.height, self.width), dtype=torch.float32)
        y = torch.tensor(0, dtype=torch.long)
        w = torch.tensor(0.0, dtype=torch.float32) # set weight to 0 => ignored in loss / metric
        return x, y, w

    def _load_video(self, video_path: str):
        """
        Load a video and return it as a resized uint8 numpy array of shape (T, H, W, C).
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        if total_frames == 0:
            raise ValueError(f"No frames in video")
        max_frames = 16
        frames = []
        for i in range(min(max_frames, total_frames)):
            frame = vr[i].asnumpy()  # Already RGB, shape (H, W, C)
            if frame is None or frame.size == 0:
                raise ValueError(f"Skipping invalid frame at index {i}")
            # Deterministic resize per-frame
            frames.append(frame)
        if len(frames) < max_frames:
            last_frame = frames[-1]
            for i in range(len(frames), max_frames):
                frames.append(last_frame)
        video_array = np.array(frames, dtype=np.uint8)  # THWC uint8 RGB
        aug_video_array = self.custom_transforms(video_array)
        aug_tchw = np.transpose(aug_video_array, (0, 3, 1, 2))

        return aug_tchw

    # def __getitem__(self, idx: int):
    #     try:
    #         item = self.data[idx]
    #         video_path = item["video_path"]
    #         video_array = self._load_video(video_path)
    #         frames = []
    #         for frame in video_array:
    #             frame = frame.transpose(1, 2, 0)
    #             frame = self.aug_list(frame)
    #             frames.append(frame)
    #         video_array = torch.stack(frames)

    #         label = item["label"]
    #         int_label = label_to_int[label]
    #         label = torch.tensor(int_label, dtype=torch.long)
    #         return video_array, label
    #     except Exception as e:
    #         print(f"Error loading item {item['video_path']}: {e}")
    #         return self.__getitem__(0)
    
    # Use this version of __getitem__ to handle error cases without crashing the dataloader
    def __getitem__(self, idx: int):
        max_tries = 8
        last_err = None

        for _ in range(max_tries):
            item = self.data[idx]
            video_path = item["video_path"]
            try:
                video_array = self._load_video(video_path)  # bạn đảm bảo trả về (T,C,H,W) uint8/float OK

                frames = []
                for frame in video_array:              # frame: C,H,W
                    frame = frame.transpose(1, 2, 0)   # H,W,C (numpy)
                    if frame.shape[-1] > 3:
                        frame = frame[..., :3]
                    elif frame.shape[-1] == 1:
                        frame = np.repeat(frame, 3, axis=-1)

                    frame = self.aug_list(frame)       # tensor C,H,W float normalized
                    frames.append(frame)

                x = torch.stack(frames)  # T,C,H,W

                y = torch.tensor(label_to_int[item["label"]], dtype=torch.long)
                w = torch.tensor(1.0, dtype=torch.float32)  # valid sample
                return x, y, w

            except Exception as e:
                last_err = e
                idx = random.randrange(len(self.data))
                
        print(f"Failed to load video after {max_tries} attempts, last error: {last_err}")
        return self._dummy_video()

    def __len__(self) -> int:
        return len(self.data)


class TrainDataset(CustomDataset):
    def __init__(self):
        train_samples, _ = split_train_val()
        super().__init__(data=train_samples, is_training=True)


class ValDataset(CustomDataset):
    def __init__(self):
        # Use validation transforms (no heavy random augs)
        _, val_samples = split_train_val()
        super().__init__(data=val_samples, is_training=False)

    # def __getitem__(self, idx: int):
    #     try:
    #         item = self.data[idx]
    #         video_path = item["video_path"]

    #         video_array = self._load_video(video_path)
    #         frames = []
    #         for frame in video_array:
    #             frame = frame.transpose(1, 2, 0)
    #             frame = self.aug_list(frame)
    #             frames.append(frame)
    #         video_array = torch.stack(frames)
    #         label = item["label"]
    #         int_label = label_to_int[label]
    #         label = torch.tensor(int_label, dtype=torch.long)
    #         dataset_name = item["dataset_name"]
    #         return video_array, label, dataset_name
    #     except Exception as e:
    #         print(f"Error loading item {item['video_path']}: {e}")
    #         return self.__getitem__(0)
    
    def _dummy_video(self):
        # (T, C, H, W) float tensor of zeros
        x = torch.zeros((self.num_frames, 3, self.height, self.width), dtype=torch.float32)
        y = torch.tensor(0, dtype=torch.long)
        w = torch.tensor(0.0, dtype=torch.float32) # set weight to 0 => ignored in loss / metric
        dataset_name = "dummy"
        return x, y, dataset_name, w
    
    def __getitem__(self, idx: int):
        max_tries = 8
        last_err = None

        for _ in range(max_tries):
            item = self.data[idx]
            video_path = item["video_path"]
            try:
                video_array = self._load_video(video_path)  # bạn đảm bảo trả về (T,C,H,W) uint8/float OK

                frames = []
                for frame in video_array:              # frame: C,H,W
                    frame = frame.transpose(1, 2, 0)   # H,W,C (numpy)
                    if frame.shape[-1] > 3:
                        frame = frame[..., :3]
                    elif frame.shape[-1] == 1:
                        frame = np.repeat(frame, 3, axis=-1)

                    frame = self.aug_list(frame)       # tensor C,H,W float normalized
                    frames.append(frame)

                x = torch.stack(frames)  # T,C,H,W

                y = torch.tensor(label_to_int[item["label"]], dtype=torch.long)
                w = torch.tensor(1.0, dtype=torch.float32)  # valid sample
                dataset_name = item["dataset_name"]
                return x, y, dataset_name, w

            except Exception as e:
                last_err = e
                # chọn index khác để retry, tránh dính mãi 1 file lỗi
                idx = random.randrange(len(self.data))
                
        print(f"Failed to load video after {max_tries} attempts, last error: {last_err}")
        return self._dummy_video()


class TestDataset(CustomDataset):
    def __init__(self):
        all_test_data = load_all_samples(data_dir=TEST_DATA_DIR, skip_datasets=set())
        test_samples = [s for samples in all_test_data.values() for s in samples]
        super().__init__(data=test_samples, is_training=False)
        

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import lightning as L
    L.seed_everything(42)
    
    train_dataset = TrainDataset()
    val_dataset = ValDataset()
    
    # Verify no overlap
    train_paths = {item["video_path"] for item in train_dataset.data}
    val_paths = {item["video_path"] for item in val_dataset.data}
    overlap = train_paths & val_paths
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Overlap: {len(overlap)}")
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping samples!"
    
    dataloader = DataLoader(train_dataset, batch_size=8, num_workers=8)
    for video_array, label, w in dataloader:
        print(video_array.shape, label, w)
        break