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
                                level_probs={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0})

def val_transforms_lvl0(image: np.ndarray) -> bytes:
    return _apply_augmentations(image, level_probs={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0})


DATA_DIR = "/mnt/bitmind/datasets/datasets"
TEST_DATA_DIR = "/mnt/bitmind/test/datasets"


label_to_int = {
    "real": 0,
    "synthetic": 1,
    "semisynthetic": 1
}


class CustomDataset(Dataset):    
    def __init__(self, is_training: bool = False, limit_per_dataset: int = None,
                 num_frames=16, height=224, width=224):
        self.is_training = is_training
        self.limit_per_dataset = limit_per_dataset
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self._load_data()
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

    def _load_data(self, data_dir: str = DATA_DIR):
        data = []
    
        for dataset_name in os.listdir(data_dir):
            if dataset_name in [
                # "eidon-video",
                # "semisynthetic-video",
                # "fakeparts-faceswap",
                # "dfd-real",
                # "dfd-fake",
                "celeb-df-v2",
                "celeb-df-v1",
                # "rtfs-10k-inswapper",
                "rtfs-10k-uniface",
                # "rtfs-10k-original_videos",
                "UADFV-fake",
                "UADFV-real",
                # Thêm các dataset không có trong round này
                "evalcrafter-t2v",
                "text-2-video-human-preferences-moonvalley-marey",
                "lovora-real"
            ]:
                continue
            dataset_sample = []
            dataset_path = os.path.join(data_dir, dataset_name)
            if dataset_name == "gasstation-generated-videos":
                week_dirs = os.listdir(dataset_path)
            else:
                week_dirs = [""]
            for week_dir in week_dirs:
                if not os.path.exists(os.path.join(dataset_path, week_dir, "sample_metadata.json")):
                    continue
                with open(os.path.join(dataset_path, week_dir, "sample_metadata.json"), "r") as f:
                    metadata = json.load(f)
                for video_name in metadata:
                    dataset_sample.append(
                        {
                            "video_path": os.path.join(dataset_path, week_dir, "samples", video_name),
                            "label": metadata[video_name]["media_type"],
                            "dataset_name": dataset_name,
                        }
                    )
                if self.limit_per_dataset is not None:
                    dataset_sample = random.sample(dataset_sample, min(self.limit_per_dataset, len(dataset_sample)))
                data.extend(dataset_sample)
        self.data = data

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
                    # ép 3 kênh để tránh 6 vs 3
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
                # chọn index khác để retry, tránh dính mãi 1 file lỗi
                idx = random.randrange(len(self.data))
                
        print(f"Failed to load video after {max_tries} attempts, last error: {last_err}")
        return self._dummy_video()

    def __len__(self) -> int:
        return len(self.data)


class TrainDataset(CustomDataset):
    def __init__(self):
        super().__init__(is_training=True, limit_per_dataset=2500)


class ValDataset(CustomDataset):
    def __init__(self):
        # Use validation transforms (no heavy random augs)
        super().__init__(is_training=False, limit_per_dataset=100)

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
                    # ép 3 kênh để tránh 6 vs 3
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
        super().__init__(is_training=False, limit_per_dataset=None)
        self._load_data(data_dir=TEST_DATA_DIR)
        

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import lightning as L
    L.seed_everything(42)
    train_dataset = TrainDataset()
    dataloader = DataLoader(train_dataset, batch_size=8, num_workers=8)
    for video_array, label in dataloader:
        print(video_array.shape, label)
        # break