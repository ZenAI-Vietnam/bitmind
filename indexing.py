import os
import json
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
import numpy as np
import cv2
import timm
from PIL import ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transforms import apply_random_augmentations

# =========================
# GLOBAL CONFIG
# =========================
ROOT_DIR = "/mnt/video/video/datasets"
BATCH_SIZE = 64
NUM_WORKERS = 4   # Linux: 4–8 | Windows: 2–4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# AUGMENTATION (GIỮ NGUYÊN)
# =========================
def _apply_augmentations(
    image: np.ndarray,
    level_probs: Dict[int, float],
    mask: np.ndarray = None
):
    image, mask, _, _ = apply_random_augmentations(
        image, mask=mask, level_probs=level_probs
    )
    return image

def val_transforms_lvl0(image: np.ndarray):
    return _apply_augmentations(
        image,
        level_probs={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
    )

# =========================
# LOAD LABEL LIST
# =========================
label_to_int = {
    "real": 0,
    "synthetic": 1,
    "semisynthetic": 1
}
metadata_list = []
for dataset_name in os.listdir(ROOT_DIR):
    dataset_path = os.path.join(ROOT_DIR, dataset_name)
    print(dataset_name)
    if dataset_name == "gasstation-generated-videos":
        week_dirs = os.listdir(dataset_path)
    else:
        week_dirs = [""]
    for week_dir in week_dirs:
        metadata_file = os.path.join(dataset_path, week_dir, "sample_metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        print(len(metadata))
        for file in metadata:
            metadata_list.append({
                "video_path": os.path.join(dataset_path, "samples", file),
                "label": label_to_int[metadata[file]['media_type']]
            })

# =========================
# DATASET
# =========================
class FrameDataset(Dataset):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.metadata = metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.metadata[idx]['label']
        video_path = self.metadata[idx]['video_path']
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        video_array = frame[None].astype(np.uint8)
        aug_thwc = val_transforms_lvl0(video_array)
        aug_tchw = np.transpose(aug_thwc, (0, 3, 1, 2)) / 255.0
        video_array = torch.from_numpy(aug_tchw[0]).float()
        return video_array, label


# =========================
# MAIN
# =========================
def main():
    # Load metadata
    dataset = FrameDataset(metadata_list)

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Load backbone
    backbone = timm.create_model(
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained=True,
        num_classes=0
    )

    backbone.to(DEVICE)
    backbone.eval()

    label_dict = {0: [], 1: []}

    # Inference
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Inference"):
            images = images.to(DEVICE, non_blocking=True)

            outputs = backbone(images)  # [B, C]

            for out, lbl in zip(outputs, labels):
                label_dict[int(lbl)].append(out.cpu())

    # Save results
    for label, feats in label_dict.items():
        if len(feats) == 0:
            continue

        stack = torch.stack(feats)
        print(f"Label {label}: {stack.shape}")

        torch.save(stack, f"label_dict_{label}.pt")

if __name__ == "__main__":
    main()
