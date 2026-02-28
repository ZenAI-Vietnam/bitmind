import os
import json
from tqdm import tqdm

# "vid_000000.mp4": {
# "source_file": "VidProM_resolve_main_ms_videos_19.tar",
# "model_name": "",
# "media_type": "synthetic",
# "dataset_path": "bitmind/VidProM",
# "archive_filename": "VidProM_resolve_main_ms_videos_19.tar",
# "member_path": "ms_videos_all/ms-665f45c9-f981-5b38-ab2f-a413322c62ad.mp4",
# "source_kind": "huggingface"
# }
  
DATA_DIR = "/mnt/bitmind/datasets/datasets"
for dataset_name in os.listdir(DATA_DIR):
    if dataset_name == "gasstation-generated-videos":
        continue
    if os.path.exists(f"{DATA_DIR}/{dataset_name}/sample_metadata.json"):
        continue
    # label = "real" if "real " in dataset_name else "synthetic"
    if dataset_name in ["GenVideo-100k-I2VGEN_XL"]:
        label = "synthetic"
    elif dataset_name in ["GenVideo-100k-real"]:
        label = "real"
    else:
        continue
    print(dataset_name, label)
    samples_dir = os.path.join(DATA_DIR, dataset_name, "samples")
    sample_metadatas = {}
    for video_name in os.listdir(samples_dir):
        # do it for folder
        metadata = {
            "source_file": video_name,
            "model_name": "",
            "media_type": label,
            "dataset_path": dataset_name,
            "archive_filename": "",
            "member_path": "",
            "source_kind": "collection"
        }
        sample_metadatas[video_name] = metadata
    with open(f"{DATA_DIR}/{dataset_name}/sample_metadata.json", "w", encoding="utf-8") as f:
        json.dump(sample_metadatas, f, ensure_ascii=False, indent=4)