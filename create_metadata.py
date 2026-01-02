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
  
DATA_DIR = "/mnt/video/video/datasets/FF23++-vid-real"
samples_dir = os.path.join(DATA_DIR, "samples")
sample_metadatas = {}
for video_name in tqdm(os.listdir(samples_dir), desc="Creating metadata"):
    # do it for folder
    metadata = {
        "source_file": video_name,
        "model_name": "",
        "media_type": "real",
        "dataset_path": "FF23++-vid-real",
        "archive_filename": "",
        "member_path": "",
        "source_kind": "collection"
    }
    sample_metadatas[video_name] = metadata
with open(f"{DATA_DIR}/sample_metadata.json", "w", encoding="utf-8") as f:
    json.dump(sample_metadatas, f, ensure_ascii=False, indent=4)