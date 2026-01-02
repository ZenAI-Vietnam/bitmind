import os
from typing import Dict

from gasbench.dataset.download import download_and_extract
from gasbench.dataset.config import discover_benchmark_datasets
from gasbench.dataset.cache import save_sample_to_cache, save_dataset_cache_files


DATASET_DIR = "video"
REAL_DATASET_DIR = "video/real"
SYNTHETIC_DATASET_DIR = "video/synthetic"
SEMI_SYNTHETIC_DATASET_DIR = "video/semisynthetic"

datasets = discover_benchmark_datasets(
    mode="full", 
    modality="video",
    gasstation_only=False,
    no_gasstation=True,
    yaml_path="new_benchmark_datasets.yaml"
)


def extract_sample_metadata(sample: Dict) -> Dict:
    """Extract metadata from sample for caching.
    
    Args:
        sample: Sample dictionary with media data and metadata
        
    Returns:
        Dictionary with extracted metadata fields
    """
    metadata = {
        "source_file": sample.get("source_file", ""),
        "model_name": sample.get("model_name", ""),
        "media_type": sample.get("media_type", ""),
    }
    
    # gasstation-specific fields
    for field in ["iso_week", "generator_hotkey", "generator_uid"]:
        if field in sample:
            metadata[field] = sample.get(field)

    return metadata


for idx, dataset in enumerate(datasets):
    # print(dataset)
    if dataset.modality != "video":
        continue
    if dataset.media_type != "abc":
        # if dataset.name not in ["kinetics400"]:
        #     continue
        # print(dataset.name)
        samples_dir = f"video/{dataset.media_type}/{dataset.name}/samples"
        os.makedirs(samples_dir, exist_ok=True)
        dataset.temp_dir = "temp"
        dataset.cache_dir = f"video/{dataset.media_type}"
        sample_metadata = {}
        next_index = 0
        media_per_archive = -1
        archives_per_dataset = -1
        for sample in download_and_extract(
            dataset, 
            media_per_archive=media_per_archive,
            archives_per_dataset=archives_per_dataset,
            temp_dir="temp",
            force_download=True,
            cache_dir=f"video/{dataset.media_type}",
            # target_week="2025W50",
            # current_week_only=True,
            num_weeks=2
        ):
            filename = save_sample_to_cache(
                sample, dataset, samples_dir, next_index
            )
            if filename:
                next_index += 1
    
            metadata = extract_sample_metadata(sample)
            print(metadata)
            break
        break
    # exit()