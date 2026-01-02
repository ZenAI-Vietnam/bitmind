import os
import json
from multiprocessing import Pool, cpu_count
import shutil
from PIL import Image
from tqdm import tqdm


def process_file(args):
    """Process a single file and return the JSON data if valid."""
    file, frames_dir, dataset_path, label, dataset_name = args
    try:
        # image = Image.open(os.path.join(frames_dir, file))
        # image.close()  # Close the image to free memory
        return json.dumps({
            "frame_path": os.path.join(frames_dir, file),
            "label": 1 if label == "synthetic" else 2 if label == "semisynthetic" else 0,
            "dataset": dataset_name.replace("-cropped", ""),
            "original_video_path": os.path.join(dataset_path, "_".join(file.split("_")[:-2]) + ".mp4"),
            "rotation_applied": 0
        }) + "\n"
    except:
        print(f"Error processing {os.pathp.join(frames_dir, file)}")
        os.system(f"rm -rf {os.path.join(frames_dir, file)}")
        return None


if __name__ == "__main__":
    num_workers = cpu_count()
    
    # read /mnt/video/benchmark_datasets.yaml
    import yaml
    with open("/mnt/video/benchmark_datasets.yaml", "r") as f:
        benchmark_datasets = yaml.load(f, Loader=yaml.FullLoader)
    dataset_names = [ds["name"] for ds in benchmark_datasets["datasets"]]
        
    for label in ["synthetic", "semisynthetic", "real"]:
        ROOT_DIR = f"/mnt/video/video/{label}"
        for dataset_name in os.listdir(ROOT_DIR):
            # if dataset_name not in ["FF23-fake", "UADFV-fake"]:
            #    # del jsonl file
            # #    os.system(f"rm -rf {os.path.join(ROOT_DIR, dataset_name, 'frame_paths_first_16.jsonl')}")
            #    continue
            dataset_path = os.path.join(ROOT_DIR, dataset_name)
            frames_dir = os.path.join(dataset_path, "frames_first_16")
            if not os.path.exists(os.path.join(dataset_path, "frames_first_16")):
                continue
            
            # Get all files to process
            files = os.listdir(frames_dir)
            # vid_006543_frame_014.png
            # keep file with name vid_{idx} and idx > 25000
            # files = [file for file in files if file.startswith("vid_") and int(file.split("_")[1]) >= 25000]
            # print(len(files))
            # # copy first item to {dataset_name}.png
            # shutil.copy(os.path.join(frames_dir, files[0]), f"samples/{dataset_name}.png")
            
            # Prepare arguments for multiprocessing
            args_list = [(file, frames_dir, dataset_path, label, dataset_name) for file in files]
            
            # Process files in parallel
            output_file = f"video/{label}/{dataset_name}/frame_paths_first_16.jsonl"
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_file, args_list),
                    total=len(files),
                    desc=f"Processing {dataset_name}"
                ))
            
            # Write results to file
            with open(output_file, "w") as f:
                for result in results:
                    if result is not None:
                        f.write(result)
            
            valid_count = sum(1 for r in results if r is not None)
            print(f"Saved {valid_count} frames to {output_file}")
            
                
