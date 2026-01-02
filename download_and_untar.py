import multiprocessing
import os
import time
import subprocess
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm
import sys
import json
import dotenv

dotenv.load_dotenv()

args = sys.argv
if len(args) != 4:
    print("Usage: python download_and_untar.py <bucket_name> <prefix> <version_id_file_path>")
    print("Example: python download_and_untar.py bitmind-data image current_version_ids.jsonl")
    sys.exit(1)

bucket_name = args[1]
prefix = args[2]
version_id_file_path = args[3]

# Replace with your actual credentials
access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
if not access_key_id or not secret_access_key:
    print("AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is not set")
    print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the environment variables")
    sys.exit(1)

# Initialize the S3 client
s3 = boto3.client(
    service_name="s3",
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    region_name="us-east-1",
)

# Transfer config for downloads
transfer_config = TransferConfig(
    multipart_threshold=512 * 1024 * 1024,  # 100 MB threshold for multipart
    multipart_chunksize=512 * 1024 * 1024,  # 100 MB chunk size
    max_concurrency=128,  # Adjust based on instance resources
    use_threads=True
)

# List objects in the bucket
# response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
# online_files = [obj['Key'] for obj in response.get('Contents', [])]
# online_files = list(set(online_files))
# online_files.sort()

# Load object list from current_version_ids.jsonl
with open(version_id_file_path, 'r') as f:
    online_files = [json.loads(line.strip()) for line in f]

file_list_files = sorted([entry for entry in online_files if entry['key'].endswith("_file_list.txt")], key=lambda x: x['key'])
tar_files = sorted([entry for entry in online_files if entry['key'].endswith(".tar")], key=lambda x: x['key'])

data_path = "./"  # Ensure this points to a high-throughput volume, e.g., /mnt/ebs/

def list_all_files(folder_path):
    results = []
    for root, _, files in os.walk(folder_path):
        results.extend(
            [
                '/' + os.path.join(root, f).replace(folder_path, "", 1).lstrip("/")
                for f in files
            ]
        )
    return results

def check_untar_folder(tar_path):
    """Verify extracted folder against origin file list."""
    folder_path = tar_path.replace(".tar", "/")
    if not os.path.isdir(folder_path):
        return False, [], []
    
    origin_file_list_path = tar_path.replace(".tar", "_file_list.txt")
    with open(origin_file_list_path, "r") as f:
        origin_list = f.read().splitlines()
    origin_set = set(origin_list)
    
    all_files = list_all_files(folder_path)
    all_files_set = set(all_files)
    
    return origin_set == all_files_set, origin_set - all_files_set, all_files_set - origin_set

def download_from_s3(key, version_id, local_path):
    """Download file from s3 with progress tracking."""
    start = time.time()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    file_size = s3.head_object(Bucket=bucket_name, Key=key)["ContentLength"]
    with tqdm(total=file_size, unit="B", unit_scale=True, desc=f"Downloading {key}") as progress_bar:
        def progress_callback(bytes_amount):
            progress_bar.update(bytes_amount)
        s3.download_file(bucket_name, key, local_path, ExtraArgs={'VersionId': version_id}, Callback=progress_callback, Config=transfer_config)
    print(f"Downloaded {key} to {local_path} in {time.time() - start:.2f} seconds")


def stream_download_and_untar(key, version_id, dirname):
    """Stream download and untar using a FIFO pipe for multipart support without saving the full tar file."""
    os.makedirs(dirname, exist_ok=True)
    safe_key = key.replace('/', '_')
    fifo_path = os.path.join(dirname, f"temp_fifo_{safe_key}")
    if os.path.exists(fifo_path):
        os.unlink(fifo_path)
    os.mkfifo(fifo_path)

    process = subprocess.Popen(['tar', 'xf', fifo_path, '-C', dirname])
    file_size = s3.head_object(Bucket=bucket_name, Key=key, VersionId=version_id)["ContentLength"]
    try:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc=f"Streaming {key}") as progress_bar:
            def progress_callback(bytes_amount):
                progress_bar.update(bytes_amount)
            s3.download_file(
                bucket_name,
                key,
                fifo_path,
                ExtraArgs={'VersionId': version_id},
                Callback=progress_callback,
                Config=transfer_config,
            )
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Streaming untar failed with code {process.returncode}")
    finally:
        if os.path.exists(fifo_path):
            os.unlink(fifo_path)


def download_from_s3_and_untar(entry):
    """Download, stream tarballs into tar, and verify extractions."""
    key = entry['key']
    version_id = entry['version_id']
    local_path = os.path.join(data_path, key)
    dirname = os.path.dirname(local_path)
    os.makedirs(dirname, exist_ok=True)
    
    if key.endswith('.tar'):
        max_retries = 5
        file_size = s3.head_object(Bucket=bucket_name, Key=key, VersionId=version_id)["ContentLength"]
        for attempt in range(max_retries):
            try:
                # Check if already extracted and verified
                verified, missing, extra = check_untar_folder(local_path)
                if verified:
                    print(f"✓ Already verified: {local_path}")
                    break
                
                print(f"Streaming download and extracting {key} (attempt {attempt+1}/{max_retries})...")
                stream_download_and_untar(key, version_id, dirname)

                # Verify
                verified, missing, extra = check_untar_folder(local_path)
                if verified:
                    print(f"✓ Verification passed: {local_path}")
                    break
                else:
                    print(f"✗ Verification failed: Missing={len(missing)}, Extra={len(extra)}")
                    time.sleep(2)
            except Exception as e:
                print(f"✗ Error processing {key}: {e}. Retry {attempt+1}/{max_retries}")
                time.sleep(2)
    else:
        if not os.path.exists(local_path):
            download_from_s3(key, version_id, local_path)


# Process files in parallel using a worker pool
if __name__ == "__main__":
    with Pool(processes=10) as pool:  # Adjust based on resources (e.g., 20 for higher concurrency)
        # download all file_list first
        pool.map(download_from_s3_and_untar, file_list_files)

        # then download all tar files
        pool.map(download_from_s3_and_untar, tar_files)
