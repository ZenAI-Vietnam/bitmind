import requests
import json
import time
import yaml
API_URL = "https://gas.bitmind.ai/api/v1/sample-analytics/misclassified-samples"
RUN_ID = "b64c7935-9dc4-4ae2-8e23-b04db53bb0f6"
MODALITY = "video"
LIMIT = 500
MAX_OFFSET = 4200

output = []

for offset in range(0, MAX_OFFSET + 1, LIMIT):
    params = {
        "run_id": RUN_ID,
        "modality": MODALITY,
        "limit": LIMIT,
        "offset": offset
    }

    print(f"📌 Fetching offset={offset} …")
    resp = requests.get(API_URL, params=params)

    # Kiểm tra response
    if resp.status_code != 200:
        print(f"🚨 Error at offset={offset}: {resp.status_code}")
        break

    data = resp.json()
    # Nếu không còn dữ liệu thì dừng sớm
    if not data:
        print("⚠️ No more data, stopping early.")
        break

    output.extend(data)

    # (Tuỳ chọn) delay để tránh bị throttle
    time.sleep(0.2)

print(f"\n🔥 Total samples fetched: {len(output)}")

# Lưu vào JSON
with open("misclassified_samples.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# get list unique source_file of dataset_name
unique_source_files = {}
for sample in output:
    dataset_name = sample["dataset_name"]
    # if "holdout" in dataset_name:
    #     continue
    source_file = sample["source_file"]
    if dataset_name not in unique_source_files:
        unique_source_files[dataset_name] = []
    if source_file not in unique_source_files[dataset_name]:
        unique_source_files[dataset_name].append(source_file)

print(unique_source_files)
# delete _hashcode from source_file
for dataset_name, source_files in unique_source_files.items():
    # if "holdout" in dataset_name:
    #     continue
    for i, source_file in enumerate(source_files):
        ext = source_file.split(".")[-1]
        real_name = ".".join(source_file.split(".")[:-1])
        real_name = "_".join(real_name.split("_")[:-1])
        unique_source_files[dataset_name][i] = real_name + "." + ext
# save to yaml

with open("unique_source_files.yaml", "w", encoding="utf-8") as f:
    yaml.dump(unique_source_files, f)