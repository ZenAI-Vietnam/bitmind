import os
import json
import subprocess
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT_DIR = "video"


def run_cmd(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def extract_frames_ffmpeg(video_path, frames_dir, num_frames, dataset_name, use_fps=False):
    """Extract frames using either first-K frames or 1fps depending on use_fps flag."""
    os.makedirs(frames_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # ---------------------------
    # Chọn chế độ trích frame
    # ---------------------------
    if use_fps:
        # Lấy 1 frame mỗi giây
        vf_expr = "fps=1"
    else:
        # Lấy num_frames frame đầu tiên
        select_expr = f"select='lte(n,{num_frames-1})'"
        vf_expr = select_expr

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf_expr,
        "-vsync", "vfr",
        os.path.join(frames_dir, f"{video_name}_frame_%03d.png")
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ---------------------------
    # Thu thập thông tin frame
    # ---------------------------
    frame_files = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.startswith(video_name) and f.endswith(".png")
    ])

    frame_data = []
    for frame_path in frame_files:
        entry = {
            "frame_path": frame_path,
            "label": 0 if "real" in video_path else 1,
            "dataset": dataset_name.replace("_old", ""),
            "original_video_path": video_path,
            "rotation_applied": 0,
        }
        frame_data.append(entry)

    return frame_data

def extract_frames_opencv(video_path, frames_dir, num_frames, dataset_name, use_fps=False):
    """Extract frames using OpenCV."""
    os.makedirs(frames_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # check if frames are already extracted
    if os.path.exists(os.path.join(frames_dir, f"{video_name}_frame_001.png")):
        return []

    # if use_fps:
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     print(f"FPS: {fps}")
    cap = cv2.VideoCapture(video_path)
    num_frames_extracted = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{video_name}_frame_{(num_frames_extracted+1):03d}.png"), frame)
        num_frames_extracted += 1
        if num_frames_extracted >= num_frames:
            break
    cap.release()
    
    # Thu thập thông tin frame giống như extract_frames_ffmpeg
    frame_files = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.startswith(video_name) and f.endswith(".png")
    ])

    frame_data = []
    for frame_path in frame_files:
        entry = {
            "frame_path": frame_path,
            "label": 0 if "real" in video_path else 1,
            "dataset": dataset_name.replace("_old", ""),
            "original_video_path": video_path,
            "rotation_applied": 0,
        }
        frame_data.append(entry)

    return frame_data

def process_partial():
    """Xử lý từng dataset, ghi ra file JSONL ngay sau khi xong mỗi video."""
    # 🔥 Thêm tùy chọn chế độ trích frame tại đây
    use_fps = False         # ← đổi thành True để lấy 1fps
    num_frames = 16         # ← chỉ dùng khi use_fps=False

    for label in ["real", "semisynthetic", "synthetic"]:
        label_dir = os.path.join(ROOT_DIR, label)
        for dataset_name in os.listdir(label_dir):

            # # # Chỉ chạy cho dataset này (như code gốc)
            # if "text-2-video-human-preferences" not in dataset_name:
            #     continue
            if dataset_name in ["dfd-real", "dfd-fake", "FF23-real", "FF23-fake", "UADFV-real", "UADFV-fake"]:
                continue
            
            # if dataset_name not in ["3massiv"]:
            #     continue

            dataset_path = os.path.join(label_dir, dataset_name)
            print(f"🟢 Extracting frames for dataset: {dataset_name}")

            video_dir = os.path.join(dataset_path, "samples")
            if not os.path.exists(video_dir) or len(os.listdir(video_dir)) <= 20:
                continue

            frames_dir = os.path.join(dataset_path, "frames_first_16")
            output_path = os.path.join(dataset_path, "frame_paths_first_16.jsonl")

            # if not os.path.exists(output_path):
            #     print(f"⚠️ Output path not found: {output_path}")
            #     continue

            # if os.path.exists(frames_dir):
            #     print(f"⚠️ Frames directory already exists: {frames_dir}")
            #     os.system(f"rm -rf {frames_dir}")

            os.makedirs(frames_dir, exist_ok=True)
            video_files = [
                os.path.join(video_dir, v)
                for v in os.listdir(video_dir)
                if v.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".gif"))
            ]
            print(f"Number of videos: {len(video_files)}")

            max_workers = 96

            with open(output_path, "a") as f:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:

                    futures = {
                        executor.submit(
                            extract_frames_opencv, v, frames_dir, num_frames, dataset_name, use_fps
                        ): v
                        for v in video_files
                    }

                    for future in tqdm(as_completed(futures), total=len(futures),
                                       desc=f"Extracting frames for {dataset_name}"):

                        video_path = futures[future]
                        try:
                            result = future.result()
                            if not result:
                                continue

                            # ghi metadata
                            for entry in result:
                                f.write(json.dumps(entry) + "\n")
                            f.flush()

                            # xóa video sau khi extract xong
                            os.remove(video_path)
                            print(f"🗑️ Deleted video after extraction: {video_path}")

                        except Exception as e:
                            print(f"❌ Error processing {video_path}: {e}")
                            os.remove(video_path)
                            print(f"🗑️ Deleted video after error: {video_path}")

            print(f"✅ Done! Saved metadata to: {output_path}")
            os.system(f"rm -rf {video_dir}")


if __name__ == "__main__":
    process_partial()
