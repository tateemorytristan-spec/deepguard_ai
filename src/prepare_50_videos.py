import os
import cv2
import numpy as np

# ============================================================
# PATHS
# ============================================================

PROJECT = r"C:\Users\Pawan\Desktop\Final Project"

REAL_FOLDER = os.path.join(
    PROJECT, "dataset_split", "train", "real"
)

FAKE_FOLDER = os.path.join(
    PROJECT, "dataset_split", "train", "fake"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT, "training_frames"
)

# ============================================================
# SETTINGS
# ============================================================

NUM_VIDEOS = 25
NUM_FRAMES = 60

# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frames(video_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        print("Could not read:", video_path)
        cap.release()
        return 0

    # Select 60 frames evenly across the video
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES,
        dtype=int
    )

    count = 0

    for i, frame_index in enumerate(frame_indices):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = cap.read()

        if not success:
            continue

        filename = f"frame_{i:03d}.jpg"

        output_path = os.path.join(
            output_folder,
            filename
        )

        cv2.imwrite(
            output_path,
            frame
        )

        count += 1

    cap.release()

    return count


# ============================================================
# PROCESS 25 VIDEOS FROM ONE CLASS
# ============================================================

def process_class(input_folder, class_name):

    videos = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".mp4")
    ])

    videos = videos[:NUM_VIDEOS]

    print()
    print("=" * 60)
    print(f"{class_name.upper()} VIDEOS")
    print(f"Found: {len(videos)}")
    print("=" * 60)

    for index, video in enumerate(videos, start=1):

        video_path = os.path.join(
            input_folder,
            video
        )

        video_name = os.path.splitext(video)[0]

        output_folder = os.path.join(
            OUTPUT_FOLDER,
            class_name,
            video_name
        )

        print(
            f"[{index}/{len(videos)}] {video}"
        )

        count = extract_frames(
            video_path,
            output_folder
        )

        print(
            f"    {count} frames extracted"
        )


# ============================================================
# RUN
# ============================================================

process_class(
    REAL_FOLDER,
    "real"
)

process_class(
    FAKE_FOLDER,
    "fake"
)

print()
print("=" * 60)
print("50-VIDEO FRAME EXTRACTION COMPLETED!")
print("=" * 60)