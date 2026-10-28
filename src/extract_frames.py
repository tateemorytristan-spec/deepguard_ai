

import cv2
import random
from pathlib import Path

PROJECT_DIR = Path(
    r"C:\Users\Pawan\Desktop\Final Project"
)

INPUT_DIR = PROJECT_DIR / "dataset_split"
OUTPUT_DIR = PROJECT_DIR / "extracted_frames"

FRAMES_PER_VIDEO = 60
VIDEOS_PER_CLASS = 100
SEED = 42

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm"
}

SPLITS = ["train", "val", "test"]
CLASSES = ["real", "fake"]

random.seed(SEED)


def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR opening:", video_path)
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        print("ERROR reading:", video_path)
        return 0

    count = min(FRAMES_PER_VIDEO, total_frames)

    indices = [
        round(i * (total_frames - 1) / max(count - 1, 1))
        for i in range(count)
    ]

    output_folder.mkdir(parents=True, exist_ok=True)
    saved = 0

    for i, frame_number in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if success:
            path = output_folder / f"frame_{i + 1:03d}.jpg"
            if cv2.imwrite(str(path), frame):
                saved += 1

    cap.release()
    return saved


def main():
    total_videos = 0
    total_frames = 0

    # Collect videos across all splits, separately by class
    for label in CLASSES:
        candidates = []

        for split in SPLITS:
            source = INPUT_DIR / split / label

            if not source.is_dir():
                print("Missing folder:", source)
                continue

            candidates.extend(
                (split, video)
                for video in source.iterdir()
                if video.is_file()
                and video.suffix.lower() in VIDEO_EXTENSIONS
            )

        if len(candidates) < VIDEOS_PER_CLASS:
            raise ValueError(
                f"Only {len(candidates)} {label} videos found; "
                f"need {VIDEOS_PER_CLASS}."
            )

        selected = random.sample(candidates, VIDEOS_PER_CLASS)
        print(f"\nSelected {label}: {len(selected)} videos")

        for split, video in selected:
            output_folder = (
                OUTPUT_DIR / split / label / video.stem
            )

            saved = extract_frames(video, output_folder)
            total_videos += 1
            total_frames += saved

            print(f"{split}/{label}/{video.name}: "
                  f"{saved} frames saved")

    print("\nFRAME EXTRACTION COMPLETED")
    print("Videos selected:", VIDEOS_PER_CLASS * 2)
    print("Videos processed:", total_videos)
    print("Frames saved:", total_frames)
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()