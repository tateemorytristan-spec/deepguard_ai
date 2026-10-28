
import random
import shutil
from pathlib import Path

# Project directory
PROJECT_DIR = Path(
    r"C:\Users\Pawan\Desktop\Final Project"
)

# Input: selected videos
INPUT_DIR = PROJECT_DIR / "selected_dataset"

# Output: train, validation, test
OUTPUT_DIR = PROJECT_DIR / "dataset_split"

# Split ratios: 70% train, 15% validation, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

random.seed(42)

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm"
}

CLASSES = ["real", "fake"]


def split_class(class_name):
    source_dir = INPUT_DIR / class_name

    videos = [
        file for file in source_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not videos:
        raise ValueError(
            f"No videos found in {source_dir}"
        )

    random.shuffle(videos)

    total = len(videos)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": videos[:train_end],
        "val": videos[train_end:val_end],
        "test": videos[val_end:]
    }

    for split_name, files in splits.items():
        destination = (
            OUTPUT_DIR / split_name / class_name
        )
        destination.mkdir(
            parents=True, exist_ok=True
        )

        for file in files:
            shutil.copy2(
                file, destination / file.name
            )

        print(
            f"{class_name} | {split_name}: "
            f"{len(files)} videos"
        )


if __name__ == "__main__":
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {INPUT_DIR}"
        )

    for class_name in CLASSES:
        split_class(class_name)

    print("\nDataset splitting completed!")
    print("Output folder:", OUTPUT_DIR)

