
import random
import shutil
from pathlib import Path

# Project paths
PROJECT_DIR = Path(
    r"C:\Users\Pawan\Desktop\Final Project"
)

REAL_DIR = PROJECT_DIR / "Real-vid"
FAKE_DIR = PROJECT_DIR / "Fake-vid"

OUTPUT_DIR = PROJECT_DIR / "selected_dataset"

# Number of videos required per class
VIDEOS_PER_CLASS = 100

# Reproducible random selection
random.seed(42)

# Supported video formats
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def select_videos(source_dir, label):
    videos = [
        p for p in source_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if len(videos) < VIDEOS_PER_CLASS:
        raise ValueError(
            f"Only {len(videos)} videos found in {source_dir}"
        )

    selected = random.sample(videos, VIDEOS_PER_CLASS)

    destination = OUTPUT_DIR / label
    destination.mkdir(parents=True, exist_ok=True)

    for index, video in enumerate(selected, start=1):
        new_name = f"{label}_{index:03d}{video.suffix.lower()}"
        shutil.copy2(video, destination / new_name)

    print(f"{label}: {len(selected)} videos copied")


if __name__ == "__main__":
    select_videos(REAL_DIR, "real")
    select_videos(FAKE_DIR, "fake")

    print("\nDataset preparation completed!")
    print("Total videos:", 2 * VIDEOS_PER_CLASS)
    print("Saved at:", OUTPUT_DIR)

