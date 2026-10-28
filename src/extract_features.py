
import os
import cv2
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from mtcnn import MTCNN

from torchvision.models import (
    resnext101_64x4d,
    ResNeXt101_64X4D_Weights
)


# ============================================================
# PATHS
# ============================================================

PROJECT_FOLDER = Path(
    r"C:\Users\Pawan\Desktop\Final Project"
)

REAL_FOLDER = (
    PROJECT_FOLDER / "dataset_split" / "train" / "real"
)

FAKE_FOLDER = (
    PROJECT_FOLDER / "dataset_split" / "train" / "fake"
)

OUTPUT_FOLDER = PROJECT_FOLDER / "training_features"


# ============================================================
# SETTINGS
# ============================================================

NUM_VIDEOS = 100
NUM_FRAMES = 60

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for label in ["real", "fake"]:
    (OUTPUT_FOLDER / label).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CHECK INPUT FOLDERS
# ============================================================

for folder in [REAL_FOLDER, FAKE_FOLDER]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Input folder not found: {folder}"
        )

print("Real folder:", REAL_FOLDER)
print("Fake folder:", FAKE_FOLDER)
print("Output folder:", OUTPUT_FOLDER)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing device:", device)


# ============================================================
# LOAD MTCNN
# ============================================================

print("\nLoading MTCNN face detector...")

detector = MTCNN()


# ============================================================
# LOAD RESNEXT101 64x4d
# ============================================================

print("Loading ResNeXt101 model...")

weights = ResNeXt101_64X4D_Weights.DEFAULT

model = resnext101_64x4d(
    weights=weights
)

# Remove ImageNet classification layer
model.fc = torch.nn.Identity()

model = model.to(device)
model.eval()

preprocess = weights.transforms()

print("Models loaded successfully.")


# ============================================================
# EXTRACT FEATURES FROM ONE VIDEO
# ============================================================

def extract_video_features(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR opening video:", video_path)
        return None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        print("ERROR reading video frames:", video_path)
        cap.release()
        return None

    # Select up to 60 evenly spaced frames
    count = min(NUM_FRAMES, total_frames)

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        count,
        dtype=int
    )

    all_features = []

    for frame_number in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_number)
        )

        success, frame = cap.read()

        if not success:
            continue

        # OpenCV BGR to RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect faces
        try:
            faces = detector.detect_faces(frame_rgb)
        except Exception as error:
            print("Face detection error:", error)
            continue

        if not faces:
            continue

        # Select the largest detected face
        face = max(
            faces,
            key=lambda item: (
                max(0, item["box"][2])
                * max(0, item["box"][3])
            )
        )

        x, y, w, h = map(int, face["box"])

        # Clip face coordinates to image boundaries
        height, width = frame_rgb.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = frame_rgb[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # Convert NumPy array to PIL image
        face_image = Image.fromarray(face_crop)

        # Preprocess for ResNeXt
        input_tensor = preprocess(
            face_image
        ).unsqueeze(0).to(device)

        # Extract 2048-dimensional features
        with torch.no_grad():
            features = model(input_tensor)

        features = (
            features.squeeze(0)
            .cpu()
            .numpy()
        )

        all_features.append(features)

    cap.release()

    if len(all_features) == 0:
        return None

    return np.asarray(
        all_features,
        dtype=np.float32
    )


# ============================================================
# PROCESS VIDEOS
# ============================================================

def process_videos(folder, label_name):

    videos = sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda path: path.name.lower()
    )

    # Select maximum 100 videos for each class
    videos = videos[:NUM_VIDEOS]

    print("\n" + "=" * 65)
    print(f"PROCESSING {label_name.upper()} VIDEOS")
    print(f"Videos selected: {len(videos)}")
    print("=" * 65)

    successful = 0
    failed = 0

    for index, video_path in enumerate(videos, start=1):

        print(
            f"\n[{index}/{len(videos)}] "
            f"{video_path.name}"
        )

        try:
            features = extract_video_features(
                video_path
            )

            if features is None:
                print("FAILED: No usable face features found.")
                failed += 1
                continue

            output_path = (
                OUTPUT_FOLDER
                / label_name
                / f"{video_path.stem}.npy"
            )

            np.save(
                output_path,
                features
            )

            successful += 1

            print("Feature shape:", features.shape)
            print("Saved:", output_path)
            print(
                f"Progress: {successful} successful, "
                f"{failed} failed"
            )

        except Exception as error:
            failed += 1
            print("ERROR processing video:", error)

    print("\n" + "-" * 65)
    print(f"{label_name.upper()} SUMMARY")
    print("Selected videos:", len(videos))
    print("Successfully processed:", successful)
    print("Failed:", failed)
    print("-" * 65)

    return successful, failed


# ============================================================
# MAIN
# ============================================================

def main():

    total_successful = 0
    total_failed = 0

    # Process up to 100 real videos
    real_success, real_failed = process_videos(
        REAL_FOLDER,
        "real"
    )

    total_successful += real_success
    total_failed += real_failed

    # Process up to 100 fake videos
    fake_success, fake_failed = process_videos(
        FAKE_FOLDER,
        "fake"
    )

    total_successful += fake_success
    total_failed += fake_failed

    # Final summary
    print("\n" + "=" * 65)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 65)

    print("Real videos successfully processed:", real_success)
    print("Fake videos successfully processed:", fake_success)

    print("Total successful videos:", total_successful)
    print("Total failed videos:", total_failed)

    print(
        "Total feature files created in this run:",
        total_successful
    )

    print("Expected maximum videos: 200")
    print("Output directory:", OUTPUT_FOLDER)

    print("=" * 65)


if __name__ == "__main__":
    main()