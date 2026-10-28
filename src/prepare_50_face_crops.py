import os
import cv2
from PIL import Image
from mtcnn import MTCNN


# ============================================================
# PATHS
# ============================================================

PROJECT = r"C:\Users\Pawan\Desktop\Final Project"

INPUT_FOLDER = os.path.join(
    PROJECT,
    "training_frames"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT,
    "training_face_crops"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_VIDEOS = 25


# ============================================================
# MTCNN
# ============================================================

detector = MTCNN()


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(input_video_folder, output_video_folder):

    os.makedirs(
        output_video_folder,
        exist_ok=True
    )

    frames = sorted([
        f for f in os.listdir(input_video_folder)
        if f.lower().endswith(".jpg")
    ])

    detected_count = 0

    for frame_file in frames:

        frame_path = os.path.join(
            input_video_folder,
            frame_file
        )

        image = cv2.imread(frame_path)

        if image is None:
            continue

        # BGR → RGB
        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Detect faces
        faces = detector.detect_faces(
            image_rgb
        )

        if len(faces) == 0:
            continue

        # Select largest face
        face = max(
            faces,
            key=lambda x: x["box"][2] * x["box"][3]
        )

        x, y, w, h = face["box"]

        # Handle negative coordinates
        x = max(0, x)
        y = max(0, y)

        # Make sure coordinates stay inside image
        h_img, w_img = image_rgb.shape[:2]

        x2 = min(x + w, w_img)
        y2 = min(y + h, h_img)

        face_crop = image_rgb[
            y:y2,
            x:x2
        ]

        if face_crop.size == 0:
            continue

        # Save face crop
        output_path = os.path.join(
            output_video_folder,
            frame_file
        )

        face_image = Image.fromarray(
            face_crop
        )

        face_image.save(
            output_path
        )

        detected_count += 1

    return detected_count, len(frames)


# ============================================================
# PROCESS REAL / FAKE
# ============================================================

def process_class(class_name):

    input_class_folder = os.path.join(
        INPUT_FOLDER,
        class_name
    )

    output_class_folder = os.path.join(
        OUTPUT_FOLDER,
        class_name
    )

    os.makedirs(
        output_class_folder,
        exist_ok=True
    )

    video_folders = sorted([
        f for f in os.listdir(input_class_folder)
        if os.path.isdir(
            os.path.join(
                input_class_folder,
                f
            )
        )
    ])

    video_folders = video_folders[:MAX_VIDEOS]

    print()
    print("=" * 60)
    print(f"{class_name.upper()} FACE CROPPING")
    print(f"Videos: {len(video_folders)}")
    print("=" * 60)

    for index, video_name in enumerate(
        video_folders,
        start=1
    ):

        input_video_folder = os.path.join(
            input_class_folder,
            video_name
        )

        output_video_folder = os.path.join(
            output_class_folder,
            video_name
        )

        print(
            f"[{index}/{len(video_folders)}] "
            f"{video_name}"
        )

        detected, total = process_video(
            input_video_folder,
            output_video_folder
        )

        print(
            f"    Faces detected: "
            f"{detected}/{total}"
        )


# ============================================================
# RUN
# ============================================================

process_class("real")
process_class("fake")


print()
print("=" * 60)
print("50-VIDEO FACE CROPPING COMPLETED!")
print("=" * 60)