import os
import cv2
import torch
import torchvision.models as models
from torchvision.models import ResNeXt101_64X4D_Weights
from mtcnn import MTCNN
import numpy as np
from PIL import Image

# --------------------------------------------------
# PATHS
# --------------------------------------------------

project_folder = r"C:\Users\Pawan\Desktop\Final Project"

real_folder = os.path.join(
    project_folder,
    "dataset_split",
    "train",
    "real"
)

fake_folder = os.path.join(
    project_folder,
    "dataset_split",
    "train",
    "fake"
)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

NUM_VIDEOS = 5
NUM_FRAMES = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# --------------------------------------------------
# MTCNN
# --------------------------------------------------

detector = MTCNN()

# --------------------------------------------------
# RESNEXT-101
# --------------------------------------------------

weights = ResNeXt101_64X4D_Weights.DEFAULT

model = models.resnext101_64x4d(weights=weights)

# Remove ImageNet classifier
model.fc = torch.nn.Identity()

model = model.to(device)
model.eval()

# ImageNet preprocessing
transform = weights.transforms()

# --------------------------------------------------
# FEATURE EXTRACTION FUNCTION
# --------------------------------------------------

def extract_features(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print("Could not read:", video_path)
        cap.release()
        return None

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES,
        dtype=int
    )

    features = []

    for frame_number in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = cap.read()

        if not success:
            continue

        # OpenCV BGR → RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MTCNN face detection
        faces = detector.detect_faces(frame_rgb)

        if len(faces) == 0:
            continue

        # Select largest detected face
        face = max(
            faces,
            key=lambda x: x["box"][2] * x["box"][3]
        )

        x, y, w, h = face["box"]

        # Prevent negative coordinates
        x = max(0, x)
        y = max(0, y)

        face_crop = frame_rgb[
            y:y+h,
            x:x+w
        ]

        if face_crop.size == 0:
            continue

        # Convert to PIL
        face_image = Image.fromarray(face_crop)

        # ResNeXt preprocessing
        face_tensor = transform(face_image)

        face_tensor = face_tensor.unsqueeze(0).to(device)

        # Extract 2048 features
        with torch.no_grad():

            feature = model(face_tensor)

        feature = feature.cpu().numpy().squeeze()

        features.append(feature)

    cap.release()

    if len(features) == 0:
        return None

    # Convert to array
    features = np.array(features)

    return features


# --------------------------------------------------
# TEST REAL VIDEOS
# --------------------------------------------------

real_videos = [
    f for f in os.listdir(real_folder)
    if f.lower().endswith(".mp4")
][:NUM_VIDEOS]

fake_videos = [
    f for f in os.listdir(fake_folder)
    if f.lower().endswith(".mp4")
][:NUM_VIDEOS]


print("\nTesting REAL videos...")

for video in real_videos:

    path = os.path.join(real_folder, video)

    print("\nReal:", video)

    features = extract_features(path)

    if features is not None:
        print("Feature shape:", features.shape)
        print("Label: 0 (Real)")
    else:
        print("Feature extraction failed.")


print("\nTesting FAKE videos...")

for video in fake_videos:

    path = os.path.join(fake_folder, video)

    print("\nFake:", video)

    features = extract_features(path)

    if features is not None:
        print("Feature shape:", features.shape)
        print("Label: 1 (Fake)")
    else:
        print("Feature extraction failed.")


print("\nPilot test completed!")