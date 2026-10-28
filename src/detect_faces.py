import cv2
import os
from mtcnn import MTCNN

# Folder containing the 60 extracted frames
frames_folder = "C:/Users/Pawan/Desktop/Final Project/Frames"

# Folder where cropped faces will be saved
output_folder = "C:/Users/Pawan/Desktop/Final Project/Face_crops"

os.makedirs(output_folder, exist_ok=True)

# Create MTCNN face detector
detector = MTCNN()

# Process all frame images
for filename in sorted(os.listdir(frames_folder)):

    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(frames_folder, filename)

    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print("Could not read:", filename)
        continue

    # MTCNN expects RGB images
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = detector.detect_faces(rgb_image)

    if len(results) == 0:
        print("No face found:", filename)
        continue

    # Take the first detected face
    x, y, width, height = results[0]["box"]

    # Make sure coordinates are valid
    x = max(0, x)
    y = max(0, y)

    # Crop the face
    face = image[y:y + height, x:x + width]

    # Resize to 224 × 224
    face = cv2.resize(face, (224, 224))

    # Save cropped face
    output_path = os.path.join(output_folder, filename)

    cv2.imwrite(output_path, face)

    print("Face saved:", filename)

print("Face extraction completed!")