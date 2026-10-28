import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# 1. SETTINGS
# ============================================================

# Assumes this script is saved in: Final Project/src/
PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_DIR / "training_features"
MODEL_PATH = PROJECT_DIR / "bilstm_model.pth"

FEATURE_SIZE = 2048
SEQUENCE_LENGTH = 60

HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.001
TEST_SIZE = 0.20
RANDOM_SEED = 42


# ============================================================
# 2. REPRODUCIBILITY AND DEVICE
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Project folder:", PROJECT_DIR)
print("Features folder:", FEATURES_DIR)
print("Device:", device)


# ============================================================
# 3. DATASET CLASS
# Each .npy file must represent ONE video and have shape
# (number_of_frames, 2048).
# ============================================================

class DeepfakeDataset(Dataset):
    def __init__(self, feature_files, labels):
        self.feature_files = feature_files
        self.labels = labels

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, index):
        feature_path = self.feature_files[index]
        features = np.load(feature_path).astype(np.float32)

        if features.ndim != 2 or features.shape[1] != FEATURE_SIZE:
            raise ValueError(
                f"Expected a 2D feature array with {FEATURE_SIZE} columns, "
                f"but got {features.shape} in {feature_path}"
            )

        if features.shape[0] == 0:
            raise ValueError(f"No frame features found in {feature_path}")

        original_length = features.shape[0]

        # Make every sequence exactly 60 time steps long.
        if original_length < SEQUENCE_LENGTH:
            missing = SEQUENCE_LENGTH - original_length
            padding = np.repeat(features[-1:], missing, axis=0)
            features = np.concatenate([features, padding], axis=0)
        elif original_length > SEQUENCE_LENGTH:
            features = features[:SEQUENCE_LENGTH]

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[index], dtype=torch.float32)

        return features_tensor, label_tensor


# ============================================================
# 4. COLLECT LABELED FEATURE FILES
#
# Expected structure:
# Final Project/
#   training_features/
#     real/  -> real-video .npy files
#     fake/  -> manipulated-video .npy files
# ============================================================

real_dir = FEATURES_DIR / "real"
fake_dir = FEATURES_DIR / "fake"

for folder in (real_dir, fake_dir):
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Required folder not found: {folder}\n"
            "Create training_features/real and training_features/fake, "
            "then place the corresponding .npy feature files inside them."
        )

real_files = sorted(real_dir.glob("*.npy"))
fake_files = sorted(fake_dir.glob("*.npy"))

feature_files = [str(path) for path in real_files + fake_files]
labels = [0] * len(real_files) + [1] * len(fake_files)

print("\n" + "=" * 40)
print("DATASET")
print("=" * 40)
print("Total videos:", len(feature_files))
print("Real videos:", len(real_files))
print("Fake videos:", len(fake_files))

if not real_files or not fake_files:
    raise RuntimeError(
        "Both classes must contain .npy files. Real label = 0; fake label = 1."
    )

if min(len(real_files), len(fake_files)) < 2:
    raise RuntimeError(
        "A stratified train/test split requires at least 2 files per class."
    )

# Validate feature files before training.
for path in feature_files:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2 or array.shape[1] != FEATURE_SIZE or array.shape[0] == 0:
        raise ValueError(
            f"{path} has shape {array.shape}; expected (frames, {FEATURE_SIZE}) "
            "with at least one frame."
        )

train_files, test_files, train_labels, test_labels = train_test_split(
    feature_files,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels,
)

print("\n" + "=" * 40)
print("DATA SPLIT")
print("=" * 40)
print("Training videos:", len(train_files))
print("Testing videos:", len(test_files))

train_dataset = DeepfakeDataset(train_files, train_labels)
test_dataset = DeepfakeDataset(test_files, test_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# 5. BI-LSTM MODEL (NO ATTENTION LAYER)
#
# Input: (batch, 60, 2048)
# Bi-LSTM output: (batch, 60, 512)
# The final time-step output is passed to the classifier.
# ============================================================

class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=FEATURE_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )

        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(HIDDEN_SIZE * 2, 1)

    def forward(self, x):
        # x: (batch, sequence_length, feature_size)
        output, _ = self.bilstm(x)
        # output: (batch, 60, 512)

        final_output = output[:, -1, :]
        # final_output: (batch, 512)

        logits = self.classifier(self.dropout(final_output))
        return logits.squeeze(1)


# ============================================================
# 6. CREATE MODEL, LOSS, AND OPTIMIZER
# ============================================================

model = BiLSTMModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n" + "=" * 40)
print("BI-LSTM MODEL (NO ATTENTION)")
print("=" * 40)
print(model)


# ============================================================
# 7. TRAINING
# ============================================================

print("\n" + "=" * 40)
print("STARTING TRAINING")
print("=" * 40)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for features, labels_batch in train_loader:
        features = features.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels_batch)
        loss.backward()

        # Clip gradients to help avoid unusually large LSTM gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {average_loss:.4f}")


# ============================================================
# 8. EVALUATION
# ============================================================

print("\n" + "=" * 40)
print("EVALUATING MODEL")
print("=" * 40)

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for features, labels_batch in test_loader:
        features = features.to(device)
        logits = model(features)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).int()

        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_labels.extend(labels_batch.int().numpy().tolist())

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, zero_division=0)
recall = recall_score(all_labels, all_predictions, zero_division=0)
f1 = f1_score(all_labels, all_predictions, zero_division=0)

print("\n" + "=" * 40)
print("FINAL RESULTS")
print("=" * 40)
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")


# ============================================================
# 9. SAVE MODEL
# ============================================================

torch.save(model.state_dict(), MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Bi-LSTM training completed.")
