import os
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. SETTINGS
# ============================================================

FEATURES_DIR = "C:/Users/Pawan/Desktop/Final Project/training_features"

FEATURE_SIZE = 2048
SEQUENCE_LENGTH = 60

HIDDEN_SIZE = 256
NUM_LAYERS = 2

ATTENTION_SIZE = 256

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.001


# ============================================================
# 2. DATASET
# ============================================================

class DeepfakeDataset(Dataset):

    def __init__(self, feature_files, labels):

        self.feature_files = feature_files
        self.labels = labels

    def __len__(self):

        return len(self.feature_files)

    def __getitem__(self, index):

        # Load ResNeXt features
        features = np.load(
            self.feature_files[index]
        ).astype(np.float32)

        original_length = features.shape[0]

        # ----------------------------------------------------
        # Make every video exactly 60 time steps
        # ----------------------------------------------------

        if original_length < SEQUENCE_LENGTH:

            missing = SEQUENCE_LENGTH - original_length

            # Repeat the last available feature
            last_feature = features[-1:]

            padding = np.repeat(
                last_feature,
                missing,
                axis=0
            )

            features = np.concatenate(
                [features, padding],
                axis=0
            )

        elif original_length > SEQUENCE_LENGTH:

            features = features[
                :SEQUENCE_LENGTH
            ]

        # Safety check
        if features.shape != (
            SEQUENCE_LENGTH,
            FEATURE_SIZE
        ):

            raise ValueError(
                f"Invalid feature shape "
                f"{features.shape} in "
                f"{self.feature_files[index]}"
            )

        # NumPy → PyTorch
        features = torch.tensor(
            features,
            dtype=torch.float32
        )

        # Real = 0
        # Fake = 1
        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return features, label


# ============================================================
# 3. COLLECT FEATURES
# ============================================================

fake_dir = os.path.join(
    FEATURES_DIR,
    "fake"
)

real_dir = os.path.join(
    FEATURES_DIR,
    "real"
)

feature_files = []
labels = []


# ------------------------------------------------------------
# Fake videos → label 1
# ------------------------------------------------------------

for filename in os.listdir(fake_dir):

    if filename.lower().endswith(".npy"):

        filepath = os.path.join(
            fake_dir,
            filename
        )

        feature_files.append(filepath)
        labels.append(1)


# ------------------------------------------------------------
# Real videos → label 0
# ------------------------------------------------------------

for filename in os.listdir(real_dir):

    if filename.lower().endswith(".npy"):

        filepath = os.path.join(
            real_dir,
            filename
        )

        feature_files.append(filepath)
        labels.append(0)


print("================================")
print("DATASET")
print("================================")

print("Total videos:", len(feature_files))
print("Real videos:", labels.count(0))
print("Fake videos:", labels.count(1))


# ============================================================
# 4. CHECK DATASET
# ============================================================

if len(feature_files) == 0:

    raise RuntimeError(
        "No .npy feature files found."
    )

if labels.count(0) == 0:

    raise RuntimeError(
        "No real feature files found."
    )

if labels.count(1) == 0:

    raise RuntimeError(
        "No fake feature files found."
    )


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

train_files, test_files, train_labels, test_labels = train_test_split(

    feature_files,
    labels,

    test_size=0.20,

    random_state=42,

    stratify=labels
)


print("\n================================")
print("DATA SPLIT")
print("================================")

print("Training videos:", len(train_files))
print("Testing videos:", len(test_files))


# ============================================================
# 6. DATA LOADERS
# ============================================================

train_dataset = DeepfakeDataset(
    train_files,
    train_labels
)

test_dataset = DeepfakeDataset(
    test_files,
    test_labels
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 7. TEMPORAL ATTENTION
# ============================================================

class TemporalAttention(nn.Module):

    def __init__(
        self,
        input_dim=512,
        attention_dim=256
    ):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(
                input_dim,
                attention_dim
            ),

            nn.Tanh(),

            nn.Linear(
                attention_dim,
                1
            )
        )


    def forward(self, x):

        # x:
        # (batch, 60, 512)

        # Calculate attention scores
        scores = self.attention(x)

        # scores:
        # (batch, 60, 1)

        # Normalize across time
        weights = torch.softmax(
            scores,
            dim=1
        )

        # Weighted sum of all time steps
        context = torch.sum(
            weights * x,
            dim=1
        )

        # context:
        # (batch, 512)

        return context, weights


# ============================================================
# 8. BI-LSTM + ATTENTION MODEL
# ============================================================

class BiLSTMAttentionModel(nn.Module):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # Bi-LSTM
        # ----------------------------------------------------

        self.bilstm = nn.LSTM(

            input_size=FEATURE_SIZE,

            hidden_size=HIDDEN_SIZE,

            num_layers=NUM_LAYERS,

            batch_first=True,

            bidirectional=True,

            dropout=0.3
        )


        # ----------------------------------------------------
        # Temporal Attention
        # ----------------------------------------------------

        self.attention = TemporalAttention(

            input_dim=HIDDEN_SIZE * 2,

            attention_dim=ATTENTION_SIZE
        )


        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            0.4
        )


        # ----------------------------------------------------
        # Binary classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(

            HIDDEN_SIZE * 2,

            1
        )


    def forward(self, x):

        # Input:
        # (batch, 60, 2048)

        lstm_output, _ = self.bilstm(x)

        # Output:
        # (batch, 60, 512)

        context, attention_weights = self.attention(
            lstm_output
        )

        # Context:
        # (batch, 512)

        context = self.dropout(
            context
        )

        # Classifier
        output = self.classifier(
            context
        )

        return output.squeeze(1), attention_weights


# ============================================================
# 9. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)


# ============================================================
# 10. CREATE MODEL
# ============================================================

model = BiLSTMAttentionModel().to(device)


print("\n================================")
print("BI-LSTM + TEMPORAL ATTENTION")
print("================================")

print(model)


# ============================================================
# 11. LOSS FUNCTION
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# 12. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 13. TRAINING
# ============================================================

print("\n================================")
print("STARTING TRAINING")
print("================================")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0


    for features, labels_batch in train_loader:

        features = features.to(device)

        labels_batch = labels_batch.to(device)


        # Clear gradients
        optimizer.zero_grad()


        # Forward pass
        outputs, attention_weights = model(
            features
        )


        # Loss
        loss = criterion(
            outputs,
            labels_batch
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        total_loss += loss.item()


    average_loss = (
        total_loss /
        len(train_loader)
    )


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )


# ============================================================
# 14. EVALUATION
# ============================================================

print("\n================================")
print("EVALUATING MODEL")
print("================================")


model.eval()

all_predictions = []
all_labels = []


with torch.no_grad():

    for features, labels_batch in test_loader:

        features = features.to(device)


        # Forward pass
        outputs, attention_weights = model(
            features
        )


        # Convert logits → probability
        probabilities = torch.sigmoid(
            outputs
        )


        # Probability >= 0.5 → Fake
        predictions = (
            probabilities >= 0.5
        ).int()


        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels_batch.numpy()
        )


# ============================================================
# 15. METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


# ============================================================
# 16. FINAL RESULTS
# ============================================================

print("\n================================")
print("FINAL RESULTS")
print("================================")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

MODEL_PATH = (
    "C:/Users/Pawan/Desktop/"
    "Final Project/bilstm_attention_model.pth"
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)


print("\n================================")
print("MODEL SAVED")
print("================================")

print(
    "Saved to:",
    MODEL_PATH
)

print(
    "\nBi-LSTM + Temporal Attention "
    "training completed successfully!"
)