from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
from mtcnn import MTCNN

# ---------------------------------------------------------
# Paths and model configuration (matches temporal_attention.py)
# ---------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "bilstm_attention_model.pth"
FEATURE_SIZE = 2048
SEQUENCE_LENGTH = 60
HIDDEN_SIZE = 256
NUM_LAYERS = 2
ATTENTION_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalAttention(nn.Module):
    def __init__(self, input_dim=512, attention_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x):
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights


class BiLSTMAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=FEATURE_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attention = TemporalAttention(
            input_dim=HIDDEN_SIZE * 2,
            attention_dim=ATTENTION_SIZE,
        )
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(HIDDEN_SIZE * 2, 1)

    def forward(self, x):
        lstm_output, _ = self.bilstm(x)
        context, attention_weights = self.attention(lstm_output)
        context = self.dropout(context)
        logits = self.classifier(context)
        return logits.squeeze(1), attention_weights


@st.cache_resource(show_spinner="Loading detection models…")
def load_models():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Place bilstm_attention_model.pth in the same folder as app.py."
        )

    weights = ResNeXt101_64X4D_Weights.DEFAULT
    feature_model = resnext101_64x4d(weights=weights)
    feature_model.fc = nn.Identity()
    feature_model = feature_model.to(DEVICE).eval()
    preprocess = weights.transforms()

    detector = MTCNN()
    sequence_model = BiLSTMAttentionModel().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    sequence_model.load_state_dict(checkpoint, strict=True)
    sequence_model.eval()
    return detector, feature_model, preprocess, sequence_model


def extract_video_features(video_path, detector, feature_model, preprocess, progress):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open this video. Try another MP4 file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("The video contains no readable frames.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps):
        fps = 0.0

    frame_indices = np.linspace(
        0, total_frames - 1, min(SEQUENCE_LENGTH, total_frames), dtype=int
    )

    features, previews, timestamps, face_boxes = [], [], [], []
    for idx, frame_no in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            faces = detector.detect_faces(frame_rgb)
        except Exception:
            faces = []

        if faces:
            face = max(
                faces,
                key=lambda f: max(0, f["box"][2]) * max(0, f["box"][3])
            )
            x, y, w, h = map(int, face["box"])
            height, width = frame_rgb.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size:
                face_image = Image.fromarray(crop)
                tensor = preprocess(face_image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    vec = feature_model(tensor).squeeze(0).cpu().numpy()
                features.append(vec.astype(np.float32))
                previews.append(frame_rgb)
                timestamps.append(float(frame_no / fps) if fps > 0 else float(frame_no))
                face_boxes.append((x1, y1, x2, y2))

        progress.progress((idx + 1) / len(frame_indices), text=f"Analysing sampled frame {idx + 1}/{len(frame_indices)}")
    cap.release()

    if not features:
        raise RuntimeError(
            "No usable faces were detected in the sampled frames. "
            "Try a clearer video with a visible face."
        )

    # Match the training script: repeat final feature for short sequences,
    # truncate longer sequences to 60.
    arr = np.asarray(features, dtype=np.float32)
    if len(arr) < SEQUENCE_LENGTH:
        pad = np.repeat(arr[-1:], SEQUENCE_LENGTH - len(arr), axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif len(arr) > SEQUENCE_LENGTH:
        arr = arr[:SEQUENCE_LENGTH]

    # The model always receives exactly 60 steps. Keep explanation metadata
    # only for frames that were actually extracted.
    return arr, previews[:SEQUENCE_LENGTH], timestamps[:SEQUENCE_LENGTH], face_boxes[:SEQUENCE_LENGTH], total_frames, fps


def main():
    st.set_page_config(page_title="DeepGuard AI | Deepfake Detection", page_icon="🛡️", layout="wide")
    st.title("🛡️ DeepGuard AI")
    st.caption("Deepfake video detection with temporal-attention explanations")
    st.info(
        "Upload an MP4 to analyse it with your ResNeXt101 feature extractor and "
        "BiLSTM + temporal-attention checkpoint. The explanation highlights sampled "
        "frames emphasized by the model; it is evidence about model behaviour, not proof."
    )

    with st.sidebar:
        st.subheader("Model configuration")
        st.write("Checkpoint: `bilstm_attention_model.pth`")
        st.write(f"Device: `{DEVICE}`")
        st.write("Sampled time steps: `60`")
        st.caption("First run may download pretrained ResNeXt101 weights if they are not cached.")

    uploaded = st.file_uploader("Choose an MP4 video", type=["mp4"])
    if uploaded is None:
        st.markdown("### What you'll get")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", "REAL / FAKE")
        c2.metric("Temporal explanation", "Frame importance")
        c3.metric("Evidence", "Key-frame previews")
        st.caption("Predictions are generated only after a video is uploaded and processed.")
        return

    st.video(uploaded)
    if st.button("Analyse video", type="primary", use_container_width=True):
        progress = st.progress(0, text="Preparing video…")
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
                temp.write(uploaded.getbuffer())
                temp_path = temp.name

            detector, feature_model, preprocess, sequence_model = load_models()
            feats, previews, timestamps, boxes, total_frames, fps = extract_video_features(
                temp_path, detector, feature_model, preprocess, progress
            )

            x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logit, attention = sequence_model(x)
                fake_probability = torch.sigmoid(logit).item()
                attention = attention.squeeze(0).squeeze(-1).cpu().numpy()

            label = "FAKE" if fake_probability >= 0.5 else "REAL"
            displayed_probability = fake_probability if label == "FAKE" else 1.0 - fake_probability

            st.divider()
            left, right = st.columns([1, 1])
            with left:
                st.subheader("Detection result")
                if label == "FAKE":
                    st.error("🚨 FAKE VIDEO")
                else:
                    st.success("✅ REAL VIDEO")
                st.metric("Model score for predicted class", f"{displayed_probability * 100:.1f}%")
                st.progress(float(np.clip(displayed_probability, 0, 1)))
                st.caption(
                    "This is the model's predicted score, not a guarantee of correctness "
                    "or a calibrated probability."
                )
            with right:
                st.subheader("Video details")
                st.write(f"**Sampled frames with detected faces:** {len(previews)}")
                st.write(f"**Total video frames:** {total_frames}")
                st.write(f"**FPS:** {fps:.2f}" if fps > 0 else "**FPS:** unavailable")
                st.write(f"**Processing device:** {DEVICE}")

            st.divider()
            st.subheader("Explainable AI: which sampled frames received the most attention?")
            valid_n = min(len(previews), len(attention), len(timestamps))
            weights = attention[:valid_n]
            times = timestamps[:valid_n]

            if valid_n:
                fig, ax = plt.subplots(figsize=(11, 3.5))
                ax.bar(np.arange(valid_n), weights)
                ax.set_xlabel("Sampled face frame (chronological order)")
                ax.set_ylabel("Temporal attention weight")
                ax.set_title("Temporal attention across extracted face frames")
                ax.grid(axis="y", alpha=0.25)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

                top_n = min(6, valid_n)
                top_indices = np.argsort(weights)[::-1][:top_n]
                st.markdown("#### Highest-attention frames")
                cols = st.columns(min(3, top_n))
                for rank, frame_idx in enumerate(top_indices):
                    preview = previews[int(frame_idx)].copy()
                    x1, y1, x2, y2 = boxes[int(frame_idx)]
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 255), 2)
                    col = cols[rank % len(cols)]
                    with col:
                        st.image(preview, caption=f"t={times[frame_idx]:.2f}s · attention {weights[frame_idx]:.4f}", use_container_width=True)

                st.markdown("#### Interpretation")
                st.write(
                    f"The model assigned the largest temporal attention weight to the sampled frame "
                    f"at approximately **{times[int(top_indices[0])]:.2f} seconds**. "
                    "These weights show which time steps the attention layer emphasized when forming "
                    "its sequence representation. They do not, by themselves, establish that a specific "
                    "visual artifact is fake or prove causal reasoning."
                )
                st.caption(
                    "Frames without a detected face are skipped during feature extraction. "
                    "The sequence is padded by repeating the last extracted feature when fewer than 60 "
                    "face frames are available, matching the training preprocessing."
                )

                report = (
                    "DeepGuard AI - Video Analysis Report\n"
                    f"File: {uploaded.name}\n"
                    f"Prediction: {label}\n"
                    f"Predicted-class score: {displayed_probability * 100:.1f}%\n"
                    f"Fake score: {fake_probability * 100:.1f}%\n"
                    f"Extracted face frames: {valid_n}\n"
                    "Explanation: temporal attention weights identify frames emphasized by the model; "
                    "they are not proof of manipulation.\n\n"
                    "Highest-attention sampled frames:\n"
                    + "\n".join(
                        f"- {times[int(i)]:.2f}s: weight {weights[int(i)]:.6f}"
                        for i in top_indices
                    )
                )
                st.download_button(
                    "Download analysis report (.txt)",
                    data=report,
                    file_name="deepguard_analysis_report.txt",
                    mime="text/plain",
                )
            else:
                st.warning("The model returned a prediction, but no frame-level explanation could be displayed.")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.markdown(
                "Check that the checkpoint matches the architecture in this app, that all dependencies "
                "are installed, and that the video contains readable frames with detectable faces."
            )
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
