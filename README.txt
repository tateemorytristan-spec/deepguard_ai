# DeepGuard AI GUI

## Repository contents and local assets

This repository contains the application, preprocessing and training source code,
and dependency lists. Datasets, extracted frames, face crops, NumPy features,
PyTorch checkpoints, generated reports, and temporary files are kept locally and
excluded from Git. A fresh clone does not include these assets.

To run inference, place your trained `bilstm_attention_model.pth` in the project
root. The app requires this checkpoint; installing dependencies alone is not
sufficient. To train a model, supply the labelled videos and prepare the feature
directories expected by the scripts in `src/`. Several preprocessing scripts use
local Windows paths, which must be adjusted for another computer.

The original data-containing commit is retained on a local backup branch. Push
only the code branch; do not use `git push --all` or `git push --mirror`, which
would also attempt to upload the backup history.

This Streamlit app runs your `bilstm_attention_model.pth` checkpoint on an uploaded MP4.

## Put these files together

Keep `app.py` and `bilstm_attention_model.pth` in your project root:

```text
Final Project/
├── app.py
├── bilstm_attention_model.pth
├── requirements-gui.txt
├── src/
├── training_features/
└── ...
```

## Install and run (Windows / VS Code)

Use the same Python environment in which your model dependencies work.

```powershell
python -m pip install -r requirements-gui.txt
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Notes

- The model architecture in `app.py` matches `TemporalAttention` and `BiLSTMAttentionModel` in the provided `temporal_attention.py`: 2048 features, 60 time steps, 2-layer bidirectional LSTM, 256 hidden units per direction, 256 attention dimension, and a single logit.
- The ResNeXt101 feature extractor and face-crop preprocessing are copied to match the provided `extract_features.py`.
- The first run may download torchvision's pretrained ResNeXt101 weights if they are not already cached.
- The app displays temporal attention weights and the original sampled frames with face boxes. This is an attention-based explanation, not a pixel-level Grad-CAM heatmap and not proof that a video is manipulated.
- The model's score should not be interpreted as a calibrated probability.
- If `torch.load(..., weights_only=True)` is unsupported in your installed PyTorch, update PyTorch or change that argument to `weights_only=False` only for a checkpoint file you trust.
- The code assumes your checkpoint was saved from the `BiLSTMAttentionModel` architecture in the provided `temporal_attention.py`. It does not load the standalone `Classifier` in `classifier.py`.
