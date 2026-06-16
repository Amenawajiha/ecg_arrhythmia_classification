# ECG Arrhythmia Classification using Hybrid CNN-LSTM

A deep learning system that classifies cardiac arrhythmias from single-beat ECG signals into 5 clinical categories, trained on the MIT-BIH Arrhythmia Database and extended with multi-source PhysioNet data.

---

## Overview

Cardiovascular arrhythmias are a leading cause of sudden cardiac death. Manual review of long ECG recordings is time-consuming and error-prone. This project builds an automated classifier that processes a 187-point ECG signal and identifies the arrhythmia class in milliseconds, targeting the rare and clinically significant beat types that are most likely to be missed.

---

## Results

| Class | Samples Tested | Accuracy | Avg Confidence |
|---|---|---|---|
| Supraventricular | 20 | 95.0% | 94.0% |
| Ventricular | 20 | 95.0% | 97.2% |
| Fusion | 20 | 85.0% | 96.1% |

Benchmarked against published CNN-LSTM results on MIT-BIH (95–97% range). Fusion accuracy reflects inherent morphological ambiguity and class rarity (803 training samples vs 72,000 Normal).

---

## Architecture — Hybrid CNN-LSTM

```
Input (batch × 1 × 187)
  ↓
CNN Block 1: Conv1d(1→32, k=5) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
CNN Block 2: Conv1d(32→64, k=5) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
CNN Block 3: Conv1d(64→128, k=5) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
  ↓ (batch × 128 × 23)
Bidirectional LSTM (hidden=64, layers=2, dropout=0.3) → (batch × 128)
  ↓
FC(128→64) → ReLU → Dropout(0.5) → FC(64→5)
  ↓
Output logits (5 classes)
```

**CNN blocks** extract local morphological features (QRS shape, P wave presence, T wave direction) at progressively abstract levels. **Bidirectional LSTM** captures temporal dependencies in both directions across the compressed sequence. **Fully connected layers** perform the final classification.

Total parameters: ~180,000

---

## Dataset

**Primary:** MIT-BIH Arrhythmia Database (PhysioNet)
- 48 half-hour ambulatory ECG recordings, 360 Hz
- ~110,000 annotated beats, 5-class AAMI standard
- Preprocessed to 187-point R-peak-centered windows, normalised to [0, 1]

**Extended with:**
- MIT-BIH Supraventricular Arrhythmia Database (svdb) — 78 recordings, 128 Hz
- St. Petersburg INCART Database (incartdb) — 75 recordings, 257 Hz, 12-lead
- European ST-T Database (edb) — 36 recordings, 250 Hz

All external data resampled to 360 Hz and processed through the same segmentation pipeline for compatibility.

**Class distribution (MIT-BIH):**

| Class | Label | Count | % |
|---|---|---|---|
| Normal | 0 | ~72,471 | 75.0% |
| Supraventricular | 1 | ~2,223 | 2.3% |
| Ventricular | 2 | ~5,788 | 6.0% |
| Fusion | 3 | ~803 | 0.8% |
| Unclassifiable | 4 | ~6,431 | 6.7% |

---

## Data Augmentation

Applied exclusively to rare classes (1, 2, 3) to counter class imbalance:

- **Amplitude scaling** ±20% — simulates electrode contact variation
- **Baseline wander** (0.05–0.5 Hz sine) — simulates breathing artifact
- **Gaussian noise** (σ=0.015) — simulates electrode impedance variation
- **Fusion synthesis** — interpolated blending of Normal + Ventricular beats at α ∈ [0.3, 0.7], grounded in the physiology of fusion beat formation

---

## Project Structure

```
├── ecg_test_and_image_pipeline.py   # Stress testing + ECG image → signal extraction
├── ecg_data_ingestion.py            # PhysioNet ingestion, segmentation, augmentation
├── ecg_images/
│   ├── normal.png
│   ├── supraventricular_mitbih.png
│   ├── ventricular.png
│   └── fusion_mitbih.png
└── docs/
    ├── ECG_Viva_Preparation.docx
    ├── ECG_Project_Reference.docx
    └── ECG_Architectures_and_Papers.docx
```

---

## Key Functions

| Function | What It Does |
|---|---|
| `predict_ecg(model, signal, scaler, device)` | Inference on a single 187-point signal → class + probabilities |
| `stress_test_rare_classes(...)` | Evaluates model on rare classes with confidence breakdown |
| `ecg_image_to_signal(image_path)` | Extracts 187-point signal from a clean ECG waveform image using OpenCV |
| `predict_from_image(image_path, ...)` | End-to-end: image → signal → prediction |
| `build_combined_dataset(x_train, y_train)` | Full data expansion pipeline: fetch → segment → augment → combine |
| `extract_segments(signal, r_peaks, ...)` | Cuts R-peak-centered windows from raw PhysioNet recordings |
| `generate_fusion_samples(x, y, n)` | Synthesises Fusion beats by interpolating Normal + Ventricular |

---

## Alternative Architectures Explored

**1. ResNet-1D + Squeeze-and-Excitation Attention**
Skip connections eliminate vanishing gradients, enabling deeper networks. SE blocks learn per-channel attention weights, re-emphasising diagnostically relevant feature maps per input. Global Average Pooling replaces the LSTM final hidden state bottleneck. Published benchmark: 98.63% on MIT-BIH (Murat et al., 2023).

**2. Multi-Scale CNN + Transformer Encoder**
Three parallel CNN branches (kernel sizes 3, 11, 21) extract features simultaneously at the temporal scales of QRS edges, P/T waves, and overall beat morphology. Transformer self-attention directly models long-range relationships (e.g. P wave to QRS) without sequential propagation. Best suited for the Fusion class.

---

## Literature

| Paper | Contribution |
|---|---|
| Kachuee et al. (2018) arXiv:1805.00794 | Dataset source — established 187-pt MIT-BIH preprocessing |
| Murat et al. (2023) PLOS ONE | 1D ResNet benchmark: 98.63% on MIT-BIH |
| Jiang et al. (2021) Frontiers in Physiology | HADLN: CNN + BiGRU + Attention, F1=0.911 |
| Hu et al. (2022) Computers in Biology and Medicine | Transformer encoder for ECG |
| Ansari et al. (2023) Frontiers in Physiology | Survey of 368 DL-ECG papers 2017–2023 |
| arXiv:2602.17701 (2026) | CNN-LSTM + ResNet ensemble achieves F1=0.958 |

---

## Stack

- **Framework:** PyTorch
- **Signal processing:** SciPy, NumPy
- **Image extraction:** OpenCV
- **Data ingestion:** wfdb (PhysioNet Python library)
- **Preprocessing:** scikit-learn (StandardScaler)
- **Environment:** Google Colab / Python 3.10+

---

## Setup

```bash
pip install torch scipy numpy opencv-python wfdb scikit-learn matplotlib
```

```python
# Run inference on a signal from your test set
pred_class, probs = predict_ecg(model, x_test[0], scaler, device)

# Run inference from an ECG image
pred_class, probs, signal = predict_from_image("ecg.png", model, scaler, device, debug=True)

# Expand dataset from PhysioNet
x_combined, y_combined = build_combined_dataset(x_train, y_train,
                                                 augment_multiplier=3,
                                                 n_fusion_synthetic=500)
```
