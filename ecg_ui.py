"""
ECG Arrhythmia Classifier — Streamlit Demo UI
Run with: streamlit run ecg_ui.py
"""

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.signal import resample
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import io
import time

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="ECG Arrhythmia Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Inject CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Root & Reset ── */
:root {
    --navy:       #050D1A;
    --navy-card:  #0A1628;
    --navy-border:#0F2040;
    --navy-hover: #112248;
    --teal:       #00D4AA;
    --teal-dim:   #00A882;
    --teal-glow:  rgba(0, 212, 170, 0.15);
    --cyan:       #38BDF8;
    --white:      #F0F6FF;
    --muted:      #5A7A99;
    --danger:     #FF4D6D;
    --warn:       #FFB800;
    --success:    #00D4AA;
    --font-mono:  'IBM Plex Mono', monospace;
    --font-sans:  'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-sans);
    background-color: var(--navy) !important;
    color: var(--white);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1400px; }
[data-testid="stSidebar"] { display: none; }

/* ── Header ── */
.ecg-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--navy-border);
    padding-bottom: 1.5rem;
    margin-bottom: 2.5rem;
}
.ecg-title {
    font-family: var(--font-mono);
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--teal);
}
.ecg-subtitle {
    font-size: 0.78rem;
    color: var(--muted);
    font-family: var(--font-mono);
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}
.ecg-badge {
    background: var(--teal-glow);
    border: 1px solid var(--teal-dim);
    color: var(--teal);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    padding: 0.3rem 0.8rem;
    border-radius: 2px;
}

/* ── Cards ── */
.ecg-card {
    background: var(--navy-card);
    border: 1px solid var(--navy-border);
    border-radius: 4px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.card-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--navy-border);
    padding-bottom: 0.6rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--navy-card) !important;
    border: 1px dashed var(--navy-border) !important;
    border-radius: 4px !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--teal-dim) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--teal) !important;
    color: var(--navy) !important;
    border: none !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    border-radius: 2px !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: var(--teal-dim) !important;
}
.stButton > button:disabled {
    background: var(--navy-border) !important;
    color: var(--muted) !important;
}

/* ── Result: prediction class ── */
.result-class {
    font-family: var(--font-mono);
    font-size: 2.4rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    line-height: 1.1;
    margin: 0.5rem 0 0.3rem 0;
}
.result-conf {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--muted);
    letter-spacing: 0.1em;
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.55rem 0;
}
.prob-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    color: var(--muted);
    width: 140px;
    flex-shrink: 0;
}
.prob-label.active { color: var(--white); font-weight: 600; }
.prob-track {
    flex: 1;
    height: 5px;
    background: var(--navy-border);
    border-radius: 2px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--teal-dim);
    transition: width 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}
.prob-fill.active { background: var(--teal); }
.prob-pct {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
    width: 46px;
    text-align: right;
    flex-shrink: 0;
}
.prob-pct.active { color: var(--teal); font-weight: 600; }

/* ── Stat chips ── */
.stat-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}
.stat-chip {
    background: var(--navy);
    border: 1px solid var(--navy-border);
    border-radius: 3px;
    padding: 0.5rem 0.9rem;
    font-family: var(--font-mono);
}
.stat-chip .chip-val {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--cyan);
}
.stat-chip .chip-lbl {
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.15rem;
}

/* ── Alert banners ── */
.alert-warn {
    background: rgba(255, 184, 0, 0.08);
    border: 1px solid rgba(255, 184, 0, 0.3);
    border-left: 3px solid var(--warn);
    border-radius: 3px;
    padding: 0.7rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--warn);
    letter-spacing: 0.04em;
    margin: 0.8rem 0;
}
.alert-info {
    background: rgba(0, 212, 170, 0.06);
    border: 1px solid rgba(0, 212, 170, 0.2);
    border-left: 3px solid var(--teal);
    border-radius: 3px;
    padding: 0.7rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--teal);
    letter-spacing: 0.04em;
    margin: 0.8rem 0;
}

/* ── Progress / spinner ── */
.stSpinner > div { border-top-color: var(--teal) !important; }

/* ── Matplotlib figures ── */
[data-testid="stImage"] img {
    border-radius: 3px;
    border: 1px solid var(--navy-border);
}

/* ── Info table ── */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.info-item {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
}
.info-item span { color: var(--white); font-weight: 500; }

/* ── Divider ── */
hr.ecg-divider {
    border: none;
    border-top: 1px solid var(--navy-border);
    margin: 1.5rem 0;
}

/* ── Section separator ── */
.section-sep {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--navy-border);
    margin: 2rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unclassifiable"
}

CLASS_CLINICAL = {
    0: "Normal sinus beat — SA node origin, normal conduction",
    1: "Above-ventricular origin — abnormal P wave, narrow QRS",
    2: "Ventricular origin — wide bizarre QRS, no preceding P wave",
    3: "Collision of sinus and ventricular wavefronts — intermediate morphology",
    4: "Cannot be reliably categorised — manual review required"
}

CLASS_RISK = {
    0: ("LOW", "#00D4AA"),
    1: ("MODERATE", "#FFB800"),
    2: ("HIGH", "#FF4D6D"),
    3: ("MODERATE", "#FFB800"),
    4: ("UNKNOWN", "#5A7A99")
}

CONFIDENCE_THRESHOLD = 0.85   # below this → show low-confidence warning


# ── Model loading (cached) ────────────────────────────────────
@st.cache_resource
def load_model_checkpoint():
    """Load model, scaler, and device from checkpoint file."""
    import os
    
    checkpoint_path = 'ecg_model_checkpoint.pth'
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {checkpoint_path}\n"
            f"Current directory: {os.getcwd()}\n"
            f"Files present: {os.listdir('.')}"
        )
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Recreate model from architecture specs
    model_arch = checkpoint['model_architecture']
    
    # Import CNN_LSTM class (define it if not already)
    import torch.nn as nn
    import torch.nn.functional as F
    
    class CNN_LSTM(nn.Module):
        def __init__(self, n_classes=5, input_channels=1, input_length=187):
            super(CNN_LSTM, self).__init__()
            self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(32)
            self.pool1 = nn.MaxPool1d(2)
            self.dropout1 = nn.Dropout(0.2)
            
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.MaxPool1d(2)
            self.dropout2 = nn.Dropout(0.2)
            
            self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool3 = nn.MaxPool1d(2)
            self.dropout3 = nn.Dropout(0.2)
            
            lstm_input_size = input_length // 8
            self.lstm = nn.LSTM(
                input_size=128,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.3,
                bidirectional=True
            )
            
            self.fc1 = nn.Linear(128, 64)
            self.dropout4 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(64, n_classes)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.dropout1(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)
            x = self.dropout2(x)
            
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)
            x = self.dropout3(x)
            
            x = x.permute(0, 2, 1)
            lstm_out, (hidden, cell) = self.lstm(x)
            x = lstm_out[:, -1, :]
            
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout4(x)
            x = self.fc2(x)
            
            return x
    
    # Create and load model
    model = CNN_LSTM(
        n_classes=model_arch['n_classes'],
        input_channels=model_arch['input_channels'],
        input_length=model_arch['input_length']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Extract scaler and other components
    scaler = checkpoint['scaler']
    
    return model, scaler, device


# ── ECG image → signal ────────────────────────────────────────
def ecg_image_to_signal(img_bytes, target_length=187):
    """Convert uploaded image bytes → 187-point normalised ECG signal."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    raw_signal = np.full(w, np.nan)
    for col in range(w):
        dark_rows = np.where(binary[:, col] > 0)[0]
        if len(dark_rows) > 0:
            raw_signal[col] = np.mean(dark_rows)

    nan_mask = np.isnan(raw_signal)
    if nan_mask.all():
        raise ValueError("No waveform detected in image.")

    x_valid = np.where(~nan_mask)[0]
    y_valid  = raw_signal[~nan_mask]
    raw_signal = np.interp(np.arange(w), x_valid, y_valid)

    inverted    = (h - 1) - raw_signal
    signal_norm = (inverted - inverted.min()) / (inverted.max() - inverted.min() + 1e-8)
    signal_187  = resample(signal_norm, target_length)
    signal_187  = np.clip(signal_187, 0.0, 1.0).astype(np.float32)

    return signal_187, img, binary, raw_signal


def predict_ecg(model, signal, scaler, device):
    model.eval()
    signal_scaled = scaler.transform(signal.reshape(1, -1))
    signal_tensor = torch.FloatTensor(signal_scaled).unsqueeze(1).to(device)
    with torch.no_grad():
        output = model(signal_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities.cpu().numpy()[0]


# ── Extraction debug plot ─────────────────────────────────────
def make_extraction_figure(img_bgr, binary, raw_signal, signal_187):
    """Returns a matplotlib figure showing the extraction pipeline."""
    h, w = img_bgr.shape[:2]

    fig = plt.figure(figsize=(14, 5.5), facecolor="#050D1A")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.35, wspace=0.25,
                            left=0.04, right=0.97,
                            top=0.92, bottom=0.06)

    ax_style = dict(facecolor="#0A1628")
    title_kw = dict(color="#F0F6FF", fontsize=9,
                    fontfamily="monospace", pad=6)
    tick_kw  = dict(colors="#5A7A99", labelsize=7)

    # 1. Original
    ax1 = fig.add_subplot(gs[0, :2], **ax_style)
    ax1.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title("01 / ORIGINAL IMAGE", **title_kw)
    ax1.axis("off")

    # 2. Binary
    ax2 = fig.add_subplot(gs[0, 2], **ax_style)
    ax2.imshow(binary, cmap="gray")
    ax2.set_title("02 / OTSU THRESHOLD", **title_kw)
    ax2.axis("off")

    # 3. Overlay
    ax3 = fig.add_subplot(gs[1, :2], **ax_style)
    ax3.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), aspect="auto")
    ax3.plot(np.arange(w), raw_signal,
             color="#00D4AA", linewidth=1.4, alpha=0.9, label="Extracted trace")
    ax3.set_title("03 / WAVEFORM EXTRACTION", **title_kw)
    ax3.legend(fontsize=7, loc="upper right",
               facecolor="#0A1628", edgecolor="#0F2040",
               labelcolor="#00D4AA")
    ax3.axis("off")

    # 4. Final signal
    ax4 = fig.add_subplot(gs[1, 2], **ax_style)
    t = np.arange(len(signal_187))
    ax4.fill_between(t, signal_187, alpha=0.12, color="#00D4AA")
    ax4.plot(t, signal_187, color="#00D4AA", linewidth=1.5)
    ax4.set_title("04 / RESAMPLED  (187 PTS)", **title_kw)
    ax4.set_xlabel("Sample index", color="#5A7A99", fontsize=7)
    ax4.tick_params(axis="x", **tick_kw)
    ax4.tick_params(axis="y", **tick_kw)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#0F2040")
    ax4.grid(True, color="#0F2040", linewidth=0.6)

    return fig


# ── Probability chart ─────────────────────────────────────────
def make_prob_figure(probs, pred_class):
    """Horizontal bar chart of class probabilities."""
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor="#0A1628")
    ax.set_facecolor("#0A1628")

    labels = list(CLASS_NAMES.values())
    colors = ["#00D4AA" if i == pred_class else "#0F2040"
              for i in range(len(labels))]
    edge   = ["#00D4AA" if i == pred_class else "#1A3050"
              for i in range(len(labels))]

    bars = ax.barh(labels, probs * 100,
                   color=colors, edgecolor=edge, linewidth=0.8,
                   height=0.55)

    for bar, p, c in zip(bars, probs, colors):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%",
                va="center", ha="left",
                fontsize=8, fontfamily="monospace",
                color="#00D4AA" if c == "#00D4AA" else "#5A7A99")

    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.tick_params(axis="y", colors="#F0F6FF", labelsize=8)
    ax.tick_params(axis="x", colors="#5A7A99", labelsize=7)
    ax.set_xlabel("Probability (%)", color="#5A7A99", fontsize=7,
                  fontfamily="monospace")
    ax.xaxis.set_tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#0F2040")
    ax.grid(axis="x", color="#0F2040", linewidth=0.6)

    fig.tight_layout(pad=0.6)
    return fig


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


# ═══════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════
def main():

    # ── Header ────────────────────────────────────────────────
    st.markdown("""
    <div class="ecg-header">
        <div>
            <div class="ecg-title">🫀 &nbsp; ECG Arrhythmia Classifier</div>
            <div class="ecg-subtitle">MIT-BIH · 5-CLASS AAMI · CNN-LSTM · SINGLE-BEAT ANALYSIS</div>
        </div>
        <div class="ecg-badge">DEMO BUILD</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ────────────────────────────────────────────
    model_ready = False
    try:
        model, scaler, device = load_model_checkpoint()
        model_ready = True
        # Store in session state for later use
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["device"] = device
    except FileNotFoundError as e:
        error_msg = str(e)
        model_ready = False

    if not model_ready:
        st.markdown(f"""
        <div class="alert-warn">
        ⚠ &nbsp; MODEL NOT LOADED — {error_msg if 'error_msg' in locals() else 'Unknown error'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="ecg-card">
        <div class="card-label">Setup — Ensure Model File Exists</div>
        """, unsafe_allow_html=True)

        st.code("""
# Make sure these files are in the same directory as ecg_ui.py:
#   - ecg_model_checkpoint.pth
#   - (or best_ecg_model.pth and scaler_combined.pkl as backups)

# If running from Colab:
# 1. Download from Google Drive
# 2. Place in the working directory
# 3. Run: streamlit run ecg_ui.py
        """, language="python")

        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()  # Stop execution here if model not loaded


    # ── Layout: two columns ───────────────────────────────────
    col_left, col_right = st.columns([1, 1.35], gap="large")

    # ════════════════════════════════════════════════════════════
    # LEFT COLUMN — Input
    # ════════════════════════════════════════════════════════════
    with col_left:

        st.markdown('<div class="ecg-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Input — ECG Image</div>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop a single-beat ECG image here",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )

        if uploaded:
            img_bytes = uploaded.read()
            st.image(img_bytes, use_container_width=True,
                     caption="Uploaded image")

            # Image metadata
            arr  = np.frombuffer(img_bytes, np.uint8)
            img_ = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_ is not None:
                h, w = img_.shape[:2]
                st.markdown(f"""
                <div class="info-grid">
                    <div class="info-item">Size<br><span>{w} × {h} px</span></div>
                    <div class="info-item">Format<br><span>{uploaded.type.split("/")[1].upper()}</span></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<hr class="ecg-divider">', unsafe_allow_html=True)

        # ── Requirements note ──────────────────────────────────
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace;
                    font-size: 0.68rem; color: #5A7A99; line-height: 1.8;">
        IMAGE REQUIREMENTS<br>
        ─────────────────────<br>
        ● Single beat, R-peak centered<br>
        ● Dark waveform on white background<br>
        ● No grid lines or axis labels<br>
        ● Black/dark ink only (not red/blue)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Analyse button ─────────────────────────────────────
        run_btn = st.button(
            "Analyse ECG",
            disabled=(uploaded is None)
        )

        if uploaded is None:
            st.markdown("""
            <div style="font-family: 'IBM Plex Mono', monospace;
                        font-size: 0.68rem; color: #5A7A99;
                        text-align: center; margin-top: 0.5rem;">
            Upload an image to enable analysis
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # RIGHT COLUMN — Output
    # ════════════════════════════════════════════════════════════
    with col_right:

        if not uploaded:
            # ── Idle state ─────────────────────────────────────
            st.markdown("""
            <div class="ecg-card" style="min-height: 340px; display: flex;
                 align-items: center; justify-content: center;
                 flex-direction: column; gap: 0.8rem;">
                <div style="font-family: 'IBM Plex Mono', monospace;
                            font-size: 2.5rem; opacity: 0.15;">⬡</div>
                <div style="font-family: 'IBM Plex Mono', monospace;
                            font-size: 0.7rem; letter-spacing: 0.2em;
                            text-transform: uppercase; color: #5A7A99;">
                    Awaiting Input
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif run_btn or "last_result" in st.session_state:

            # Run inference if button was just clicked
            if run_btn:
                with st.spinner("Extracting waveform…"):
                    try:
                        img_bytes = uploaded.getvalue()
                        signal_187, img_bgr, binary, raw_sig = \
                            ecg_image_to_signal(img_bytes)
                        time.sleep(0.3)  # brief pause for UX polish

                        pred_class, probs = predict_ecg(
                            st.session_state["model"],
                            signal_187,
                            st.session_state["scaler"],
                            st.session_state["device"]
                        )

                        # Store result
                        st.session_state["last_result"] = {
                            "pred_class":  pred_class,
                            "probs":       probs,
                            "signal_187":  signal_187,
                            "img_bgr":     img_bgr,
                            "binary":      binary,
                            "raw_sig":     raw_sig,
                        }
                        st.session_state["last_error"] = None

                    except Exception as e:
                        st.session_state["last_error"]  = str(e)
                        st.session_state["last_result"] = None

            # ── Render result ──────────────────────────────────
            if st.session_state.get("last_error"):
                st.markdown(f"""
                <div class="alert-warn">
                ⚠ &nbsp; EXTRACTION FAILED — {st.session_state["last_error"]}
                <br><br>
                Check that the image has a dark waveform on a white background
                and the R-peak is centered in the frame.
                </div>
                """, unsafe_allow_html=True)

            elif st.session_state.get("last_result"):
                res        = st.session_state["last_result"]
                pred_class = res["pred_class"]
                probs      = res["probs"]
                confidence = probs[pred_class]
                risk_label, risk_color = CLASS_RISK[pred_class]

                # ── Classification result card ─────────────────
                st.markdown('<div class="ecg-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">Classification Result</div>',
                            unsafe_allow_html=True)

                # Class + risk
                st.markdown(f"""
                <div style="display: flex; align-items: flex-start;
                            justify-content: space-between;">
                    <div>
                        <div style="font-family: 'IBM Plex Mono', monospace;
                                    font-size: 0.65rem; letter-spacing: 0.2em;
                                    text-transform: uppercase; color: #5A7A99;
                                    margin-bottom: 0.4rem;">
                            PREDICTED CLASS
                        </div>
                        <div class="result-class">{CLASS_NAMES[pred_class]}</div>
                        <div class="result-conf">
                            {CLASS_CLINICAL[pred_class]}
                        </div>
                    </div>
                    <div style="text-align: right; flex-shrink: 0; margin-left: 1rem;">
                        <div style="font-family: 'IBM Plex Mono', monospace;
                                    font-size: 0.6rem; letter-spacing: 0.18em;
                                    color: #5A7A99; margin-bottom: 0.3rem;">
                            RISK LEVEL
                        </div>
                        <div style="font-family: 'IBM Plex Mono', monospace;
                                    font-size: 1.0rem; font-weight: 600;
                                    color: {risk_color}; letter-spacing: 0.1em;">
                            {risk_label}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Stat chips
                entropy = -np.sum(probs * np.log(probs + 1e-9))
                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-chip">
                        <div class="chip-val">{confidence*100:.1f}%</div>
                        <div class="chip-lbl">Confidence</div>
                    </div>
                    <div class="stat-chip">
                        <div class="chip-val">{entropy:.2f}</div>
                        <div class="chip-lbl">Entropy</div>
                    </div>
                    <div class="stat-chip">
                        <div class="chip-val">{np.argmax(probs)}</div>
                        <div class="chip-lbl">Class Index</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Low-confidence warning
                if confidence < CONFIDENCE_THRESHOLD:
                    st.markdown(f"""
                    <div class="alert-warn" style="margin-top: 0.9rem;">
                    ⚠ &nbsp; LOW CONFIDENCE ({confidence*100:.1f}%) —
                    Prediction is below the {CONFIDENCE_THRESHOLD*100:.0f}%
                    threshold. Manual review is advised.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-info" style="margin-top: 0.9rem;">
                    ✓ &nbsp; HIGH CONFIDENCE ({confidence*100:.1f}%) —
                    Prediction exceeds the {CONFIDENCE_THRESHOLD*100:.0f}%
                    reliability threshold.
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # ── Probability breakdown ──────────────────────
                st.markdown('<div class="ecg-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">Class Probabilities</div>',
                            unsafe_allow_html=True)

                for i, (name, p) in enumerate(zip(CLASS_NAMES.values(), probs)):
                    active = i == pred_class
                    lbl_cls  = "prob-label active" if active else "prob-label"
                    fill_cls = "prob-fill active"   if active else "prob-fill"
                    pct_cls  = "prob-pct active"    if active else "prob-pct"
                    st.markdown(f"""
                    <div class="prob-row">
                        <div class="{lbl_cls}">{name}</div>
                        <div class="prob-track">
                            <div class="{fill_cls}"
                                 style="width: {p*100:.1f}%"></div>
                        </div>
                        <div class="{pct_cls}">{p*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # ── Extraction pipeline ────────────────────────
                with st.expander("▸  View Extraction Pipeline", expanded=False):
                    ext_fig  = make_extraction_figure(
                        res["img_bgr"], res["binary"],
                        res["raw_sig"], res["signal_187"]
                    )
                    st.image(fig_to_bytes(ext_fig), use_container_width=True)

                    # Signal stats
                    sig = res["signal_187"]
                    st.markdown(f"""
                    <div class="info-grid" style="margin-top: 0.8rem;">
                        <div class="info-item">Signal length<br><span>187 samples</span></div>
                        <div class="info-item">Amplitude range<br><span>[{sig.min():.3f}, {sig.max():.3f}]</span></div>
                        <div class="info-item">Mean<br><span>{sig.mean():.4f}</span></div>
                        <div class="info-item">Std dev<br><span>{sig.std():.4f}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

        elif uploaded and not run_btn:
            # Image uploaded but button not yet clicked
            st.markdown("""
            <div class="ecg-card" style="min-height: 200px; display: flex;
                 align-items: center; justify-content: center;
                 flex-direction: column; gap: 0.6rem;">
                <div style="font-family: 'IBM Plex Mono', monospace;
                            font-size: 0.7rem; letter-spacing: 0.2em;
                            text-transform: uppercase; color: #5A7A99;">
                    Image Ready — Press Analyse ECG
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top: 3rem; border-top: 1px solid #0F2040;
                padding-top: 1rem; display: flex;
                justify-content: space-between; align-items: center;">
        <div style="font-family: 'IBM Plex Mono', monospace;
                    font-size: 0.62rem; color: #2A4060; letter-spacing: 0.1em;">
            TRAINED ON MIT-BIH ARRHYTHMIA DATABASE &nbsp;·&nbsp;
            5-CLASS AAMI EC57 STANDARD &nbsp;·&nbsp; NOT FOR CLINICAL USE
        </div>
        <div style="font-family: 'IBM Plex Mono', monospace;
                    font-size: 0.62rem; color: #2A4060; letter-spacing: 0.1em;">
            CNN-LSTM · PyTorch
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()