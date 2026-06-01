# =======================================================
#  casting_inspector.py
#  Cast Defect Detection System — MGT388 Group 15
#  COMSATS University Islamabad | Spring 2026
#
#  HOW TO RUN:
#    streamlit run casting_inspector.py
#
#  REQUIREMENTS:
#    pip install -r requirements.txt
#
#  PUT CDD_model.pth in the SAME FOLDER as this file.
# =======================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import time
import os
import io

# ──────────────────────────────────────────────────────
# PAGE CONFIG  ← must be the FIRST Streamlit command
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Cast Defect Inspector | MGT388",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ──────────────────────────────────────────────────────
# CUSTOM CSS  — Industrial dark-blue theme
# ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@400;600;700;800&display=swap');

/* ── Global ── */
.stApp {
    background-color: #080c18;
    font-family: 'Sora', sans-serif;
}
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1300px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: #94a3b8 !important;
    font-size: 13px;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 24px 0 12px;
}
.hero h1 {
    font-family: 'Sora', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #93c5fd 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero p { color: #64748b; font-size: 14px; }

/* ── Divider ── */
.divider { border-top: 1px solid #1a2744; margin: 16px 0; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #0d1525 !important;
    border: 2px dashed #2a4a8a !important;
    border-radius: 14px !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
}

/* ── Result cards ── */
.result-defective {
    background: linear-gradient(135deg, #3b0a0a 0%, #1a0505 100%);
    border: 2px solid #dc2626;
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    animation: pulse-red 2s ease-in-out infinite;
}
.result-ok {
    background: linear-gradient(135deg, #052e16 0%, #021a0d 100%);
    border: 2px solid #16a34a;
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    animation: pulse-green 2s ease-in-out infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
    50%       { box-shadow: 0 0 18px 4px rgba(220,38,38,0.18); }
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(22,163,74,0); }
    50%       { box-shadow: 0 0 18px 4px rgba(22,163,74,0.18); }
}
.result-label-defective {
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #fca5a5;
    letter-spacing: 2px;
}
.result-label-ok {
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #86efac;
    letter-spacing: 2px;
}
.result-sub { font-size: 13px; color: #94a3b8; margin-top: 6px; }

/* ── Info cards ── */
.info-card {
    background: #0f1e35;
    border-left: 3px solid #3b82f6;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 13px;
    color: #cbd5e1;
    line-height: 1.75;
}

/* ── Stat badge ── */
.stat-box {
    background: #0d1525;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 14px 12px;
    text-align: center;
}
.stat-val { font-size: 22px; font-weight: 800; color: #60a5fa; }
.stat-lbl { font-size: 11px; color: #4b5563; margin-top: 3px; }

/* ── Code monospace spans ── */
.mono { font-family: 'JetBrains Mono', monospace; font-size: 12px;
        background: #0d1525; padding: 2px 6px; border-radius: 4px;
        color: #60a5fa; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 1px solid #1e2d4a;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 13px;
    border-radius: 8px 8px 0 0;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #0f1e35 !important;
    color: #60a5fa !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* ── Metric tweaks ── */
[data-testid="stMetricValue"] {
    font-family: 'Sora', sans-serif !important;
    font-weight: 800 !important;
    color: #60a5fa !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 12px !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    padding: 8px 22px !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div { border-radius: 10px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: #0d1525 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 8px !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# CNN MODEL DEFINITION  ← MUST match training exactly
# ══════════════════════════════════════════════════════
class CDD(nn.Module):
    """
    Cast Defect Detector CNN
    3 Conv blocks (32→64→128 filters) + 2 FC layers
    Input : RGB image tensor  (batch, 3, 128, 128)
    Output: 2 class scores    (def_front, ok_front)
    """
    def __init__(self):
        super(CDD, self).__init__()

        # Conv Block 1: 3 channels → 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=32,  kernel_size=3, padding=1, stride=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Conv Block 2: 32 → 64 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Conv Block 3: 64 → 128 feature maps
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn3   = nn.BatchNorm2d(128)

        # Shared MaxPool  (halves spatial dims each time)
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        # Fully Connected: 128×16×16 = 32,768 → 256 → 2
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B,3,128,128) → (B,32,64,64)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B,32,64,64)  → (B,64,32,32)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (B,64,32,32)  → (B,128,16,16)
        x = x.view(x.size(0), -1)                       # Flatten        → (B,32768)
        x = F.relu(self.fc1(x))                         # FC1            → (B,256)
        x = self.dropout(x)                              # Dropout
        x = self.fc2(x)                                  # FC2            → (B,2)
        return x


# ──────────────────────────────────────────────────────
# IMAGE TRANSFORMS  ← same pipeline used during training
# ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((128, 128)),               # Shrink to 128×128
    transforms.ToTensor(),                        # PIL → tensor, [0,255]→[0,1]
    transforms.Normalize(                         # [0,1] → [-1,+1]
        mean=[0.5, 0.5, 0.5],
        std =[0.5, 0.5, 0.5]
    )
])

CLASS_NAMES = ['def_front', 'ok_front']


# ──────────────────────────────────────────────────────
# LOAD MODEL  (cached — runs only once per session)
# ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """
    Load the CDD model weights from disk.
    Returns (model, device, error_string_or_None)
    """
    if not os.path.exists(path):
        return None, 'cpu', f'File not found: {path}'
    try:
        device = torch.device('cpu')   # CPU is fine for inference
        model  = CDD().to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        model.eval()                   # Dropout OFF, BN uses running stats
        return model, device, None
    except Exception as e:
        return None, 'cpu', str(e)


# ──────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ──────────────────────────────────────────────────────
def predict(pil_image: Image.Image, model, device):
    """
    Run one image through CDD model.
    Returns: (class_name, confidence_%, [prob_def, prob_ok], time_ms)
    """
    img    = pil_image.convert('RGB')               # ensure 3 channels
    tensor = transform(img).unsqueeze(0).to(device) # (1,3,128,128)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)                       # raw scores
        probs  = F.softmax(logits, dim=1)            # → probabilities
        conf, pred_idx = torch.max(probs, dim=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    pred_class = CLASS_NAMES[pred_idx.item()]
    return pred_class, conf.item() * 100, probs[0].tolist(), elapsed_ms


# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Cast Defect Inspector")
    st.markdown("**Group 15 | MGT388 Deep Learning**")
    st.markdown("---")

    # ── Model path ──
    st.markdown("### ⚙️ Model File")
    model_path = st.text_input(
        label      = "Path to CDD_model.pth",
        value      = "CDD_model.pth",
        help       = "Must be in the same folder as casting_inspector.py"
    )

    model, device, err = load_model(model_path)

    if err is None:
        st.success("✅ Model loaded")
        total_params = sum(p.numel() for p in model.parameters())
        st.caption(f"Parameters: {total_params:,}")
    else:
        st.error(f"❌ {err}")
        st.markdown("""
        <div class='info-card'>
            <b>Fix:</b> Download <span class='mono'>CDD_model.pth</span>
            from your Google Drive and place it in the same folder as
            <span class='mono'>casting_inspector.py</span>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Architecture ──
    st.markdown("### 🧠 Architecture")
    st.markdown("""
    <div class="info-card">
        <b>Name:</b> CDD (Cast Defect Detector)<br>
        <b>Framework:</b> PyTorch<br>
        <b>Conv Blocks:</b> 3 (32 → 64 → 128 filters)<br>
        <b>Kernel:</b> 3×3 · Pad 1 · Stride 1<br>
        <b>Pooling:</b> MaxPool2d (2×2)<br>
        <b>FC Layers:</b> 32,768 → 256 → 2<br>
        <b>Dropout:</b> 50% · BatchNorm2d (bonus)<br>
        <b>Activation:</b> ReLU<br>
        <b>Input:</b> 128×128 RGB image
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Training stats ──
    st.markdown("### 📊 Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Test Accuracy",  "99.86%")
        st.metric("False Positives", "0")
    with c2:
        st.metric("Train Accuracy", "96.74%")
        st.metric("False Negatives", "1")
    st.caption("Trained on 6,633 images · Tested on 715 images")

    st.markdown("---")
    st.caption("COMSATS University Islamabad · Spring 2026")


# ══════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════

# ── Hero ──
st.markdown("""
<div class="hero">
  <h1>🏭 Cast Defect Detection System</h1>
  <p>Upload a pump impeller image · AI classifies it as Defective or Non-Defective in milliseconds</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Tabs ──
tab_single, tab_batch, tab_about = st.tabs([
    "🖼️  Single Image",
    "📦  Batch Prediction",
    "ℹ️  About the Model"
])


# ╔══════════════════════════════════════════╗
# ║  TAB 1 — Single Image Prediction         ║
# ╚══════════════════════════════════════════╝
with tab_single:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Upload column ──
    with col_left:
        st.markdown("### 📤 Upload Casting Image")
        uploaded = st.file_uploader(
            label   = "Drag & drop or click to browse",
            type    = ['jpg', 'jpeg', 'png'],
            key     = "single_upload",
            label_visibility = "collapsed"
        )

        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption=f"📎 {uploaded.name}", use_column_width=True)

            # Image metadata
            st.markdown(f"""
            <div class="info-card">
                📐 <b>Original size:</b> {image.size[0]} × {image.size[1]} px &nbsp;·&nbsp;
                🎨 <b>Mode:</b> {image.mode} &nbsp;·&nbsp;
                📁 <b>{uploaded.name}</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0d1525;border:2px dashed #1e2d4a;border-radius:14px;
                        padding:48px 24px;text-align:center;color:#4b5563;font-size:14px;">
                📷 Upload an impeller image (JPG / PNG)<br>
                <span style="font-size:12px;">Supported: casting front-view images 300×300 recommended</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Result column ──
    with col_right:
        st.markdown("### 🎯 Detection Result")

        if uploaded is None:
            st.markdown("""
            <div style="background:#0d1525;border:2px dashed #1e2d4a;border-radius:14px;
                        padding:48px 24px;text-align:center;color:#4b5563;font-size:14px;">
                ⬅️ Upload an image to see the AI prediction
            </div>
            """, unsafe_allow_html=True)

        elif model is None:
            st.error("❌ Model not loaded. Fix the model path in the sidebar first.")

        else:
            with st.spinner("🔍 Analyzing image..."):
                image = Image.open(uploaded)
                pred_class, confidence, probs, ms = predict(image, model, device)

            # ── Result badge ──
            if pred_class == 'def_front':
                st.markdown(f"""
                <div class="result-defective">
                    <div class="result-label-defective">⚠️ DEFECTIVE</div>
                    <div class="result-sub">Casting defect detected in this impeller</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-ok">
                    <div class="result-label-ok">✅ NON-DEFECTIVE</div>
                    <div class="result-sub">No casting defect found — part is OK</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Probability bars ──
            st.markdown("#### Confidence Breakdown")

            def_pct = probs[0] * 100
            ok_pct  = probs[1] * 100

            st.markdown(f"**⚠️ Defective (def_front)**")
            st.progress(def_pct / 100,
                        text=f"{def_pct:.2f}%")

            st.markdown(f"**✅ Non-Defective (ok_front)**")
            st.progress(ok_pct / 100,
                        text=f"{ok_pct:.2f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Metrics row ──
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Prediction",  pred_class)
            with m2:
                st.metric("Confidence",  f"{confidence:.2f}%")
            with m3:
                st.metric("Inference",   f"{ms:.1f} ms")

            # ── Recommendation ──
            if pred_class == 'def_front':
                st.error("🚫 **Recommendation:** Remove this part from the production line. Manual inspection required.")
            else:
                st.success("🟢 **Recommendation:** Part passed quality check. Cleared for assembly.")


# ╔══════════════════════════════════════════╗
# ║  TAB 2 — Batch Prediction                ║
# ╚══════════════════════════════════════════╝
with tab_batch:
    st.markdown("### 📦 Batch Image Prediction")
    st.markdown("Upload **multiple images** at once. Results appear in a table you can download as CSV.")

    batch_files = st.file_uploader(
        label             = "Upload multiple casting images",
        type              = ['jpg', 'jpeg', 'png'],
        accept_multiple_files = True,
        key               = "batch_upload"
    )

    if batch_files and model is None:
        st.error("❌ Model not loaded. Fix the model path in the sidebar first.")

    elif batch_files:
        st.markdown(f"**{len(batch_files)} image(s) queued for prediction...**")

        progress_bar = st.progress(0, text="Starting...")
        results      = []

        for i, f in enumerate(batch_files):
            img = Image.open(f)
            pred, conf, probs, ms = predict(img, model, device)

            results.append({
                "Filename"           : f.name,
                "Result"             : "⚠️ DEFECTIVE" if pred == "def_front" else "✅ NON-DEFECTIVE",
                "Confidence (%)"     : round(conf, 2),
                "def_front prob (%)" : round(probs[0] * 100, 2),
                "ok_front prob (%)"  : round(probs[1] * 100, 2),
                "Inference (ms)"     : round(ms, 1),
            })

            pct  = (i + 1) / len(batch_files)
            progress_bar.progress(pct, text=f"Analyzing {f.name} ({i+1}/{len(batch_files)})...")

        progress_bar.empty()

        # ── Summary metrics ──
        n_defective = sum(1 for r in results if "DEFECTIVE" in r["Result"] and "NON" not in r["Result"])
        n_ok        = len(results) - n_defective

        st.markdown("#### Batch Summary")
        s1, s2, s3, s4 = st.columns(4)
        with s1: st.metric("Total Images",    len(results))
        with s2: st.metric("⚠️ Defective",    n_defective)
        with s3: st.metric("✅ Non-Defective", n_ok)
        with s4: st.metric("Defect Rate",     f"{n_defective/len(results)*100:.1f}%")

        st.markdown("#### Results Table")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Download CSV ──
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label     = "⬇️  Download Results as CSV",
            data      = csv_data,
            file_name = "cast_defect_results.csv",
            mime      = "text/csv"
        )
    else:
        st.info("👆 Upload multiple images above to begin batch classification.")


# ╔══════════════════════════════════════════╗
# ║  TAB 3 — About the Model                 ║
# ╚══════════════════════════════════════════╝
with tab_about:
    st.markdown("### ℹ️ About the Cast Defect Detection Model")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏗️ Architecture (CDD Model)")
        arch_data = {
            "Layer"        : ["Conv Block 1", "Conv Block 2", "Conv Block 3", "Flatten", "FC Layer 1", "Dropout", "FC Layer 2"],
            "Operation"    : ["Conv2d+BN+ReLU+Pool", "Conv2d+BN+ReLU+Pool", "Conv2d+BN+ReLU+Pool",
                              "view(batch, -1)", "Linear+ReLU", "p=0.5", "Linear (output)"],
            "Output Shape" : ["32 × 64 × 64", "64 × 32 × 32", "128 × 16 × 16",
                              "32,768", "256", "256 (50% off)", "2 classes"],
        }
        st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 📊 Training Configuration")
        config_data = {
            "Hyperparameter" : ["Loss Function", "Optimizer", "Learning Rate", "Batch Size",
                                "Epochs", "Dropout", "Regularization", "Image Size", "Normalization"],
            "Value"          : ["CrossEntropyLoss", "Adam", "0.001", "32", "15", "50% (p=0.5)",
                                "BatchNorm2d (bonus)", "128×128 px", "mean=0.5, std=0.5"],
        }
        st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)

    st.markdown("#### 🎯 Test Set Performance")
    perf_data = {
        "Metric"   : ["Test Accuracy", "Precision (def_front)", "Recall (def_front)",
                      "F1-Score", "True Positives (TP)", "True Negatives (TN)",
                      "False Positives (FP)", "False Negatives (FN)"],
        "Value"    : ["99.86%", "1.00 (100%)", "0.998 (99.78%)",
                      "1.00 (rounded)", "452", "262", "0", "1"],
        "Meaning"  : ["714/715 correct", "No good parts rejected wrongly",
                      "1 defect missed out of 453", "Combined precision+recall",
                      "Defective correctly caught", "OK parts correctly passed",
                      "Zero good parts wasted", "One defect slipped through"],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    st.markdown("#### 📚 Dataset Info")
    st.markdown("""
    <div class="info-card">
        <b>Source:</b> Casting Product Image Data for Quality Inspection (Kaggle)<br>
        <b>Subject:</b> Top-view images of submersible pump impellers<br>
        <b>Classes:</b> def_front (defective) · ok_front (non-defective)<br>
        <b>Training:</b> 3,758 defective + 2,875 OK = <b>6,633 images</b><br>
        <b>Testing:</b>  453 defective + 262 OK = <b>715 images</b><br>
        <b>Original size:</b> 300×300 pixels · RGB · 3 channels<br>
        <b>Resized to:</b> 128×128 pixels for model input
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center;color:#2d3748;font-size:12px;font-family:"Sora",sans-serif;'>
    MGT388 Deep Learning &nbsp;·&nbsp; Group 15 &nbsp;·&nbsp;
    COMSATS University Islamabad &nbsp;·&nbsp; Spring 2026<br>
    Model: CDD (Cast Defect Detector) &nbsp;·&nbsp; Test Accuracy: 99.86% &nbsp;·&nbsp;
    Built with Streamlit + PyTorch
</p>
""", unsafe_allow_html=True)
