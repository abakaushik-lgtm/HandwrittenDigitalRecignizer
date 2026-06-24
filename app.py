import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from model import MNISTCNN

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — full dark mode, consistent throughout
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        /* ─── ROOT TOKENS ─────────────────────────────────── */
        :root {
            --bg-base:      #0f172a;
            --bg-surface:   #1e293b;
            --bg-elevated:  #273044;
            --border:       rgba(255,255,255,0.08);
            --accent:       #8b5cf6;
            --accent-light: #a78bfa;
            --pink:         #f472b6;
            --green:        #34d399;
            --blue:         #3b82f6;
            --text-primary: #f1f5f9;
            --text-muted:   #94a3b8;
            --text-faint:   #64748b;
        }

        /* ─── GLOBAL APP SHELL ────────────────────────────── */
        html, body {
            font-family: 'Outfit', sans-serif !important;
        }


        /* ─── SIDEBAR ─────────────────────────────────────── */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {
            background: #0d1526 !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: var(--border) !important;
        }
        /* Sidebar markdown tables */
        section[data-testid="stSidebar"] table {
            background: var(--bg-surface) !important;
            color: var(--text-primary) !important;
            border-collapse: collapse !important;
            width: 100% !important;
            font-size: 0.78rem !important;
        }
        section[data-testid="stSidebar"] th {
            background: var(--bg-elevated) !important;
            color: var(--accent-light) !important;
            padding: 6px 8px !important;
            border: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] td {
            padding: 5px 8px !important;
            border: 1px solid var(--border) !important;
            color: var(--text-muted) !important;
        }


        /* ─── TABS ────────────────────────────────────────── */
        [data-testid="stTabs"] {
            background: transparent !important;
        }
        button[data-baseweb="tab"] {
            background: var(--bg-surface) !important;
            color: var(--text-muted) !important;
            border-bottom: 2px solid transparent !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--accent-light) !important;
            border-bottom: 2px solid var(--accent) !important;
            background: var(--bg-elevated) !important;
        }
        [data-testid="stTabsContent"] {
            background: transparent !important;
        }

        /* ─── RADIO BUTTONS ───────────────────────────────── */
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] span {
            color: var(--text-primary) !important;
        }
        [data-testid="stRadio"] > div {
            background: transparent !important;
        }

        /* ─── FILE UPLOADER ───────────────────────────────── */
        [data-testid="stFileUploader"] {
            background: var(--bg-surface) !important;
            border: 1px dashed var(--accent) !important;
            border-radius: 10px !important;
            padding: 8px !important;
        }
        [data-testid="stFileUploader"] * {
            color: var(--text-primary) !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: var(--text-muted) !important;
        }

        /* ─── BUTTONS ─────────────────────────────────────── */
        [data-testid="stButton"] > button {
            background: linear-gradient(135deg, var(--accent) 0%, #6d28d9 100%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            transition: opacity 0.2s ease !important;
        }
        [data-testid="stButton"] > button:hover {
            opacity: 0.85 !important;
        }

        /* ─── SPINNER / INFO / WARNING BOXES ─────────────── */
        [data-testid="stAlert"] {
            background: var(--bg-surface) !important;
            border-color: var(--border) !important;
            color: var(--text-primary) !important;
        }

        /* ─── IMAGES (uploaded preview) ───────────────────── */
        [data-testid="stImage"] img {
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
        }

        /* ─── PLOTLY CHART BACKGROUNDS ────────────────────── */
        .js-plotly-plot .plotly,
        .js-plotly-plot .plotly .svg-container {
            background: transparent !important;
        }

        /* ─── SCROLLBAR ───────────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-base); }
        ::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 3px; }

        /* ─── METRIC CARDS (custom HTML) ──────────────────── */
        .metric-card {
            background: var(--bg-surface);
            border-radius: 14px;
            padding: 24px 16px;
            border: 1px solid var(--border);
            text-align: center;
            box-shadow: 0 4px 24px rgba(0,0,0,0.3);
            transition: transform 0.25s ease, border-color 0.25s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            border-color: var(--accent);
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--accent-light) !important;
            margin-bottom: 6px;
        }
        .metric-label {
            font-size: 0.82rem;
            color: var(--text-muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }

        /* ─── HEADER / SUBHEADER ──────────────────────────── */
        .main-header {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            background: linear-gradient(to right, var(--accent-light), var(--pink));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            margin-bottom: 0.2rem;
            line-height: 1.15;
        }
        .subheader {
            color: var(--text-muted) !important;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        /* ─── PREDICTION BOX ──────────────────────────────── */
        .prediction-box {
            background: rgba(139, 92, 246, 0.12);
            border-radius: 14px;
            padding: 24px;
            border: 1px solid rgba(139, 92, 246, 0.35);
            text-align: center;
        }
        .prediction-digit {
            font-size: 5rem;
            font-weight: 800;
            color: var(--pink) !important;
            line-height: 1;
            margin: 10px 0;
        }
        .prediction-conf {
            font-size: 1.2rem;
            color: var(--green) !important;
            font-weight: 600;
        }

        /* ─── DIVIDERS ────────────────────────────────────── */
        hr {
            margin: 2rem 0 !important;
            border-color: var(--border) !important;
        }

        /* ─── CANVAS BORDER ───────────────────────────────── */
        canvas {
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
        }
    </style>
""", unsafe_allow_html=True)


# Helper function to load model
@st.cache_resource
def load_model(weights_path):
    model = MNISTCNN()
    # Force loading on CPU to avoid device mismatch issues in Streamlit threads
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Helper function to load training metrics
@st.cache_data
def load_metrics(metrics_path):
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

# Centering and normalization preprocessing function (MNIST style)
def preprocess_digit(img):
    """
    Takes a PIL image, finds the bounding box of the digit, resizes it
    to fit within a 20x20 box preserving aspect ratio, centers it inside
    a 28x28 black canvas, and normalizes it.
    """
    # 1. Convert to grayscale and convert to numpy array
    img_gray = img.convert('L')
    np_img = np.array(img_gray)
    
    # 2. Find bounding box of the digit (pixels > threshold)
    # In case canvas background is dark, find bright pixels
    threshold = 15
    non_empty = np.argwhere(np_img > threshold)
    
    if non_empty.size == 0:
        # Return blank 28x28
        return Image.new('L', (28, 28), 0), np.zeros((1, 1, 28, 28), dtype=np.float32)
    
    # Bounding box coordinates
    ymin, xmin = non_empty.min(axis=0)
    ymax, xmax = non_empty.max(axis=0)
    
    # Crop the digit with a small padding
    cropped = np_img[ymin:ymax+1, xmin:xmax+1]
    
    # 3. Resize cropped image to fit in a 20x20 box
    h, w = cropped.shape
    max_dim = max(h, w)
    scale = 20.0 / max_dim
    new_h, new_w = int(h * scale), int(w * scale)
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    
    cropped_pil = Image.fromarray(cropped)
    resized_digit = cropped_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # 4. Paste centered in 28x28 black canvas
    centered_img = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    centered_img.paste(resized_digit, (paste_x, paste_y))
    
    # 5. Transform to PyTorch Tensor [1, 1, 28, 28] and normalize
    # ToTensor converts PIL Image (0-255) to float tensor in range [0, 1.0]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor_img = transform(centered_img).unsqueeze(0) # [1, 1, 28, 28]
    
    return centered_img, tensor_img

# Paths
weights_file = "mnist_cnn.pth"
metrics_file = "training_metrics.json"

# Page Header
st.markdown('<div class="main-header">Handwritten Digit Recognizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">An intelligent Deep Learning system built on PyTorch and MNIST to recognize digits (0–9) in real-time.</div>', unsafe_allow_html=True)

# Check if model files exist
model_trained = os.path.exists(weights_file) and os.path.exists(metrics_file)

if not model_trained:
    st.warning("⚠️ **Model Weights or Metrics Not Found!**")
    st.info("The CNN is currently being trained or has not been trained yet. Once the background training script completes, this page will automatically update.")
    
    # Display a button to run the training in the foreground as backup
    if st.button("Train Model Now (Takes ~2 mins on CPU)"):
        with st.spinner("Training PyTorch CNN on MNIST dataset..."):
            import subprocess
            result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("🎉 Model trained successfully! Reloading page...")
                st.rerun()
            else:
                st.error("Training failed. Error details:")
                st.code(result.stderr)
        st.stop()
    else:
        st.stop()

# Load model and metrics
model = load_model(weights_file)
metrics = load_metrics(metrics_file)

# --- SIDEBAR CONTENT ---
st.sidebar.markdown("## ⚙️ Model Configurations")
st.sidebar.markdown(f"**Framework**: PyTorch 2.12")
st.sidebar.markdown(f"**Target Device**: CPU (Inference)")

st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 Dataset Statistics")
st.sidebar.markdown("""
- **Dataset**: MNIST
- **Total Images**: 70,000
- **Training Set**: 54,000 images
- **Validation Set**: 6,000 images
- **Test Set**: 10,000 images
- **Image Size**: $28 \\times 28 \\times 1$
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧠 CNN Architecture Summary")

# Render Markdown Table of CNN Architecture in Sidebar
st.sidebar.markdown("""
| Layer | Details | Output Shape |
| :--- | :--- | :--- |
| **Input** | Grayscale | (1, 28, 28) |
| **Conv Block 1** | 32 filters (3x3), ReLU | (32, 28, 28) |
| **MaxPool 1** | Pool size 2x2 | (32, 14, 14) |
| **Conv Block 2** | 64 filters (3x3), ReLU | (64, 14, 14) |
| **MaxPool 2** | Pool size 2x2 | (64, 7, 7) |
| **Flatten** | Reshape | (3136,) |
| **Dense 1** | 128 units, ReLU | (128,) |
| **Dropout** | Rate = 0.5 | (128,) |
| **Dense 2** | Output (Logits) | (10,) |
""")

# --- MAIN TABS ---
tab_predictor, tab_performance, tab_training, tab_cnn = st.tabs([
    "🔢 Digit Predictor",
    "📊 Model Performance",
    "📈 Training Analysis",
    "🧠 CNN Visualization"
])

# Initialize session state for the last processed image and tensor to enable CNN Visualization
if 'last_processed_img' not in st.session_state:
    st.session_state['last_processed_img'] = None
if 'last_processed_tensor' not in st.session_state:
    st.session_state['last_processed_tensor'] = None

# ==================== TAB 1: DIGIT PREDICTOR ====================
with tab_predictor:
    st.header("Upload or Draw a Digit")
    st.write("Draw a single digit in the center of the canvas below or upload a handwritten digit image to classify it.")
    
    col_input, col_pred = st.columns([1, 1], gap="large")
    
    input_image = None
    input_source = None
    
    with col_input:
        input_type = st.radio("Choose Input Method:", ["Interactive Drawing Canvas", "Upload Image File"], horizontal=True)
        
        if input_type == "Interactive Drawing Canvas":
            st.write("Draw here:")
            # Create a drawing canvas
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",  # Fixed fill color
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="canvas",
                display_toolbar=True
            )
            
            if canvas_result.image_data is not None:
                # Check if there are any strokes drawn
                img_data = canvas_result.image_data
                if np.any(img_data[:, :, :3] > 0): # Check if any color pixel is non-zero
                    input_image = Image.fromarray(img_data.astype('uint8'), 'RGBA')
                    input_source = "drawing"
                    
        else:
            uploaded_file = st.file_uploader("Upload Grayscale/RGB Handwritten Digit Image:", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", width=200)
                
                # Check if image background needs inverting
                # convert to grayscale first
                img_gray = img.convert('L')
                # check average of corner pixels to detect if background is white/light
                np_gray = np.array(img_gray)
                corner_avg = (np_gray[0,0] + np_gray[0,-1] + np_gray[-1,0] + np_gray[-1,-1]) / 4.0
                
                if corner_avg > 127: # White background, black ink
                    # Invert colors so that it matches MNIST (white digit on black background)
                    input_image = ImageOps.invert(img_gray)
                else:
                    input_image = img_gray
                    
                input_source = "upload"

    with col_pred:
        if input_image is not None:
            # Preprocess the digit
            centered_img, tensor_img = preprocess_digit(input_image)
            
            # Store in session state for CNN Visualization tab
            st.session_state['last_processed_img'] = centered_img
            st.session_state['last_processed_tensor'] = tensor_img
            
            # Run inference
            with torch.no_grad():
                logits = model(tensor_img)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()
                
            predicted_digit = int(np.argmax(probs))
            # Cap display confidence at 99.99% — true 100.00% looks unrealistic
            raw_conf = probs[predicted_digit] * 100
            display_conf = min(raw_conf, 99.99)
            # Colour the bar: green ≥90%, amber 70-89%, red <70%
            bar_color = "#34d399" if display_conf >= 90 else ("#fbbf24" if display_conf >= 70 else "#f87171")

            # Render Prediction Results
            st.subheader("Prediction Output")

            col_crop, col_stats = st.columns([1, 2])
            with col_crop:
                st.image(centered_img, caption="Preprocessed Input (28x28)", width=120)

            with col_stats:
                st.markdown(f"""
                    <div class="prediction-box">
                        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px;">
                            <div class="metric-label">Predicted Digit</div>
                            <div class="metric-label" style="font-size:0.75rem;">Confidence</div>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                            <div class="prediction-digit" style="margin:0;">{predicted_digit}</div>
                            <div style="text-align:right;">
                                <div style="font-size:1.9rem; font-weight:800; color:{bar_color};">{display_conf:.2f}%</div>
                            </div>
                        </div>
                        <div style="background:rgba(255,255,255,0.08); border-radius:999px; height:10px; overflow:hidden;">
                            <div style="
                                width:{display_conf:.2f}%;
                                height:100%;
                                border-radius:999px;
                                background:linear-gradient(90deg, {bar_color}aa, {bar_color});
                                box-shadow:0 0 8px {bar_color}88;
                                transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                            "></div>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:4px;">
                            <span style="font-size:0.7rem; color:var(--text-faint);">0%</span>
                            <span style="font-size:0.7rem; color:var(--text-faint);">50%</span>
                            <span style="font-size:0.7rem; color:var(--text-faint);">100%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.write("**Class Probabilities:**")
            
            # Render horizontal bar chart of probabilities
            df_probs = pd.DataFrame({
                'Digit': [str(i) for i in range(10)],
                'Probability': probs * 100
            })
            
            fig = px.bar(
                df_probs,
                x='Probability',
                y='Digit',
                orientation='h',
                labels={'Probability': 'Confidence (%)', 'Digit': 'Digit '},
                color='Probability',
                color_continuous_scale='Blues',
                text='Probability'
            )
            
            fig.update_layout(
                margin=dict(l=0, r=20, t=10, b=10),
                height=320,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 100]),
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 **Awaiting Input**: Draw a digit on the canvas or upload an image in the left panel to trigger predictions.")


# ==================== TAB 2: MODEL PERFORMANCE ====================
with tab_performance:
    st.header("Test Set Evaluation Metrics")
    st.write("These metrics are calculated on the unseen test dataset (10,000 images) after model training.")
    
    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['test_accuracy'] * 100:.2f}%</div>
                <div class="metric-label">Test Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['test_precision']:.4f}</div>
                <div class="metric-label">Macro Precision</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['test_recall']:.4f}</div>
                <div class="metric-label">Macro Recall</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['test_f1']:.4f}</div>
                <div class="metric-label">Macro F1-Score</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Confusion Matrix Heatmap
    st.subheader("Confusion Matrix")
    st.write("Heatmap showcasing where model classifications occur. Diagonal entries represent correct predictions.")
    
    cm = np.array(metrics['confusion_matrix'])
    
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted Digit", y="True Digit", color="Count"),
        x=[str(i) for i in range(10)],
        y=[str(i) for i in range(10)],
        color_continuous_scale="Purples",
        text_auto=True
    )
    
    fig_cm.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f8fafc',
    )
    st.plotly_chart(fig_cm, use_container_width=True)


# ==================== TAB 3: TRAINING ANALYSIS ====================
with tab_training:
    st.header("CNN Training Analysis & Learning Curves")
    st.write("Visualize how training and validation metrics changed across epochs during training.")
    
    epochs = metrics['epochs']
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    train_acc = [a * 100 for a in metrics['train_acc']]
    val_acc = [a * 100 for a in metrics['val_acc']]
    
    col_loss, col_acc = st.columns(2)
    
    with col_loss:
        st.subheader("Training & Validation Loss")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Training Loss', line=dict(color='#8b5cf6', width=2)))
        fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation Loss', line=dict(color='#f472b6', width=2, dash='dash')))
        
        fig_loss.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Cross Entropy Loss",
            margin=dict(l=0, r=0, t=30, b=10),
            plot_bgcolor='rgba(255,255,255,0.02)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(15,23,42,0.8)")
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
    with col_acc:
        st.subheader("Training & Validation Accuracy")
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Training Accuracy', line=dict(color='#34d399', width=2)))
        fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation Accuracy', line=dict(color='#3b82f6', width=2, dash='dash')))
        
        fig_acc.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Accuracy (%)",
            margin=dict(l=0, r=0, t=30, b=10),
            plot_bgcolor='rgba(255,255,255,0.02)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99, bgcolor="rgba(15,23,42,0.8)")
        )
        st.plotly_chart(fig_acc, use_container_width=True)


# ==================== TAB 4: CNN VISUALIZATION ====================
with tab_cnn:
    st.header("Inside the Brain of the Convolutional Neural Network")
    st.write("Understand feature extraction by visualizing CNN filter kernels and layer activation maps.")
    
    st.subheader("1. Convolutional Layer 1 Filters (First Layer Weights)")
    st.write("These 32 filters of shape $3 \\times 3$ learn simple patterns like edges, curves, and textures. Warm colors represent positive weights, cool colors represent negative weights.")
    
    # Get weights of conv1
    conv1_weights = model.conv1.weight.detach().numpy() # [32, 1, 3, 3]
    
    # Render weights in matplotlib grid and output to streamlit
    fig_w, axes_w = plt.subplots(4, 8, figsize=(12, 6))
    fig_w.patch.set_facecolor('#0f172a') # Match theme
    
    for i, ax in enumerate(axes_w.flat):
        if i < 32:
            kernel = conv1_weights[i, 0]
            im = ax.imshow(kernel, cmap='RdBu', vmin=-1.0, vmax=1.0)
            ax.axis('off')
            ax.set_title(f"F{i+1}", fontsize=8, color='#94a3b8')
            
    fig_w.subplots_adjust(right=0.9)
    cbar_ax = fig_w.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig_w.colorbar(im, cbar_ax)
    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')
    
    st.pyplot(fig_w)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("2. Intermediate Layer Activation Maps (Feature Maps)")
    
    # Check if there is an image to visualize
    if st.session_state['last_processed_tensor'] is not None:
        tensor_in = st.session_state['last_processed_tensor']
        
        # Pass through Conv 1
        with torch.no_grad():
            conv1_out = model.conv1(tensor_in)
            relu1_out = torch.relu(conv1_out)
            pool1_out = model.pool1(relu1_out)
            
            # Pass through Conv 2
            conv2_out = model.conv2(pool1_out)
            relu2_out = torch.relu(conv2_out)
            pool2_out = model.pool2(relu2_out)
            
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("#### Conv Block 1 Feature Maps (32 Channels, size 28x28)")
            st.write("Activations immediately after the first convolution block:")
            
            feat_maps_1 = relu1_out.squeeze().numpy() # [32, 28, 28]
            
            fig_fm1, axes_fm1 = plt.subplots(4, 8, figsize=(10, 6))
            fig_fm1.patch.set_facecolor('#0f172a')
            
            for idx, ax in enumerate(axes_fm1.flat):
                if idx < 32:
                    ax.imshow(feat_maps_1[idx], cmap='magma')
                    ax.axis('off')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            st.pyplot(fig_fm1)
            
        with col_m2:
            st.markdown("#### Conv Block 2 Feature Maps (64 Channels, size 14x14)")
            st.write("Activations immediately after the second convolution block:")
            
            feat_maps_2 = relu2_out.squeeze().numpy() # [64, 14, 14]
            
            fig_fm2, axes_fm2 = plt.subplots(8, 8, figsize=(10, 10))
            fig_fm2.patch.set_facecolor('#0f172a')
            
            for idx, ax in enumerate(axes_fm2.flat):
                if idx < 64:
                    ax.imshow(feat_maps_2[idx], cmap='magma')
                    ax.axis('off')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            st.pyplot(fig_fm2)
    else:
        st.info("💡 **Awaiting Input**: Draw or upload a digit in the **Digit Predictor** tab to see how the model extracts its features in real-time.")
