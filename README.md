# Handwritten Digit Recognition System

An intelligent Handwritten Digit Recognition system that identifies digits (0–9) in real-time using Deep Learning. The project features a Convolutional Neural Network (CNN) built with **PyTorch** and trained on the **MNIST dataset**, and a premium **Streamlit web application** dashboard.

---

## Features

1. **🔢 Real-time Digit Predictor**:
   - **Interactive Drawing Canvas**: Draw digits directly on the screen with eraser/brush sizing.
   - **File Uploader**: Upload digit images (PNG/JPG/JPEG). Feature includes auto-detection of background brightness and inversion (e.g., if a user uploads a black ink drawing on white paper, it is automatically inverted to match MNIST's white-on-black format).
   - **Confidence Metrics**: Displays predictions with confidence levels and live, interactive Plotly probability bar charts.
   - **MNIST Bounding Box Alignment**: Sub-pixel centering and scaling of user drawings/uploads to fit the exact distribution of the MNIST dataset for maximum classification robustness.

2. **📊 Model Performance Tab**:
   - Displays evaluation metrics: **Test Accuracy ($\ge 98\%$)**, **Macro Precision**, **Macro Recall**, and **Macro F1-Score**.
   - Features an interactive **Confusion Matrix Heatmap** powered by Plotly to analyze misclassifications.

3. **📈 Training Analysis Tab**:
   - High-fidelity plots showing **Training vs Validation Loss** and **Training vs Validation Accuracy** over epochs to study learning curves.

4. **🧠 CNN Visualization Tab**:
   - **Layer 1 Filters**: Visual representation of the weights (kernels) of the first convolutional layer.
   - **Intermediate Feature Maps**: Live rendering of intermediate activation maps from Conv Block 1 (32 channels) and Conv Block 2 (64 channels) for the active user input digit.

---

## Directory Structure

```
├── model.py                # PyTorch CNN Network architecture
├── train.py                # Dataset download, training loop, evaluation metrics
├── app.py                  # Streamlit Web Dashboard implementation
├── requirements.txt        # Project package dependencies
├── mnist_cnn.pth           # Saved model state dict (generated after training)
└── training_metrics.json   # Training logs and test evaluation statistics (generated after training)
```

---

## Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** (tested up to Python 3.14.5) installed on your system.

### 2. Install Dependencies
Clone this repository (or navigate to the workspace directory) and install the packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Run the training script to fetch the MNIST dataset, train the CNN, evaluate test metrics, and save weights:
```bash
python train.py
```
*Note: Training runs for 15 epochs and completes in about 2 minutes on standard CPUs. Once training completes, it produces `mnist_cnn.pth` and `training_metrics.json`.*

### 4. Run the Streamlit Dashboard
Launch the interactive web interface:
```bash
streamlit run app.py
```
This will start a local server, usually opening automatically in your browser at `http://localhost:8501/`.
