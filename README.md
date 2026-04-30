<p align="center">
  <h1 align="center">🔮 VisionCaption — AI Image Captioning</h1>
  <p align="center">
    Generate natural-language descriptions for any image using deep learning.<br/>
    Built with <b>TensorFlow / Keras</b> and deployed as an interactive <b>Streamlit</b> web app.
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
  <a href="https://keras.io/"><img src="https://img.shields.io/badge/Keras-3.x-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-features">Features</a> •
  <a href="#-model-architectures">Models</a> •
  <a href="#-results">Results</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-project-structure">Structure</a>
</p>

---

## 🚀 Live Demo

> **Try it now →** [**VisionCaption on Streamlit Cloud**](https://image-captioning-fhjajrfwi4bus2xtrmu2mq.streamlit.app/)

---

## ✨ Features

| Feature | Description |
|:--------|:------------|
| 🧠 **Three Model Architectures** | Compare captions from VGG16 Flat, VGG16 Spatial + Attention, and ResNet50 Flat encoders |
| 🎨 **Premium Dark UI** | Glassmorphism design with animated gradients, hover effects, and a polished dark-luxury theme |
| ⚡ **On-the-Fly Feature Extraction** | Upload any image — the app automatically extracts features using the matching Keras backbone |
| 🔄 **Model Swapping** | Switch between V1 / V2 / V3 in real time; the UI resets and loads the selected model seamlessly |
| 🔧 **Cross-Version Keras Compatibility** | Custom deserialization patches (`NotEqual`, `Attention`, `Lambda`) ensure models trained on Kaggle (Keras 3.8) load correctly everywhere |
| 📊 **BLEU Score Evaluation** | Corpus-level BLEU-1 through BLEU-4 scores computed on the Flickr8k test set |
| 🔥 **Grad-CAM Visualizations** | Word-level heatmaps show which image regions the model focuses on when predicting each token |
| 📝 **Auto Tokenizer Generation** | If `tokenizer.pkl` is missing, the app downloads Flickr8k via `kagglehub` and rebuilds it automatically |

---

## 🏗️ Model Architectures

The project implements and compares **three** encoder–decoder architectures trained on the **Flickr8k** dataset (8,091 images, 40,455 captions).

### V1 — VGG16 Flat Baseline

```
Image: VGG16 fc2 (4096-d) → Dropout(0.4) → Dense(256, ReLU)
Text:  Embedding(vocab, 256) → Dropout(0.4) → LSTM(256)
Fusion: Add(image, text) → Dense(256, ReLU) → Dense(vocab, softmax)
```

### V2 — VGG16 Spatial + Soft Attention

```
Image: VGG16 block5_pool (7×7×512 → 49×512) → Dense(256) per location
Text:  Embedding(vocab, 256) → Dropout(0.4) → LSTM(256)
Attention: Luong-style dot-product over 49 spatial locations
Fusion: Concat(context, LSTM) → Dense(256, ReLU) → Dense(vocab, softmax)
```

### V3 — ResNet50 Flat *(Best)*

```
Image: ResNet50 avg pool (2048-d) → Dropout(0.4) → Dense(256, ReLU)
Text:  Embedding(vocab, 256) → Dropout(0.4) → LSTM(256)
Fusion: Add(image, text) → Dense(256, ReLU) → Dense(vocab, softmax)
```

> All models use **Adam** optimizer (lr = 1e-3), **categorical cross-entropy** loss, and are trained for up to **30 epochs** with **EarlyStopping** (patience 4), **ReduceLROnPlateau** (factor 0.5, patience 2), and **ModelCheckpoint**.

---

## 📊 Results

### BLEU Scores on Flickr8k Test Set (810 images)

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|:------|:------:|:------:|:------:|:------:|
| V1 – VGG16 Flat | 0.56 | 0.33 | 0.23 | 0.15 |
| V2 – VGG16 Attention | 0.54 | 0.31 | 0.21 | 0.14 |
| **V3 – ResNet50 Flat** | **0.57** | **0.34** | **0.24** | **0.16** |

### Training Summary

| Model | Epochs Run | Best Val Loss | Final Train Loss |
|:------|:----------:|:-------------:|:----------------:|
| V1 – VGG16 Flat | 30 | ≈ 3.54 | ≈ 2.51 |
| V2 – VGG16 Attention | 22 (early stop) | ≈ 3.88 | ≈ 2.66 |
| **V3 – ResNet50 Flat** | **30** | **≈ 3.39** | **≈ 2.27** |

> **Key Finding:** ResNet50 (V3) achieves the best performance with the lowest validation loss and highest BLEU scores, demonstrating that a deeper backbone encoder significantly improves captioning quality.

---

## 🛠️ Getting Started

### Prerequisites

- **Python 3.10+**
- **pip** or **conda**

### 1. Clone the Repository

```bash
git clone https://github.com/Mazen149/Image-Captioning.git
cd Image-Captioning
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 <b>requirements.txt</b></summary>

```
streamlit==1.57.0
numpy==2.4.4
Pillow==12.2.0
tensorflow==2.21.0
kagglehub==1.0.1
contractions==0.1.73
```
</details>

### 3. Generate the Tokenizer

The app requires `models/tokenizer.pkl` to decode model predictions into text. Run the utility script **once** before launching:

```bash
python create_tokenizer.py
```

> **Note:** If the tokenizer file is missing when the app starts, it will automatically download Flickr8k via `kagglehub` and build the tokenizer on the fly.

### 4. Add Model Weights

Place the trained `.h5` weight files inside the `models/` directory:

| File | Model |
|:-----|:------|
| `model_v1_vgg_flat.h5` | V1 — VGG16 Flat |
| `model_v2_vgg_spatial_attention.h5` | V2 — VGG16 Spatial + Attention |
| `model_v3_resnet_flat.h5` | V3 — ResNet50 Flat |

### 5. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
Image-Captioning/
├── app.py                          # Streamlit web application
├── create_tokenizer.py             # Utility to generate tokenizer.pkl
├── Image Captioning Notebook.ipynb # Full training & evaluation notebook (Kaggle)
├── requirements.txt                # Python dependencies
├── main.py                         # Entry point helper
├── models/
│   ├── model_v1_vgg_flat.h5        # V1 weights
│   ├── model_v2_vgg_spatial_attention.h5  # V2 weights
│   └── model_v3_resnet_flat.h5     # V3 weights
```

---

## 📓 Notebook Overview

The Jupyter notebook (`Image Captioning Notebook.ipynb`) is organized into **9 sections**:

1. **Setup & Imports** — Environment configuration, GPU detection, library imports
2. **Data Preparation** — Flickr8k download, caption cleaning, tokenizer fitting, train/val/test split
3. **Data Generator** — Custom batch generators for flat and spatial feature formats
4. **Model V1 — VGG16 Flat** — Architecture definition, training, loss curves
5. **Model V2 — VGG16 Spatial + Attention** — Soft attention architecture, training, loss curves
6. **Model V3 — ResNet50 Flat** — ResNet50 backbone, training, loss curves
7. **Evaluation** — BLEU-1 through BLEU-4 computation on the test set
8. **Grad-CAM Visualization** — Word-level attention heatmaps for interpretability
9. **Sample Predictions** — Side-by-side image + generated caption visualization

> **Training environment:** Kaggle GPU (NVIDIA Tesla P100 / T4), TensorFlow 2.19, Keras 3.8

---

## 🔮 Tech Stack

| Component | Technology |
|:----------|:-----------|
| Deep Learning Framework | TensorFlow 2.21 / Keras 3.x |
| Image Encoders | VGG16, ResNet50 (ImageNet pre-trained) |
| Sequence Decoder | LSTM with Embedding layer |
| Attention Mechanism | Luong-style dot-product soft attention |
| Interpretability | Grad-CAM heatmaps |
| Web Application | Streamlit 1.57 |
| Dataset | Flickr8k (8,091 images, 40,455 captions) |
| Training Platform | Kaggle (GPU) |
