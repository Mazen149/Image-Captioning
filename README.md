<p align="center">
  <h1 align="center">🔮 VisionCaption — AI Image Captioning</h1>
  <p align="center">
    Generate natural-language descriptions for any image using deep learning.<br/>
    Built with <b>PyTorch</b> and deployed as an interactive <b>Streamlit</b> web app.
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-model-architecture">Architecture</a> •
  <a href="#-results">Results</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-experiments--models-tried">Experiments</a> •
  <a href="#-tech-stack">Tech Stack</a>
</p>

---

## 🚀 Live Demo

> **Try it now →** [**VisionCaption on Streamlit Cloud**](https://image-captioning-fhjajrfwi4bus2xtrmu2mq.streamlit.app/)

---

## 🛠️ Quick Start

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
streamlit
numpy
Pillow
torch
torchvision
```
</details>

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> **Note:** The model weights (`best_model.pt`) and vocabulary (`vocab.json`) are already included in the `zipped_results/` directory. No additional setup is needed.

---

## ✨ Features

Highlights of the best model and the app:

| Feature                             | Description                                                                                                                       |
| :---------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| 🧠 **SCST Fine-Tuned Model**         | State-of-the-art Self-Critical Sequence Training for optimizing CIDEr scores directly                                             |
| 🔍 **Soft Attention Mechanism**      | Bahdanau-style attention over spatial CNN features — the model "looks" at relevant image regions for each word                    |
| 🎯 **Beam Search Decoding**          | Implements beam search with linguistic constraints to ban premature sentence endings and prevent grammatically dangling captions  |
| 🎨 **Premium Dark UI**               | Glassmorphism design with animated gradients, hover effects, and a polished dark-luxury theme                                     |
| ⚡ **On-the-Fly Feature Extraction** | Upload any image — ResNet101 extracts spatial features and the attention LSTM generates a caption                                 |
| 📊 **BLEU & CIDEr Evaluation**       | BLEU-1 through BLEU-4 and CIDEr scores computed on the test set                                                                   |
| 🔥 **Attention Visualizations**      | Word-level attention heatmaps show which image regions the model focuses on when predicting each token                            |
| 📦 **Self-Contained Checkpoint**     | The `best_model.pt` checkpoint bundles model weights, CNN weights, vocabulary, and hyperparameters — no external tokenizer needed |

---

## 🏗️ Model Architecture

The project implements a two-stage **encoder–decoder** architecture trained on the combined **Flickr8k + Flickr30k** datasets.

### Encoder — ResNet101

```
Image → ResNet101 (ImageNet pre-trained, last 2 layers frozen/unfrozen)
     → Remove classification head
     → Spatial features: (batch, 7×7, 2048)
     → Linear projection: (batch, 49, 512)
```

### Decoder — Attention LSTM

```
At each time step t:
  zₜ       = SoftAttention(projected_features, h_{t-1})
  input_t  = [Embedding(w_{t-1}) ; zₜ]
  hₜ, cₜ  = LSTMCell(input_t, h_{t-1}, c_{t-1})
  P(wₜ)   = softmax(Linear(hₜ))
```

### Inference Strategy

* **Beam Search Decoding:** Explores multiple candidate sequences in parallel to find the overall most probable complete sentence.
* **Linguistic Constraints:** Dynamically prevents premature `<END>` tokens after prepositions or articles (e.g., "in", "a", "with") to eliminate grammatically dangling captions.

### Training Pipeline

| Stage                        | Description                                                                     |
| :--------------------------- | :------------------------------------------------------------------------------ |
| **Stage 1 — CE Warmup**      | Cross-entropy training with teacher forcing (CNN frozen → top layers unfrozen)  |
| **Stage 2 — SCST Fine-Tune** | Self-Critical Sequence Training — optimizes CIDEr reward directly via REINFORCE |

> **Hyperparameters:** embed_dim=512, hidden_dim=512, attention_dim=512, dropout=0.5, Adam optimizer

---

## 📊 Results

### Model Comparison — All Scores

| Model                        | Backbone       | Attention    | Data       |   BLEU-1   |   BLEU-2   |   BLEU-4   |   CIDEr    |
| :--------------------------- | :------------- | :----------- | :--------- | :--------: | :--------: | :--------: | :--------: |
| V1 — VGG16 Flat              | VGG16          | None         | Flickr8k   |   0.5084   |   0.3131   |   0.1201   |     —      |
| V2 — VGG16 Spatial           | VGG16          | Dot-product  | Flickr8k   |   0.5476   |   0.3445   |   0.1398   |     —      |
| V3 — ResNet50 Flat           | ResNet50       | None         | Flickr8k   |   0.5942   |   0.3751   |   0.1594   |     —      |
| CE Warmup (ResNet-101)       | ResNet-101     | Bahdanau     | 8k+30k     |   0.4392   |   0.3016   |   0.1279   |   0.0881   |
| **★ SCST Fine-Tuned (Best)** | **ResNet-101** | **Bahdanau** | **8k+30k** | **0.7229** | **0.5377** | **0.2759** | **0.6587** |

> **Key Finding:** SCST fine-tuning dramatically improves all metrics — BLEU-4 jumps from 0.13 to 0.28 and CIDEr from 0.09 to 0.66 — by directly optimizing the evaluation metric via reinforcement learning.

---

## 📂 Project Structure

```
Image-Captioning/
├── app.py                          # Streamlit web application (PyTorch)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── notebooks/
│   ├── Image Captioning Best Model.ipynb
│   └── Image Captioning With Different Models.ipynb
```

---

## 🧪 Experiments & Models Tried

Throughout the development of this project, several architectures and training strategies were explored across two main notebooks in the `notebooks/` directory:

### 1. Exploratory Models (`Image Captioning With Different Models.ipynb`)
This notebook focused on the Flickr8k dataset to evaluate various baseline and attention-based architectures:
- **V1 (Baseline): VGG16 + LSTM** — Extracted a flat 4096-d feature vector from VGG16 and passed it to an LSTM decoder.
- **V2: VGG16 Spatial Features + Attention** — Extracted spatial feature maps from VGG16 (`block5_pool`) to implement an attention mechanism, allowing the model to focus on specific image regions.
- **V3: ResNet50 + LSTM** — Replaced VGG16 with ResNet50, extracting a flat 2048-d vector for a baseline LSTM decoder without attention.

### 2. The Best Model (`Image Captioning Best Model.ipynb`)
This notebook scales up the dataset (Flickr8k + Flickr30k combined) and implements the final, high-performing architecture used in the Streamlit web app:
- **ResNet-101 + Bahdanau Attention** — Extracts robust spatial features combined with soft attention for word-level interpretability.
- **Stage 1 (CE Warmup):** Standard Cross-Entropy training with CNN fine-tuning.
- **Stage 2 (SCST Fine-Tuning):** Self-Critical Sequence Training using Reinforcement Learning to directly optimize the CIDEr metric.

#### Best Model Pipeline Overview
1. **Setup & Configuration** — Hyperparameters, GPU detection, library imports
2. **Data Preparation** — Flickr8k + Flickr30k download, caption cleaning, vocabulary building, train/val/test split
3. **CNN Feature Extraction** — ResNet101 spatial feature extraction and caching
4. **Model Architecture** — FeatureProjection, SoftAttention, CaptionerAttention (LSTM decoder)
5. **Stage 1 — CE Warmup Training** — Cross-entropy training with learning rate scheduling
6. **Stage 2 — SCST Fine-Tuning** — Self-Critical Sequence Training with CIDEr reward
7. **Evaluation** — BLEU-1 through BLEU-4 and CIDEr computation on the test set
8. **Attention Visualization** — Word-level attention heatmaps for interpretability
9. **Sample Predictions** — Side-by-side image + generated caption visualization

> **Training environment:** Kaggle GPU (NVIDIA Tesla T4 / P100), PyTorch 2.x

---

## 🔮 Tech Stack

| Component               | Technology                                        |
| :---------------------- | :------------------------------------------------ |
| Deep Learning Framework | PyTorch 2.x                                       |
| Image Encoder           | ResNet101 (ImageNet pre-trained)                  |
| Sequence Decoder        | Attention LSTM with Bahdanau-style soft attention |
| Inference Strategy      | Beam Search with dynamic linguistic constraints   |
| Training Strategy       | Cross-Entropy Warmup → SCST (REINFORCE)           |
| Interpretability        | Spatial attention heatmaps                        |
| Web Application         | Streamlit                                         |
| Dataset                 | Flickr8k + Flickr30k (combined)                   |
| Training Platform       | Kaggle (GPU)                                      |
