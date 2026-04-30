# Streamlit Image Captioning App

This directory contains the Streamlit application for the Image Captioning Project.

## Prerequisites
Before running the application, make sure you have the following installed in your Python environment:
- `streamlit`
- `tensorflow`
- `pillow`
- `numpy`

If you are running this locally in a Conda environment, make sure to activate the environment that has TensorFlow installed. If you are running this on Kaggle, you can run Streamlit and expose it using `ngrok` or `localtunnel`.

## Missing Tokenizer
The Streamlit application requires `models/tokenizer.pkl` to decode the model's predictions into text. Since the original training data (`captions.txt`) was not present in this directory, I have provided a utility script to download the standard Flickr8k captions and generate the tokenizer.

**Important:** You must run the `create_tokenizer.py` script once *before* running the Streamlit app. This script requires `tensorflow` to be installed.

```bash
# Run this once to generate models/tokenizer.pkl
python create_tokenizer.py
```

## Running the App
Once the tokenizer is created and you have all your model weights (`.h5` files) inside the `models/` directory, you can run the application with:

```bash
streamlit run app.py
```

## Features
- **Premium UI:** Glassmorphism design with a dynamic gradient background.
- **Model Selection:** Choose between V1 (VGG16 Flat), V2 (VGG16 Spatial), or V3 (ResNet50 Flat).
- **On-the-Fly Feature Extraction:** The app automatically extracts the relevant features from any uploaded image using the corresponding Keras Application before generating the caption.
