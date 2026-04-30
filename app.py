import streamlit as st
import os
import pickle
import re
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess  # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionCaption — AI Image Captioning",
    page_icon="🔮",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark-luxury theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── reset & base ─────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e4e4e7;
}

/* hide default Streamlit footer & hamburger */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}

/* ── hero ─────────────────────────────────────────────── */
.hero-section {
    text-align: center;
    padding: 3rem 1rem 2rem 1rem;
}
.hero-title {
    font-size: 3.4rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #a78bfa, #6366f1, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 1.15rem;
    font-weight: 300;
    color: #71717a;
    margin-top: .8rem;
}

/* ── card ──────────────────────────────────────────────── */
.card {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    transition: border-color .3s, box-shadow .3s;
}
.card:hover {
    border-color: #6366f1;
    box-shadow: 0 0 24px rgba(99, 102, 241, .15);
}

/* ── caption result ───────────────────────────────────── */
.caption-result {
    background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(167,139,250,.08));
    border: 1px solid rgba(99,102,241,.35);
    border-radius: 12px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin-top: 1.2rem;
    animation: pulse-glow 2s ease-in-out infinite alternate;
}
.caption-result p {
    font-size: 1.35rem;
    font-weight: 500;
    color: #f4f4f5;
    margin: 0;
    line-height: 1.6;
}

/* ── constrain image preview ─────────────────────────── */
div[data-testid="stImage"] img {
    max-height: 350px;
    width: auto;
    object-fit: contain;
    margin: 0 auto;
    display: block;
    border-radius: 10px;
}
@keyframes pulse-glow {
    0%  { box-shadow: 0 0 8px  rgba(99,102,241,.15); }
    100%{ box-shadow: 0 0 24px rgba(99,102,241,.30); }
}

/* ── section titles ───────────────────────────────────── */
.section-label {
    font-size: .85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #a1a1aa;
    margin-bottom: .6rem;
}

/* ── uploader area ────────────────────────────────────── */
div[data-testid="stFileUploader"] {
    background: linear-gradient(145deg, #16161b, #1e1e26);
    border-radius: 16px;
    padding: 1.6rem;
    border: 2px dashed #3f3f46;
    transition: border-color .35s, background .35s, box-shadow .35s;
    position: relative;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #818cf8;
    background: linear-gradient(145deg, #1a1a22, #22222c);
    box-shadow: 0 0 20px rgba(99,102,241,.12);
}

/* uploader inner text */
div[data-testid="stFileUploader"] section {
    color: #71717a !important;
}
div[data-testid="stFileUploader"] section > div {
    color: #71717a !important;
}
div[data-testid="stFileUploader"] small {
    color: #52525b !important;
    font-size: .78rem !important;
}

/* browse button inside uploader */
div[data-testid="stFileUploader"] button {
    background: rgba(99,102,241,.15) !important;
    color: #a78bfa !important;
    border: 1px solid rgba(99,102,241,.3) !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    transition: background .25s, border-color .25s, box-shadow .25s !important;
}
div[data-testid="stFileUploader"] button:hover {
    background: rgba(99,102,241,.25) !important;
    border-color: #818cf8 !important;
    box-shadow: 0 0 12px rgba(99,102,241,.2) !important;
}

/* drag-active state */
div[data-testid="stFileUploader"]:focus-within {
    border-color: #a78bfa;
    box-shadow: 0 0 24px rgba(167,139,250,.18);
}

/* uploaded file chip */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: rgba(99,102,241,.08) !important;
    border: 1px solid rgba(99,102,241,.2) !important;
    border-radius: 10px !important;
    color: #e4e4e7 !important;
}

/* ── spinner / status ─────────────────────────────────── */
.stSpinner > div {
    border-top-color: #6366f1 !important;
}

/* ── selectbox (model selector) ─────────────────────── */
div[data-testid="stSelectbox"] label {
    color: #a1a1aa !important;
    font-weight: 500;
}
div[data-testid="stSelectbox"] > div > div {
    background: #1e1e24;
    border: 1px solid #27272a;
    border-radius: 10px;
    color: #ffffff;
    font-weight: 500;
    transition: border-color .25s, box-shadow .25s;
}
div[data-testid="stSelectbox"] > div > div:hover,
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #a78bfa;
    box-shadow: 0 0 14px rgba(99,102,241,.25);
}

/* ── generate button ─────────────────────────────────── */
div.stButton > button {
    background: linear-gradient(135deg, #6366f1, #818cf8);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: .7rem 1.6rem;
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    width: 100%;
    transition: box-shadow .3s, transform .15s;
}
div.stButton > button:hover {
    box-shadow: 0 0 22px rgba(99,102,241,.45);
    transform: translateY(-1px);
}
div.stButton > button:active {
    transform: translateY(0);
}

/* ── info banner ──────────────────────────────────────── */
.info-banner {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    color: #52525b;
}
.info-banner .icon {
    font-size: 3rem;
    margin-bottom: .6rem;
}
.info-banner .text {
    font-size: 1.05rem;
    font-weight: 400;
}

/* ── status pills ─────────────────────────────────────── */
.status-pill {
    display: inline-block;
    padding: .25rem .9rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .5px;
    text-transform: uppercase;
}
.status-pill.ready   { background: rgba(34,197,94,.15);  color: #4ade80; }
.status-pill.missing { background: rgba(239,68,68,.15);  color: #f87171; }

/* ── footer ───────────────────────────────────────────── */
.custom-footer {
    text-align: center;
    color: #3f3f46;
    font-size: .8rem;
    padding: 2rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "models"
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
MAX_CAPTION_LENGTH = 34

SUPPORTED_IMAGE_TYPES = [
    "jpg", "jpeg", "png", "bmp", "gif",
    "tiff", "tif", "webp", "ico",
]

MODEL_OPTIONS = {
    "🚀 ResNet50 Flat  (V3)": "V3",
    "📦 VGG16 Flat  (V1)": "V1",
    "🔍 VGG16 Spatial Attention  (V2)": "V2",
}

MODEL_WEIGHTS = {
    "V1": "model_v1_vgg_flat.h5",
    "V2": "model_v2_vgg_spatial_attention.h5",
    "V3": "model_v3_resnet_flat.h5",
}

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer helpers (auto-create if missing using kagglehub + exact notebook logic)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_caption(caption: str) -> str:
    """Replicate the exact caption cleaning from the training notebook."""
    import contractions  # type: ignore
    caption = caption.lower()
    caption = contractions.fix(caption)
    caption = re.sub(r'[^a-z\s]', '', caption)
    caption = re.sub(r'\s+', ' ', caption).strip()
    caption = " ".join(w for w in caption.split() if len(w) > 1)
    return caption


def _build_tokenizer() -> Tokenizer:
    """Download Flickr8k via kagglehub and build the same tokenizer used during training."""
    import kagglehub  # type: ignore

    dataset_path = kagglehub.dataset_download("adityajn105/flickr8k")
    captions_file = os.path.join(dataset_path, "captions.txt")

    captions_by_image: dict[str, list[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) < 2:
                continue
            image_id = parts[0].split(".")[0]
            caption = _clean_caption(parts[1])
            caption = "startseq " + caption + " endseq"
            captions_by_image.setdefault(image_id, []).append(caption)

    all_captions = [c for caps in captions_by_image.values() for c in caps]

    tok = Tokenizer()
    tok.fit_on_texts(all_captions)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tok, f)

    return tok


@st.cache_resource(show_spinner="Loading tokenizer …")
def load_tokenizer() -> Tokenizer:
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, "rb") as f:
            return pickle.load(f)
    # Auto-create from Flickr8k captions via kagglehub
    return _build_tokenizer()

# ─────────────────────────────────────────────────────────────────────────────
# Cross-version Keras compatibility
# ─────────────────────────────────────────────────────────────────────────────
# Models trained on Kaggle (TF 2.19 / Keras 3.8) use an internal "NotEqual"
# operation for Embedding masking. The class exists in `keras.src.ops.numpy`
# but isn't auto-registered for deserialization, so we pass it explicitly.

from keras.src.ops.numpy import NotEqual as _NotEqual  # type: ignore


class _FixedAttention(tf.keras.layers.Attention):
    """Patches Attention deserialization: score_mode was saved as a function
    object instead of the string 'dot' in the training Keras version."""

    @classmethod
    def from_config(cls, config):
        if "score_mode" in config and not isinstance(config["score_mode"], str):
            config["score_mode"] = "dot"
        return super().from_config(config)


class _FixedLambda(tf.keras.layers.Lambda):
    """Patches Lambda deserialization: the V2 spatial-attention model uses two
    Lambda layers whose ``output_shape`` wasn't properly serialized across
    Keras versions, and whose serialized function code references ``tf``
    which is not in scope after deserialization.  The two layers are:

    * ``query_expand``    – ``tf.expand_dims(x, 1)``  (N,256) → (N,1,256)
    * ``context_squeeze`` – ``x[:, 0, :]``            (N,1,256) → (N,256)

    We override both ``call`` (to fix the missing ``tf`` reference) and
    ``compute_output_shape`` (to fix the missing output-shape metadata).
    """

    def call(self, inputs, **kwargs):
        # The serialized lambda code references 'tf' which isn't in
        # scope after cross-version deserialization.  Keras 3 wraps the
        # resulting NameError so we can't catch it.  Instead we skip
        # super().call() entirely and implement the two known ops.
        rank = len(inputs.shape)
        if rank == 2:
            # query_expand: (batch, D) → (batch, 1, D)
            return tf.expand_dims(inputs, axis=1)
        if rank == 3:
            # context_squeeze: (batch, 1, D) → (batch, D)
            return inputs[:, 0, :]
        return super().call(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        try:
            return super().compute_output_shape(input_shape)
        except (ValueError, NotImplementedError):
            pass
        # expand_dims: (batch, D) → (batch, 1, D)
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            return (input_shape[0], 1, input_shape[1])
        # squeeze: (batch, 1, D) → (batch, D)
        if (isinstance(input_shape, (list, tuple))
                and len(input_shape) == 3
                and input_shape[1] == 1):
            return (input_shape[0], input_shape[2])
        raise ValueError(
            f"_FixedLambda cannot infer output shape for {input_shape}"
        )


_CUSTOM_OBJECTS = {
    "NotEqual": _NotEqual,
    "Attention": _FixedAttention,
    "Lambda": _FixedLambda,
}

# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading feature extractor …")
def load_feature_extractor(model_key: str):
    if model_key == "V1":
        base = VGG16(weights="imagenet")
        extractor = Model(inputs=base.input, outputs=base.layers[-2].output)
        return extractor, vgg_preprocess, (224, 224)
    elif model_key == "V2":
        base = VGG16(weights="imagenet", include_top=False)
        extractor = Model(inputs=base.input, outputs=base.output)
        return extractor, vgg_preprocess, (224, 224)
    elif model_key == "V3":
        base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        extractor = Model(inputs=base.input, outputs=base.output)
        return extractor, resnet_preprocess, (224, 224)
    return None, None, None


@st.cache_resource(show_spinner="Loading caption model …")
def load_caption_model(model_key: str):
    path = os.path.join(MODEL_DIR, MODEL_WEIGHTS.get(model_key, ""))
    if os.path.exists(path):
        return load_model(path, compile=False, safe_mode=False,
                          custom_objects=_CUSTOM_OBJECTS)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_image(image: Image.Image, target_size: tuple) -> np.ndarray:
    """Convert a PIL image to an RGB numpy array of the correct size."""
    image = image.convert("RGB")  # handles RGBA, palette, greyscale …
    image = image.resize(target_size)
    return np.expand_dims(np.array(image, dtype=np.float32), axis=0)


def extract_features(image: Image.Image, extractor, preprocess_func,
                     target_size, model_key: str = ""):
    arr = _prepare_image(image, target_size)
    arr = preprocess_func(arr)
    features = extractor.predict(arr, verbose=0)
    # V2 spatial-attention model was trained on features reshaped from
    # (7, 7, 512) → (49, 512).  The raw VGG16 extractor produces
    # (1, 7, 7, 512), so we flatten the spatial dims here.
    if model_key == "V2" and features.ndim == 4:
        features = features.reshape(features.shape[0], -1, features.shape[-1])
    return features


def _token_id_to_word(token_id: int, tokenizer: Tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == token_id:
            return word
    return None


def generate_caption(model, features, tokenizer: Tokenizer, max_length: int) -> str:
    text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding="post")
        preds = model.predict([features, seq], verbose=0)
        token_id = int(np.argmax(preds))
        word = _token_id_to_word(token_id, tokenizer)
        if word is None:
            break
        text += " " + word
        if word == "endseq":
            break
    return text.replace("startseq", "").replace("endseq", "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def _model_status_html(key: str) -> str:
    path = os.path.join(MODEL_DIR, MODEL_WEIGHTS.get(key, ""))
    if os.path.exists(path):
        return '<span class="status-pill ready">ready</span>'
    return '<span class="status-pill missing">not found</span>'


def main():
    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-section">
            <p class="hero-title">VisionCaption</p>
            <p class="hero-subtitle">
                Powered by deep learning — upload any image and let the AI describe it in natural language.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Layout ────────────────────────────────────────────────────────────
    left, right = st.columns([2, 3], gap="large")

    # ── Left column: controls ─────────────────────────────────────────────
    with left:
        st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)
        selected_label = st.selectbox(
            "Architecture",
            list(MODEL_OPTIONS.keys()),
            label_visibility="collapsed",
        )
        model_key = MODEL_OPTIONS[selected_label]

        # Show status pill
        st.markdown(
            f"<p style='margin:.4rem 0 1.6rem 0;'>Weights: {_model_status_html(model_key)}</p>",
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-label">Upload</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag & drop or browse",
            type=SUPPORTED_IMAGE_TYPES,
            label_visibility="collapsed",
        )

    # ── Track model changes to clear stale captions ────────────────────────
    if "last_model" not in st.session_state:
        st.session_state.last_model = model_key
    if st.session_state.last_model != model_key:
        st.session_state.last_model = model_key
        st.session_state.pop("caption", None)

    # ── Right column: preview + caption ───────────────────────────────────
    with right:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            st.markdown('<p class="section-label">Preview</p>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

            generate_btn = st.button("✨  Generate Caption", use_container_width=True)

            if generate_btn:
                with st.spinner("Analyzing image …"):
                    try:
                        tokenizer = load_tokenizer()
                        extractor, preprocess_func, target_size = load_feature_extractor(model_key)
                        caption_model = load_caption_model(model_key)

                        if caption_model is None:
                            st.error(
                                f"Model weights for **{selected_label}** were not found in `{MODEL_DIR}/`.  "
                                "Please make sure the `.h5` file is present."
                            )
                        else:
                            features = extract_features(image, extractor, preprocess_func, target_size, model_key)
                            caption = generate_caption(caption_model, features, tokenizer, MAX_CAPTION_LENGTH)
                            st.session_state.caption = caption
                    except Exception as exc:
                        st.error(f"Something went wrong: {exc}")

            # Show persisted caption
            if "caption" in st.session_state:
                st.markdown('<p class="section-label">Generated Caption</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="caption-result"><p>"{st.session_state.caption.capitalize()}"</p></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.session_state.pop("caption", None)
            st.markdown(
                """
                <div class="info-banner">
                    <div class="icon">📷</div>
                    <div class="text">Upload an image on the left to get started.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="custom-footer">VisionCaption • Built with Streamlit & TensorFlow</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
