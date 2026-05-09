import streamlit as st
import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms  # type: ignore

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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

/* ── reset & base ─────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 50%, #0d0b1a 0%, #050505 50%, #0a0a12 100%);
    background-size: 200% 200%;
    animation: gradientBG 15s ease infinite;
    color: #e4e4e7;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* hide default Streamlit footer & hamburger */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}

/* ── hero ─────────────────────────────────────────────── */
.hero-section {
    text-align: center;
    padding: 4rem 1rem 3rem 1rem;
    position: relative;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4.8rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(to right, #c084fc, #818cf8, #38bdf8, #818cf8, #c084fc);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 6s linear infinite;
    margin: 0;
    line-height: 1.1;
    filter: drop-shadow(0 0 30px rgba(139, 92, 246, 0.25));
}
@keyframes shine {
    to { background-position: 200% center; }
}

.hero-subtitle {
    font-size: 1.3rem;
    font-weight: 300;
    color: #a1a1aa;
    margin-top: 1.2rem;
    letter-spacing: 0.5px;
}

/* ── card / glassmorphism ──────────────────────────────────────────────── */
.card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 2.2rem;
    margin-bottom: 1.5rem;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}
.card:hover {
    border-color: rgba(139, 92, 246, 0.4);
    box-shadow: 0 15px 45px 0 rgba(139, 92, 246, 0.15);
    transform: translateY(-2px);
}

/* ── caption result (WOW effect) ───────────────────────────────────── */
.caption-result {
    position: relative;
    background: rgba(15, 12, 25, 0.7);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
    z-index: 1;
}
.caption-result::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 16px;
    padding: 2px;
    background: linear-gradient(135deg, #c084fc, #38bdf8, #818cf8);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    opacity: 0.8;
    animation: borderGlow 3s ease-in-out infinite alternate;
}
@keyframes borderGlow {
    0% { filter: brightness(1) drop-shadow(0 0 5px rgba(139,92,246,0.3)); }
    100% { filter: brightness(1.5) drop-shadow(0 0 15px rgba(56,189,248,0.6)); }
}

.caption-result p {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 500;
    color: #ffffff;
    margin: 0;
    line-height: 1.5;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    letter-spacing: 0.5px;
}

/* ── constrain image preview ─────────────────────────── */
div[data-testid="stImage"] img {
    max-height: 420px;
    width: auto;
    object-fit: contain;
    margin: 0 auto;
    display: block;
    border-radius: 16px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.05);
}

/* ── section titles ───────────────────────────────────── */
.section-label {
    font-size: 0.95rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #818cf8;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 10px;
    height: 10px;
    background: #818cf8;
    border-radius: 50%;
    box-shadow: 0 0 12px #818cf8;
}

/* ── uploader area ────────────────────────────────────── */
div[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.015);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2.5rem;
    border: 2px dashed rgba(255, 255, 255, 0.15);
    transition: all 0.4s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #c084fc;
    background: rgba(139, 92, 246, 0.05);
    box-shadow: 0 0 30px rgba(139, 92, 246, 0.1);
}

/* uploader inner text */
div[data-testid="stFileUploader"] section { color: #a1a1aa !important; }
div[data-testid="stFileUploader"] section > div { color: #a1a1aa !important; }
div[data-testid="stFileUploader"] small { color: #71717a !important; font-size: .85rem !important; }

/* browse button inside uploader */
div[data-testid="stFileUploader"] button {
    background: rgba(139, 92, 246, 0.1) !important;
    color: #c084fc !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stFileUploader"] button:hover {
    background: rgba(139, 92, 246, 0.25) !important;
    border-color: #c084fc !important;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.3) !important;
    transform: translateY(-1px) !important;
}

/* uploaded file chip */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: rgba(139, 92, 246, 0.1) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: 12px !important;
    color: #f4f4f5 !important;
    padding: 0.5rem !important;
}

/* ── spinner / status ─────────────────────────────────── */
.stSpinner > div {
    border-top-color: #c084fc !important;
}

/* ── generate button ─────────────────────────────────── */
div.stButton > button {
    background: linear-gradient(135deg, #818cf8, #c084fc, #38bdf8);
    background-size: 200% auto;
    color: #ffffff;
    border: none;
    border-radius: 14px;
    padding: 0.9rem 1.8rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    cursor: pointer;
    width: 100%;
    transition: all 0.4s ease;
    box-shadow: 0 10px 20px rgba(139, 92, 246, 0.25);
}
div.stButton > button:hover {
    background-position: right center;
    box-shadow: 0 15px 30px rgba(56, 189, 248, 0.4);
    transform: translateY(-2px) scale(1.01);
}
div.stButton > button:active {
    transform: translateY(1px);
}

/* ── info banner ──────────────────────────────────────── */
.info-banner {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 4rem 2rem;
    text-align: center;
    color: #a1a1aa;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.2);
}
.info-banner .icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    filter: drop-shadow(0 0 15px rgba(255,255,255,0.1));
}
.info-banner .text {
    font-size: 1.2rem;
    font-weight: 300;
}

/* ── status pills ─────────────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.status-pill.ready { 
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(21,128,61,0.15));  
    color: #4ade80; 
    border: 1px solid rgba(34,197,94,0.3);
}
.status-pill.missing { 
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(185,28,28,0.15));  
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.3);
}

/* ── metrics badge ───────────────────────────────────── */
.metrics-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}
.metric-badge {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(192, 132, 252, 0.3);
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #e4e4e7;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.metric-badge:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(56, 189, 248, 0.15);
    color: #ffffff;
}

/* ── footer ───────────────────────────────────────────── */
.custom-footer {
    text-align: center;
    color: #52525b;
    font-size: 0.85rem;
    padding: 3rem 0 1rem 0;
    font-weight: 400;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = "zipped_results"
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pt")
VOCAB_PATH = os.path.join(RESULTS_DIR, "vocab.json")

SUPPORTED_IMAGE_TYPES = [
    "jpg", "jpeg", "png", "bmp", "gif",
    "tiff", "tif", "webp", "ico",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization — must match the CNN_TRANSFORM used during training
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),          # training used direct resize, NOT crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary — mirrors the Vocabulary class used during training
# ─────────────────────────────────────────────────────────────────────────────
class Vocabulary:
    PAD, START, END, UNK = "<PAD>", "<START>", "<END>", "<UNK>"

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}

    @classmethod
    def from_json(cls, path: str) -> "Vocabulary":
        """Load vocabulary from the saved vocab.json file."""
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            word2idx = json.load(f)
        vocab.word2idx = word2idx
        vocab.idx2word = {idx: word for word, idx in word2idx.items()}
        return vocab

    def __len__(self) -> int:
        return len(self.word2idx)

# ─────────────────────────────────────────────────────────────────────────────
# Model architecture — exact mirror of the training notebook
# ─────────────────────────────────────────────────────────────────────────────

class FeatureProjection(nn.Module):
    def __init__(self, cnn_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(cnn_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.relu(self.proj(features))


class SoftAttention(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, attention_dim)
        self.W_h = nn.Linear(hidden_dim, attention_dim)
        self.W_e = nn.Linear(attention_dim, 1)

    def forward(self, A: torch.Tensor, h: torch.Tensor):
        score = self.W_e(
            torch.tanh(self.W_a(A) + self.W_h(h).unsqueeze(1))
        ).squeeze(2)
        alpha = F.softmax(score, dim=1)
        z = (alpha.unsqueeze(2) * A).sum(dim=1)
        return z, alpha


class CaptionerAttention(nn.Module):
    """
    CNN spatial features → attention at every step → LSTM → softmax.

    At every step t:
      zₜ        = Attention(A, h_{t-1})
      hₜ, cₜ   = LSTM([embed(w_{t-1}) ; zₜ], h_{t-1}, c_{t-1})
      P(wₜ)    = softmax(fc(hₜ))
    """
    def __init__(self, vocab_size: int, cnn_dim: int, embed_dim: int,
                 hidden_dim: int, attention_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = FeatureProjection(cnn_dim, hidden_dim)
        self.attention = SoftAttention(hidden_dim, attention_dim)
        self.init_h = nn.Linear(hidden_dim, hidden_dim)
        self.init_c = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, A: torch.Tensor):
        mean_A = A.mean(dim=1)
        return torch.tanh(self.init_h(mean_A)), torch.tanh(self.init_c(mean_A))

    def forward(self, features, captions, lengths):
        batch_size = features.size(0)
        A = self.proj(features)
        embeds = self.embedding(captions)
        h, c = self.init_hidden(A)
        max_len = max(lengths) - 1
        outputs = torch.zeros(batch_size, max_len,
                               self.fc_out.out_features).to(features.device)
        for t in range(max_len):
            z, _ = self.attention(A, h)
            lstm_in = torch.cat([self.dropout(embeds[:, t, :]),
                                  self.dropout(z)], dim=1)
            h, c = self.lstm(lstm_in, (h, c))
            outputs[:, t, :] = self.fc_out(self.dropout(h))
        return outputs

    @torch.no_grad()
    def generate(self, features: torch.Tensor, vocab: Vocabulary,
                 max_len: int = 50):
        """Greedy decoding — single image inference."""
        A = self.proj(features)
        h, c = self.init_hidden(A)
        word_id = torch.tensor([vocab.word2idx[vocab.START]]).to(features.device)
        result, all_alphas = [], []

        _DANGLING = {
            "a", "an", "the", "in", "on", "at", "to", "of", "with",
            "for", "by", "from", "and", "or", "is", "are", "its",
            "his", "her", "their", "into", "over", "near", "through",
        }
        dangling_ids = {vocab.word2idx[w] for w in _DANGLING if w in vocab.word2idx}

        for _ in range(max_len):
            embed = self.embedding(word_id)
            z, alpha = self.attention(A, h)
            lstm_in = torch.cat([embed, z], dim=1)
            h, c = self.lstm(lstm_in, (h, c))
            logits = self.fc_out(h)
            logits[:, vocab.word2idx[vocab.UNK]] = float("-inf")
            # Ban premature <END> if the previous word was an article/preposition
            if word_id.item() in dangling_ids:
                logits[:, vocab.word2idx[vocab.END]] = float("-inf")
                
            word_id = logits.argmax(dim=1)
            word = vocab.idx2word[word_id.item()]
            if word == vocab.END:
                break
            result.append(word)
            all_alphas.append(alpha.squeeze(0).cpu().numpy())

        # Strip trailing dangling function words
        _DANGLING = {
            "a", "an", "the", "in", "on", "at", "to", "of", "with",
            "for", "by", "from", "and", "or", "is", "are", "its",
            "his", "her", "their", "into", "over", "near", "through",
        }
        while result and result[-1] in _DANGLING:
            result.pop()
            if all_alphas:
                all_alphas.pop()

        return " ".join(result), all_alphas

    @torch.no_grad()
    def beam_search(self, features: torch.Tensor, vocab: Vocabulary,
                    beam_size: int = 5, max_len: int = 50):
        """Beam search decoding — produces more complete, higher-quality captions.

        Explores `beam_size` candidate captions in parallel and returns the
        highest-scoring complete sentence (one that ends with <END>).
        Falls back to the best incomplete candidate if none finish.
        """
        device = features.device
        A = self.proj(features)          # (1, 49, hidden)
        h, c = self.init_hidden(A)       # each (1, hidden)

        # Expand for beam_size copies
        A = A.expand(beam_size, -1, -1)  # (beam, 49, hidden)
        h = h.expand(beam_size, -1).contiguous()
        c = c.expand(beam_size, -1).contiguous()

        start_id = vocab.word2idx[vocab.START]
        end_id   = vocab.word2idx[vocab.END]
        unk_id   = vocab.word2idx[vocab.UNK]

        _DANGLING = {
            "a", "an", "the", "in", "on", "at", "to", "of", "with",
            "for", "by", "from", "and", "or", "is", "are", "its",
            "his", "her", "their", "into", "over", "near", "through",
        }
        dangling_ids = {vocab.word2idx[w] for w in _DANGLING if w in vocab.word2idx}

        # Each beam: (log_prob, [word_ids], h, c, [alphas], finished)
        beams = [(0.0, [start_id], h[0:1], c[0:1], [], False)]
        completed = []

        for _ in range(max_len):
            candidates = []
            for log_p, seq, h_b, c_b, alphas_b, done in beams:
                if done:
                    completed.append((log_p, seq, alphas_b))
                    continue

                word_id = torch.tensor([seq[-1]], device=device)
                embed = self.embedding(word_id)           # (1, embed)
                z, alpha = self.attention(
                    A[0:1], h_b
                )                                         # z: (1, hidden)
                lstm_in = torch.cat([embed, z], dim=1)
                h_new, c_new = self.lstm(lstm_in, (h_b, c_b))
                logits = self.fc_out(h_new)                # (1, vocab)
                logits[:, unk_id] = float("-inf")
                # Ban premature <END> if the previous word was an article/preposition
                if seq[-1] in dangling_ids:
                    logits[:, end_id] = float("-inf")
                
                log_probs = F.log_softmax(logits, dim=1).squeeze(0)

                # Take top beam_size expansions from this beam
                topk_logp, topk_ids = log_probs.topk(beam_size)
                alpha_np = alpha.squeeze(0).cpu().numpy()

                for k in range(beam_size):
                    wid = topk_ids[k].item()
                    new_logp = log_p + topk_logp[k].item()
                    new_seq = seq + [wid]
                    new_alphas = alphas_b + [alpha_np]
                    is_done = (wid == end_id)
                    candidates.append(
                        (new_logp, new_seq, h_new, c_new, new_alphas, is_done)
                    )

            if not candidates:
                break

            # Keep top beam_size candidates by length-normalized score
            candidates.sort(
                key=lambda x: x[0] / max(len(x[1]) - 1, 1), reverse=True
            )
            beams = candidates[:beam_size]

            # Collect any newly completed beams
            still_going = []
            for b in beams:
                if b[5]:  # finished
                    completed.append((b[0], b[1], b[4]))
                else:
                    still_going.append(b)
            beams = still_going

            if not beams:
                break

        # Also add any remaining incomplete beams
        for b in beams:
            completed.append((b[0], b[1], b[4]))

        if not completed:
            return "", []

        # Pick the best: prefer completed sentences, score by length-normalized log-prob
        # Give a bonus to completed sequences (those ending with END)
        def score_fn(item):
            lp, seq, _ = item
            length = max(len(seq) - 1, 1)  # exclude START
            ends_properly = (seq[-1] == end_id)
            # Length-normalized log-prob + completion bonus
            return (lp / length) + (1.0 if ends_properly else 0.0)

        best = max(completed, key=score_fn)
        _, best_seq, best_alphas = best

        # Decode word IDs to string, stripping START and END
        words = []
        for wid in best_seq:
            w = vocab.idx2word.get(wid, vocab.UNK)
            if w == vocab.END:
                break
            if w not in (vocab.START, vocab.PAD):
                words.append(w)

        # Post-process: strip trailing dangling function words (articles,
        # prepositions, conjunctions) that create incomplete-sounding captions.
        _DANGLING = {
            "a", "an", "the", "in", "on", "at", "to", "of", "with",
            "for", "by", "from", "and", "or", "is", "are", "its",
            "his", "her", "their", "into", "over", "near", "through",
        }
        while words and words[-1] in _DANGLING:
            words.pop()
            if best_alphas:
                best_alphas = best_alphas[:-1]

        return " ".join(words), best_alphas

# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading vocabulary …")
def load_vocab() -> Vocabulary:
    return Vocabulary.from_json(VOCAB_PATH)


def _build_cnn_extractor(backbone: str) -> nn.Module:
    """Build the CNN architecture (head removed) matching the training setup."""
    if backbone == "resnet101":
        base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        extractor = nn.Sequential(*list(base.children())[:-2])
    elif backbone == "vgg16":
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        extractor = base.features
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return extractor


@st.cache_resource(show_spinner="Loading caption model …")
def load_caption_model():
    """Load the SCST fine-tuned captioner AND fine-tuned CNN from the checkpoint.

    The checkpoint bundles both `model_state` (decoder) and `cnn_state`
    (fine-tuned CNN encoder).  Using the fine-tuned CNN is critical —
    vanilla ImageNet weights produce much weaker captions because the
    top ResNet layers were unfrozen and optimised during SCST training.
    """
    if not os.path.exists(MODEL_PATH):
        return None, None, None

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # ── Decoder ───────────────────────────────────────────────────────
    vocab_size = checkpoint["model_state"]["embedding.weight"].shape[0]
    cnn_dim = checkpoint["model_state"]["proj.proj.weight"].shape[1]
    embed_dim = checkpoint.get("embed_dim", 512)
    hidden_dim = checkpoint.get("hidden_dim", 512)
    attention_dim = checkpoint.get("attention_dim", 512)
    dropout = checkpoint.get("dropout", 0.5)
    backbone = checkpoint.get("backbone", "resnet101")

    model = CaptionerAttention(
        vocab_size=vocab_size,
        cnn_dim=cnn_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    # ── CNN encoder (fine-tuned weights from SCST training) ───────────
    extractor = _build_cnn_extractor(backbone)
    if "cnn_state" in checkpoint and checkpoint["cnn_state"]:
        extractor.load_state_dict(checkpoint["cnn_state"])
    extractor.to(DEVICE)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

    metrics = checkpoint.get("metrics", {})

    return model, extractor, metrics

# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(image: Image.Image, extractor: nn.Module) -> torch.Tensor:
    """Extract CNN spatial features from a PIL image.

    Replicates the exact reshaping used in the training notebook:
        feat = cnn(tensor)                             # (1, C, 7, 7)
        feat = feat.squeeze(0).permute(1,2,0)          # (7, 7, C)
        feat = feat.reshape(49, -1).unsqueeze(0)       # (1, 49, C)
    """
    img = image.convert("RGB")
    img_tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    feat = extractor(img_tensor)                                # (1, C, 7, 7)
    feat = feat.squeeze(0).permute(1, 2, 0).reshape(49, -1)    # (49, C)
    return feat.unsqueeze(0)                                    # (1, 49, C)

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def _model_status_html() -> str:
    if os.path.exists(MODEL_PATH):
        return '<span class="status-pill ready">ready</span>'
    return '<span class="status-pill missing">not found</span>'


def _metrics_html(metrics: dict) -> str:
    if not metrics:
        return ""
    badges = "".join(
        f'<span class="metric-badge">{k}: {v:.4f}</span>'
        for k, v in metrics.items()
    )
    return f'<div class="metrics-row">{badges}</div>'


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
        st.markdown(
            """
            <div class="card" style="padding: 1.2rem 1.4rem;">
                <p style="margin:0; font-weight:600; color:#e4e4e7;">
                    🧠 ResNet101 + Attention LSTM
                </p>
                <p style="margin:.3rem 0 0 0; font-size:.88rem; color:#71717a;">
                    SCST Fine-Tuned · Flickr8k + Flickr30k
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show status pill
        st.markdown(
            f"<p style='margin:.4rem 0 .6rem 0;'>Weights: {_model_status_html()}</p>",
            unsafe_allow_html=True,
        )

        # Load model & show metrics
        caption_model, cnn_extractor, metrics = load_caption_model()
        if metrics:
            st.markdown(
                f"<p class='section-label' style='margin-top:.8rem;'>Performance</p>"
                f"{_metrics_html(metrics)}",
                unsafe_allow_html=True,
            )

        st.markdown(
            '<p class="section-label" style="margin-top:1.6rem;">Upload</p>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drag & drop or browse",
            type=SUPPORTED_IMAGE_TYPES,
            label_visibility="collapsed",
        )

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
                        if caption_model is None:
                            st.error(
                                f"Model weights were not found at `{MODEL_PATH}`.  "
                                "Please make sure `best_model.pt` is present in the "
                                "`zipped_results/` directory."
                            )
                        else:
                            vocab = load_vocab()
                            features = extract_features(image, cnn_extractor)
                            caption, alphas = caption_model.beam_search(
                                features, vocab, beam_size=5, max_len=50
                            )
                            st.session_state.caption = caption
                            st.session_state.alphas = alphas
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
            st.session_state.pop("alphas", None)
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
        '<div class="custom-footer">VisionCaption • Built with Streamlit & PyTorch</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
