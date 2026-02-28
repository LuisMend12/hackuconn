# 🛡️ HackUConn — AI Image Protection Toolkit

A comprehensive toolkit for **protecting images from unauthorized AI usage**. Built at HackUConn, this project combines three complementary defense strategies:

| Tool | What it does | Approach |
|------|-------------|----------|
| **PhotoGuard** | Blocks AI editing (img2img) | Adversarial perturbation on the VAE latent space |
| **Image Obfuscator** | Embeds ownership markers & degrades scrapeability | Invisible/visible watermarks, pixelation, blur |
| **Content Filter** | Screens text & images before they enter training data | PII redaction, toxicity detection, quality scoring, AI-detection via CLIP |

---

## 🚀 Quick Start

### 1. Clone & set up a virtual environment

```bash
git clone https://github.com/LuisMend12/hackuconn.git
cd hackuconn
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

Choose the requirements file that matches the tool you want to run:

```bash
# PhotoGuard (Stable Diffusion + Streamlit app)
pip install -r requirements-photoguard.txt

# Image Obfuscator (CLIP-based poisoning + watermarking)
pip install -r requirements-obfuscator.txt

# Content / image filter only
pip install -r requirements.txt
```

### 3. Run

```bash
# PhotoGuard app
streamlit run photoguard_app.py

# Image Obfuscator app
streamlit run app.py
```

---

## 📸 PhotoGuard — Adversarial Image Protection

Adds an **invisible adversarial perturbation** to an image so that Stable Diffusion's img2img pipeline cannot faithfully edit it.

### How it works

1. Upload an image (resized to 512 × 512).
2. A PGD (Projected Gradient Descent) attack maximizes the latent-space norm of the image through the Stable Diffusion VAE encoder.
3. The perturbed image looks nearly identical to the original but produces incoherent results when fed to AI editing tools.
4. You can **test the protection** in-app by running img2img on both the original and protected images side by side.

### Files

| File | Purpose |
|------|---------|
| `photoguard_app.py` | Streamlit UI — upload, protect, and test |
| `photoguard_model.py` | Loads & caches the Stable Diffusion img2img pipeline |
| `photoguard_attack.py` | PGD attack implementation |
| `photoguard_utils.py` | Image preprocessing & recovery helpers |

### Tunable parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Epsilon | 0.06 | Perturbation bound — higher = stronger protection, more visible |
| Step size | 0.02 | PGD gradient step size |
| Iterations | 1000 | Number of PGD steps — more = stronger |

> **Note:** A CUDA GPU is strongly recommended. CPU mode works but is significantly slower.

---

## 🎨 Image Obfuscator — Watermarking & Degradation

A Streamlit app styled like an Instagram dark-mode feed. Applies safe, reversible transformations to discourage AI training on your images.

### Features

- **Invisible watermark** — LSB steganographic marker embedded in pixel data (preserved in PNG, lost in JPEG)
- **Visible watermark** — configurable text overlay with adjustable opacity and size
- **Pixelation** — downsamples and re-upsamples to degrade fine detail
- **Gaussian blur** — softens the image to reduce training signal
- **Download** — export as PNG (recommended for invisible watermark) or JPEG

### Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI with Instagram dark-mode design |
| `image_obfuscator.py` | Core API — `obfuscate_image()`, LSB embed/extract, visible watermark |

---

## 🔍 Content Filter — Text & Image Screening

A rule-based pipeline that labels training data as **KEEP**, **QUARANTINE**, or **DROP** with explainable findings.

### Text filter (`filter.py`)

Detects and redacts:
- **PII** — emails, phone numbers, SSNs, credit card numbers
- **Self-harm / violence / sexual content** — phrase-based detection
- **Prompt injection** — common jailbreak patterns
- **Low-quality text** — very short or high-entropy strings

```bash
python filter.py --in data.jsonl --out labeled.jsonl --write_cleaned
```

### Image filter (`filter_images.py`)

Screens images for quality and AI origin:
- Resolution, aspect ratio, blur, luminance checks
- Perceptual hash–based duplicate detection
- **CLIP-based AI-generated image detection** (estimates probability an image is AI-made)

```bash
python filter_images.py --dir images/ --report report.txt
```

---

## 📁 Project Structure

```
hackuconn/
├── photoguard_app.py          # PhotoGuard Streamlit app
├── photoguard_model.py        # Stable Diffusion pipeline loader
├── photoguard_attack.py       # PGD adversarial attack
├── photoguard_utils.py        # Image pre/post-processing
├── app.py                     # Image Obfuscator Streamlit app
├── image_obfuscator.py        # Watermark, pixelation, blur logic
├── filter.py                  # Text content filter (PII, toxicity)
├── filter_images.py           # Image quality + AI detection filter
├── index.html                 # Simple image upload page
├── data.jsonl                 # Sample text data for filter demo
├── requirements.txt           # Base dependencies
├── requirements-photoguard.txt
└── requirements-obfuscator.txt
```

---

## ⚙️ Requirements

- **Python 3.10+**
- **CUDA GPU** recommended for PhotoGuard and the image obfuscator's CLIP model
- ~5 GB disk for Stable Diffusion v1.5 weights (downloaded automatically on first run)

---

## 📄 License

Built at [HackUConn](https://www.hackuconn.com/). See repository for license details.
