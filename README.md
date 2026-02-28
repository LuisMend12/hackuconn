<p align="center">
  <img src="logo.png" alt="MadeByMe Logo" width="200"/>
</p>

<h1 align="center">MadeByMe</h1>

<p align="center">
  Protect your images from unauthorized AI editing.
</p>

---

## What is MadeByMe?

MadeByMe adds an invisible layer of protection to your images so that AI tools like Stable Diffusion **cannot faithfully edit them**. The protected image looks identical to the original to the human eye, but produces broken or incoherent results when an AI tries to modify it.

---

## How It Works

1. **Upload** your image.
2. **Protect** — MadeByMe runs an adversarial attack on Stable Diffusion's image encoder, making tiny pixel-level changes that are invisible to humans but confuse AI.
3. **Test** — Try editing both the original and protected images with an AI prompt to see the difference.

---

## Parameters

| Parameter | Default | What it does |
|-----------|---------|--------------|
| **Epsilon** | 0.06 | Maximum amount each pixel can change. Higher = stronger protection, but the image may look slightly different. |
| **Step Size** | 0.02 | How much the protection adjusts each round. |
| **Iterations** | 1000 | How many rounds of adjustments. More = stronger protection, takes longer. |

---

## Getting Started

### Requirements

- Python 3.10+
- CUDA GPU recommended (~5 GB for Stable Diffusion weights, downloaded automatically)

### Setup

```bash
git clone https://github.com/LuisMend12/hackuconn.git
cd hackuconn
python -m venv venv
source venv/bin/activate
pip install -r requirements-photoguard.txt
```

### Run

```bash
streamlit run photoguard_app.py
```

Then open **http://localhost:8501** in your browser.

---

## Project Structure

```
├── photoguard_app.py        # Streamlit UI
├── photoguard_model.py      # Stable Diffusion pipeline loader
├── photoguard_attack.py     # PGD adversarial attack
├── photoguard_utils.py      # Image preprocessing utilities
├── requirements-photoguard.txt
└── README.md
```

---

## Built at [HackUConn](https://www.hackuconn.com/)
---

## Acknowledgements

This project builds upon ideas and code from **PhotoGuard** by MadryLab.

PhotoGuard Repository:
https://github.com/MadryLab/photoguard

If you use this project, please also cite the original PhotoGuard work.

## 📄 License

Built at [HackUConn](https://www.hackuconn.com/). See repository for license details.
