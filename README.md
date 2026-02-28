# 🖼️ Image Dataset Quality Filter for AI Training

A rule-based, explainable image filtering pipeline designed to clean and prepare image datasets for AI / ML training workflows.

This tool automatically analyzes images and classifies them into:

* ✅ KEEP
* ⚠️ QUARANTINE
* ❌ DROP

It generates a structured JSONL report and can optionally reorganize images into separate folders.

---

# 🚀 Why This Exists

When training computer vision models (ViT, CNNs, object detection, segmentation, etc.), low-quality images can:

* Decrease model accuracy
* Increase noise in training
* Slow convergence
* Waste GPU compute
* Introduce duplicate bias

This filter gives you **transparent, explainable quality control** before model training.

---

# 🔍 What It Detects

The filter evaluates each image using measurable signals:

## 1️⃣ Corrupt or Unreadable Files

* Uses PIL verification
* Automatically dropped

---

## 2️⃣ Resolution Checks

Default minimum:

* Width ≥ 256 px
* Height ≥ 256 px

Too small → QUARANTINE or DROP

---

## 3️⃣ Blur Detection

Uses **Variance of Laplacian** (standard blur metric):

* Very blurry → DROP
* Moderately blurry → QUARANTINE
* Sharp → KEEP

---

## 4️⃣ Brightness Analysis

Based on mean grayscale intensity (0–255):

* Too dark → DROP
* Slightly dark → QUARANTINE
* Overexposed → DROP

---

## 5️⃣ Aspect Ratio Outliers

Default acceptable range:

0.25 ≤ width/height ≤ 4.0

Prevents:

* Extreme panoramas
* Tall cropped artifacts
* Broken frame extractions

---

## 6️⃣ Duplicate Detection

Uses 64-bit perceptual hash (pHash).

Images with small Hamming distance are treated as duplicates.

* Keeps first instance
* Drops near-identical copies

---

# 📦 Installation

Install dependencies:

```
pip install opencv-python pillow tqdm numpy
```

---

# 📂 Project Structure

```
project/
│
├── image_filter.py
├── README.md
├── images/                # your dataset
├── report.jsonl           # generated report
├── keep/                  # optional
├── quarantine/            # optional
└── drop/                  # optional
```

---

# 🛠️ Usage

## Basic Usage

```
python3 image_filter.py \
  --in_dir ./images \
  --out_jsonl report.jsonl
```

This:

* Recursively scans all images
* Analyzes them
* Produces `report.jsonl`

---

## With Automatic Sorting

```
python3 image_filter.py \
  --in_dir ./images \
  --out_jsonl report.jsonl \
  --copy_keep_dir keep \
  --copy_quarantine_dir quarantine \
  --copy_drop_dir drop
```

This:

* Copies good images into `keep/`
* Moves questionable images into `quarantine/`
* Moves bad images into `drop/`

---

# ⚙️ Custom Thresholds

Override defaults from the CLI:

### Resolution

```
--min_width 512 --min_height 512
```

---

### Blur Sensitivity

```
--blur_drop 25
--blur_quarantine 75
```

Lower values = stricter blur detection.

---

### Duplicate Sensitivity

```
--dup_hamming 4
```

Lower values = stricter duplicate detection.

---

# 📊 Output Format (JSONL)

Each line in `report.jsonl` contains:

```
{
  "path": "images/img_001.jpg",
  "metrics": {
    "width": 1920,
    "height": 1080,
    "aspect_ratio": 1.777,
    "blur_var_lap": 112.4,
    "mean_luma": 98.2,
    "phash": 123456789
  },
  "decision": {
    "decision": "KEEP",
    "reasons": [],
    "risk": 0.0
  },
  "error": null
}
```

---

# 🧠 Decision Logic

| Risk Score | Decision   |
| ---------- | ---------- |
| < 0.35     | KEEP       |
| 0.35–0.85  | QUARANTINE |
| ≥ 0.85     | DROP       |

Duplicates are dropped automatically (first instance kept).

---

# 🎯 Example Use Cases

### Vision Transformer (ViT) Training

Remove blurry or dark frames before fine-tuning.

### Dashcam / Roadway Dataset Cleaning

Filter:

* Overexposed glare frames
* Night-only frames
* Motion-blurred frames
* Corrupt camera captures

### Object Detection

Remove:

* Blank frames
* Duplicate extractions
* Cropped artifacts

---

# 📈 Scaling to Large Datasets

For datasets larger than 100k images:

* Replace linear duplicate search with BK-tree
* Store hashes in SQLite
* Parallelize with multiprocessing
* Batch hash comparisons

---

# 🛑 Important Limitations

This is a **quality filter**, not a content moderation system.

It does NOT detect:

* NSFW content
* Violence
* Faces
* Logos
* Copyrighted material
* Illegal content

To extend it, integrate:

* CLIP classifier
* NSFW model
* YOLO object detector
* Face detection

---

# 🔬 Advanced Extensions

Possible future improvements:

* Edge-density filtering (low-information frames)
* Histogram clipping detection
* Motion blur estimation
* Sky/road-only heuristics
* Domain-specific detection rules
* Model-based quality scoring

---

# 📜 License

MIT License — free to use and modify.

---

