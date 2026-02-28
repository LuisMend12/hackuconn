---

````markdown
# 🖼️ Image Dataset Filter for AI Training

A rule-based, explainable image filtering pipeline designed to clean and prepare image datasets for AI/ML training.

This tool helps you automatically detect and remove:

- ❌ Corrupt / unreadable images
- 📏 Images that are too small
- 🌫️ Very blurry images
- 🌑 Extremely dark images
- 🌕 Overexposed / too bright images
- 📐 Extreme aspect ratios
- 🔁 Near-duplicate images (perceptual hash based)

It outputs a structured JSONL report and can optionally organize images into:

- `KEEP`
- `QUARANTINE`
- `DROP`

---

# 🚀 Why Use This?

When training vision models (ViT, CNNs, object detection, etc.), poor-quality images:

- Reduce accuracy
- Increase noise
- Slow convergence
- Waste compute budget

This tool gives you **transparent filtering decisions** before training.

---

# 📦 Installation

Install dependencies:

```bash
pip install opencv-python pillow tqdm numpy
````

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

# 🛠️ How To Run

### Basic Usage

```bash
python3 image_filter.py \
  --in_dir ./images \
  --out_jsonl report.jsonl
```

---

### With Automatic Sorting

```bash
python3 image_filter.py \
  --in_dir ./images \
  --out_jsonl report.jsonl \
  --copy_keep_dir keep \
  --copy_quarantine_dir quarantine \
  --copy_drop_dir drop
```

This will:

* Copy good images into `keep/`
* Copy questionable images into `quarantine/`
* Copy bad images into `drop/`

---

# 📊 What The Filter Checks

## 1️⃣ Corrupt Images

* Uses PIL verification
* Automatically dropped

---

## 2️⃣ Minimum Resolution

Default:

* `min_width = 256`
* `min_height = 256`

Too small → QUARANTINE or DROP

Override:

```bash
--min_width 512 --min_height 512
```

---

## 3️⃣ Blur Detection

Uses **Variance of Laplacian**:

* Very blurry → DROP
* Somewhat blurry → QUARANTINE

Override:

```bash
--blur_drop 25
--blur_quarantine 75
```

---

## 4️⃣ Brightness Detection

Mean grayscale intensity (0–255):

* Too dark → DROP
* Slightly dark → QUARANTINE
* Too bright → DROP

---

## 5️⃣ Aspect Ratio Outliers

Default acceptable range:

```
0.25 ≤ width/height ≤ 4.0
```

Prevents extreme panoramas or tall crops.

---

## 6️⃣ Duplicate Detection

Uses 64-bit perceptual hash (pHash).

Images with small Hamming distance are treated as duplicates.

Override threshold:

```bash
--dup_hamming 4
```

Lower = stricter duplicate detection.

---

# 📄 Output Format (JSONL)

Each line of `report.jsonl` looks like:

```json
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

| Risk Level | Decision   |
| ---------- | ---------- |
| < 0.35     | KEEP       |
| 0.35–0.85  | QUARANTINE |
| ≥ 0.85     | DROP       |

Duplicates are automatically dropped (keeps first instance).

---

# 🎯 Example Use Cases

### Vision Transformer (ViT) Training

Remove blurry/noisy frames before fine-tuning.

### Dashcam / Roadway Datasets

Filter:

* Overexposed glare frames
* Night-only frames
* Motion-blurred frames

### Object Detection

Remove:

* Blank images
* Low-detail frames
* Corrupt captures

---

# ⚙️ Customization

You can modify thresholds in `FilterConfig` inside the script:

```python
@dataclass
class FilterConfig:
    min_width: int = 256
    min_height: int = 256
    blur_drop_threshold: float = 30.0
    ...
```

---

# 📈 Scaling to Large Datasets

For datasets >100k images:

* Replace linear duplicate scan with BK-tree
* Store pHashes in SQLite or Redis
* Parallelize using multiprocessing

---

# 🛑 Important Notes

This is a **quality filter**, not a content moderation model.

It does NOT detect:

* NSFW content
* Faces
* Violence
* Logos
* Copyrighted material

For that, integrate:

* CLIP classifier
* NSFW detector
* YOLO object detector
* Face detection

---

# 🔬 Advanced Extensions

Possible improvements:

* Entropy-based low-information detection
* Edge density thresholding
* Histogram clipping detection
* Motion blur estimation
* Sky/road-only frame detection
* Domain-specific heuristics

---

# 👨‍💻 Author

Built for AI dataset cleaning and model training pipelines.

---

# 📜 License

MIT License — use freely and modify.

```

---

If you'd like, I can also generate:

- A **more research-style README (for GitHub portfolio)**
- A **hackathon-ready README**
- A **transportation dataset–specific README**
- Or a **model-evaluation-integrated version (quality score + training cost analysis)**
```
