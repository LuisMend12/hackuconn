import streamlit as st
from PIL import Image
import io

from image_obfuscator import obfuscate_image

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="HackUConn Obfuscator", page_icon="📷", layout="wide")

# -----------------------------
# Instagram DARK MODE CSS
# -----------------------------
st.markdown(
    """
    <style>
      /* --- IG Dark base --- */
      .stApp {
        background: #000000; /* Instagram dark */
        color: #f5f5f5;
      }

      /* Centered content / phone-ish width */
      section.main > div {
        max-width: 980px;
        margin: 0 auto;
        padding-top: 12px;
      }

      /* Card (IG post container vibe) */
      .ig-card {
        background: #0f0f0f;
        border: 1px solid #262626; /* IG border */
        border-radius: 16px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.55);
        padding: 14px 14px 10px 14px;
      }

      /* Top bar */
      .ig-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 10px 12px 10px;
      }
      .ig-logo {
        font-weight: 800;
        letter-spacing: -0.5px;
        font-size: 18px;
        color: #f5f5f5;
      }
      .ig-icons {
        display: flex;
        gap: 10px;
        opacity: 0.9;
        font-size: 18px;
        color: #f5f5f5;
      }
      .ig-divider {
        height: 1px;
        background: #262626;
        margin: 6px 0 12px 0;
      }

      /* Story-like chips */
      .chip-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 8px 0 8px 0;
      }
      .chip {
        border: 1px solid #262626;
        padding: 8px 12px;
        border-radius: 999px;
        background: #121212;
        font-size: 13px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.35);
        color: #f5f5f5;
      }

      /* Image frame */
      .img-frame {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #262626;
        background: #000;
      }

      /* Buttons (IG-like: primary blue, pill) */
      div.stButton > button {
        width: 100%;
        border-radius: 999px !important;
        padding: 0.78rem 1rem !important;
        font-weight: 800 !important;
        border: 0 !important;
        background: #0095f6 !important; /* IG blue */
        color: white !important;
      }
      div.stButton > button:hover {
        filter: brightness(0.95);
      }

      /* Download button: dark outlined */
      div.stDownloadButton > button {
        width: 100%;
        border-radius: 999px !important;
        padding: 0.78rem 1rem !important;
        font-weight: 800 !important;
        border: 1px solid #262626 !important;
        background: #121212 !important;
        color: #f5f5f5 !important;
      }
      div.stDownloadButton > button:hover {
        filter: brightness(1.06);
      }

      /* File uploader dark */
      [data-testid="stFileUploader"] section {
        background: #121212 !important;
        border: 1px dashed #262626 !important;
        border-radius: 14px !important;
        color: #f5f5f5 !important;
      }

      /* Inputs dark */
      input, textarea {
        background: #121212 !important;
        color: #f5f5f5 !important;
        border: 1px solid #262626 !important;
        border-radius: 12px !important;
      }

      /* Selectbox dark */
      [data-baseweb="select"] > div {
        background: #121212 !important;
        border: 1px solid #262626 !important;
        border-radius: 12px !important;
        color: #f5f5f5 !important;
      }

      /* Expanders */
      details {
        background: #0f0f0f !important;
        border: 1px solid #262626 !important;
        border-radius: 14px !important;
        padding: 6px 10px !important;
      }

      /* Alerts */
      [data-testid="stAlert"] {
        background: #121212 !important;
        border: 1px solid #262626 !important;
        border-radius: 14px !important;
        color: #f5f5f5 !important;
      }

      /* Reduce Streamlit chrome */
      header {visibility: hidden;}
      footer {visibility: hidden;}

      /* Labels */
      .section-title {
        font-size: 14px;
        font-weight: 800;
        margin: 8px 0 8px 0;
        opacity: 0.92;
        color: #f5f5f5;
      }
      .muted {
        font-size: 13px;
        color: #a8a8a8;
      }

      /* Make all normal text lighter */
      p, label, span, div {
        color: #f5f5f5 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Layout wrapper
# -----------------------------
st.markdown('<div class="ig-card">', unsafe_allow_html=True)

# Top bar
st.markdown(
    """
    <div class="ig-topbar">
      <div class="ig-logo">HackUConn • Obfuscator</div>
      <div class="ig-icons">♡ ⊕ ✉️</div>
    </div>
    <div class="ig-divider"></div>
    """,
    unsafe_allow_html=True,
)

# Upload
st.markdown('<div class="section-title">Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")

# Settings header
st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)

# Quick toggles
chipA, chipB, chipC, chipD = st.columns([1, 1, 1, 1])
with chipA:
    pixelate = st.toggle("Pixelate", value=False)
with chipB:
    blur = st.toggle("Blur", value=False)
with chipC:
    watermark_mode = st.selectbox("Watermark", ["invisible", "visible", "none"], index=0)
with chipD:
    fmt = st.selectbox("Download as", ["PNG (recommended)", "JPG"], index=0)

with st.expander("More controls", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        watermark_text = st.text_input("Watermark text / marker", value="DO NOT TRAIN")
        watermark_opacity = st.slider("Visible watermark opacity", 0, 255, 70, 5, help="Used only if Watermark=visible")
        watermark_scale = st.slider("Visible watermark size", 0.02, 0.15, 0.06, 0.01, help="Used only if Watermark=visible")
    with c2:
        pixelate_factor = st.slider("Pixelate factor", 2, 30, 10, 1, disabled=not pixelate)
        blur_radius = st.slider("Blur radius", 0.0, 6.0, 1.2, 0.1, disabled=not blur)

st.markdown(
    f"""
    <div class="chip-row">
      <div class="chip">📌 Watermark: <b>{watermark_mode}</b></div>
      <div class="chip">🧩 Pixelate: <b>{"On" if pixelate else "Off"}</b></div>
      <div class="chip">🌫️ Blur: <b>{"On" if blur else "Off"}</b></div>
    </div>
    <div class="muted">Tip: Use <b>invisible</b> to embed a marker without visible text. Save as <b>PNG</b> to preserve it.</div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="ig-divider"></div>', unsafe_allow_html=True)

# -----------------------------
# Feed-like image area
# -----------------------------
if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Original</div>', unsafe_allow_html=True)
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(original_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

        obfuscate_clicked = st.button("Obfuscate ✨")

        if obfuscate_clicked:
            output_image = obfuscate_image(
                original_image,
                watermark_text=watermark_text,
                watermark_mode=watermark_mode,
                watermark_opacity=watermark_opacity,
                watermark_scale=watermark_scale,
                pixelate=pixelate,
                pixelate_factor=pixelate_factor,
                blur=blur,
                blur_radius=blur_radius,
            )

            st.success("Done ✅")

            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(output_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            buffer = io.BytesIO()

            if fmt.startswith("PNG"):
                output_image.save(buffer, format="PNG")
                filename = "obfuscated.png"
                mime = "image/png"
            else:
                output_image.save(buffer, format="JPEG", quality=95)
                filename = "obfuscated.jpg"
                mime = "image/jpeg"

            buffer.seek(0)

            st.download_button(
                "Download ⬇️",
                data=buffer,
                file_name=filename,
                mime=mime,
            )
else:
    st.info("Upload an image to get started.")

st.markdown("</div>", unsafe_allow_html=True)