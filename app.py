import streamlit as st
from PIL import Image
import io

# Import your obfuscator function
from image_obfuscator import obfuscate_image

st.set_page_config(layout="wide")
st.title("HackUConn Image Obfuscator")

st.subheader("Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "webp"]
)

# ---- Controls ----
st.subheader("Settings")

colA, colB, colC = st.columns(3)

with colA:
    watermark_text = st.text_input("Watermark text", value="DO NOT TRAIN")

with colB:
    watermark_opacity = st.slider(
        "Watermark opacity",
        min_value=0,
        max_value=255,
        value=70,
        step=5
    )

with colC:
    watermark_scale = st.slider(
        "Watermark size",
        min_value=0.02,
        max_value=0.15,
        value=0.06,
        step=0.01
    )

col1, col2, col3 = st.columns(3)

with col1:
    pixelate = st.checkbox("Pixelate", value=False)

with col2:
    pixelate_factor = st.slider(
        "Pixelate factor",
        2,
        30,
        10,
        1,
        disabled=not pixelate
    )

with col3:
    blur = st.checkbox("Blur", value=False)

blur_radius = st.slider(
    "Blur radius",
    0.0,
    6.0,
    1.2,
    0.1,
    disabled=not blur
)

# ---- Processing ----
if uploaded_file is not None:

    original_image = Image.open(uploaded_file).convert("RGB")

    st.image(original_image, caption="Original Image", use_column_width=True)

    if st.button("Obfuscate Image"):

        output_image = obfuscate_image(
            original_image,
            watermark_text=watermark_text,
            watermark_opacity=watermark_opacity,
            watermark_scale=watermark_scale,
            pixelate=pixelate,
            pixelate_factor=pixelate_factor,
            blur=blur,
            blur_radius=blur_radius,
        )

        st.success("Obfuscation Complete ✅")

        st.image(output_image, caption="Obfuscated Image", use_column_width=True)

        # Prepare download
        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        buffer.seek(0)

        st.download_button(
            "Download Obfuscated Image",
            data=buffer,
            file_name="obfuscated.png",
            mime="image/png"
        )