"""
Streamlit app: upload an image, protect it with PhotoGuard, and optionally test that editing is blocked.
"""
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T

from photoguard_utils import preprocess
from photoguard_model import get_pipe
from photoguard_attack import pgd

to_pil = T.ToPILImage()

st.set_page_config(page_title="PhotoGuard", page_icon="🛡️", layout="centered")
st.title("🛡️ PhotoGuard")
st.caption("Protect images from AI editing: upload an image to add an invisible adversarial perturbation so tools like Stable Diffusion can't faithfully edit it.")

uploaded = st.file_uploader("Upload an image to protect", type=["png", "jpg", "jpeg"], help="Image will be resized to 512×512 for processing.")

with st.expander("Attack parameters (advanced)", expanded=False):
    eps = st.slider("Epsilon (perturbation bound)", 0.02, 0.15, 0.06, 0.01, help="Higher = stronger protection, more visible change.")
    step_size = st.slider("Step size", 0.005, 0.05, 0.02, 0.005)
    iters = st.slider("Iterations", 100, 2000, 1000, 100, help="More iterations = stronger protection.")

if uploaded is not None:
    init_image = Image.open(uploaded).convert("RGB")
    init_image = T.functional.center_crop(T.functional.resize(init_image, 512), [512, 512])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(init_image, use_container_width=True)

    if st.button("Protect image with PhotoGuard", type="primary"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            st.warning("Running on CPU. Protection will be slow; GPU is recommended.")
        with st.spinner("Loading Stable Diffusion and running attack…"):
            pipe = get_pipe(device)
            with torch.autocast(device):
                X = preprocess(init_image).to(device)
                if device == "cuda":
                    X = X.half()
                adv_X = pgd(
                    X,
                    model=pipe.vae.encode,
                    clamp_min=-1,
                    clamp_max=1,
                    eps=eps,
                    step_size=step_size,
                    iters=iters,
                    device=device,
                )
                adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
            adv_image = to_pil(adv_X[0].float().cpu()).convert("RGB")

        with col2:
            st.subheader("Protected image")
            st.image(adv_image, use_container_width=True)
        st.success("Done. The protected image looks almost identical but will resist AI editing.")

        st.divider()
        st.subheader("Test: try to edit with AI")
        prompt = st.text_input("Edit prompt (e.g. 'dog under heavy rain')", value="dog under heavy rain and muddy ground real")
        if st.button("Run img2img on both images"):
            with st.spinner("Running Stable Diffusion on original and protected image…"):
                seed = 9222
                strength = 0.5
                guidance = 7.5
                steps = 50
                with torch.autocast(device):
                    torch.manual_seed(seed)
                    gen_nat = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance, num_inference_steps=steps).images[0]
                    torch.manual_seed(seed)
                    gen_adv = pipe(prompt=prompt, image=adv_image, strength=strength, guidance_scale=guidance, num_inference_steps=steps).images[0]
            a, b, c = st.columns(3)
            with a:
                st.caption("Original image")
                st.image(init_image, use_container_width=True)
            with b:
                st.caption("Generated from original (edit works)")
                st.image(gen_nat, use_container_width=True)
            with c:
                st.caption("Generated from protected (edit blocked)")
                st.image(gen_adv, use_container_width=True)
            st.info("The protected image should produce a very different or incoherent result, so the edit is effectively blocked.")
