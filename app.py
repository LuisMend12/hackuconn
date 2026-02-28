import os
import shlex
import subprocess
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Nightshade Frontend", layout="wide")

st.title("Nightshade (research code) — simple frontend")

st.markdown(
    "This UI wraps the repo scripts. "
    "You still need data in the expected pickle format: each file contains keys `img` and `text`."
)

# --- Inputs ---
data_dir = st.text_input("Input data directory (pickle files)", value="data/")
concept = st.text_input("Source concept (e.g., dog)", value="dog")
num = st.number_input("Number of candidates", min_value=1, max_value=1000, value=100, step=1)

selected_dir = st.text_input("Selected output directory (from step 1)", value="selected_data/")
target = st.text_input("Target concept (e.g., cat)", value="cat")
outdir = st.text_input("Final output directory (poisoned images)", value="output/")

st.divider()
st.subheader("Run")

log_box = st.empty()

def run_cmd(cmd: str):
    log_box.code(f"$ {cmd}\n", language="bash")
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    for line in proc.stdout:
        output_lines.append(line)
        log_box.code(f"$ {cmd}\n\n{''.join(output_lines)}", language="bash")
    return proc.wait()

col1, col2 = st.columns(2)

with col1:
    if st.button("Step 1: Select candidates"):
        # From README: python3 data_extraction.py --directory data/ --concept dog --num 100
        cmd = f"python3 data_extraction.py --directory {shlex.quote(data_dir)} --concept {shlex.quote(concept)} --num {int(num)}"
        rc = run_cmd(cmd)
        st.success(f"Step 1 finished (exit code {rc}). Output should be in: {selected_dir} (or whatever the script uses).")

with col2:
    if st.button("Step 2: Generate Nightshade"):
        # README shows data_extraction.py here too, but repo includes gen_poison.py.
        # Try gen_poison.py first; if your repo actually wants a different script, swap it here.
        cmd = f"python3 gen_poison.py --directory {shlex.quote(selected_dir)} --target_name {shlex.quote(target)} --outdir {shlex.quote(outdir)}"
        rc = run_cmd(cmd)
        st.success(f"Step 2 finished (exit code {rc}). Output folder: {outdir}")

st.divider()
st.subheader("Results")
if Path(outdir).exists():
    st.write("Output files:")
    for p in sorted(Path(outdir).glob("*"))[:50]:
        st.write(str(p))
else:
    st.info("No output yet. Run Step 2.")