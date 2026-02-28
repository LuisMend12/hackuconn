import streamlit as st
import subprocess
import shlex
from pathlib import Path

st.title("HackUConn Image Tools")

st.subheader("Run Image Obfuscator")

input_dir = st.text_input("Input Images Folder", value="images")
output_dir = st.text_input("Output Folder", value="output")

log_box = st.empty()

def run_command(cmd):
    log_box.code(f"$ {cmd}\n", language="bash")
    process = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    logs = ""
    for line in process.stdout:
        logs += line
        log_box.code(f"$ {cmd}\n\n{logs}", language="bash")

    return process.wait()

if st.button("Run Image Obfuscator"):
    cmd = f"python3 image_obfuscator.py --in_dir {input_dir} --out_dir {output_dir}"
    rc = run_command(cmd)
    st.success(f"Finished with exit code {rc}")

st.divider()

st.subheader("Run Image Quality Filter")

report_file = st.text_input("Report Output File", value="report.txt")

if st.button("Run Image Filter"):
    cmd = f"python3 filter_images.py --in_dir {input_dir} --out_txt {report_file}"
    rc = run_command(cmd)
    st.success(f"Filter completed with exit code {rc}")

st.divider()

st.subheader("Results")

if Path(output_dir).exists():
    st.write("Generated files:")
    for p in sorted(Path(output_dir).glob("*"))[:20]:
        st.write(str(p))

if Path(report_file).exists():
    st.download_button(
        "Download Report",
        data=open(report_file, "rb"),
        file_name=report_file,
    )