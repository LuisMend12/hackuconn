const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file');
const previewWrap = document.getElementById('previewWrap');
const preview = document.getElementById('preview');
const checkmarkBtn = document.getElementById('checkmarkBtn');

function showImage(file) {
  if (!file || !file.type.startsWith('image/')) return;

  const url = URL.createObjectURL(file);
  preview.src = url;
  previewWrap.style.display = 'block';
  checkmarkBtn.style.display = 'inline-flex';
}

dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) showImage(file);
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) showImage(file);
});

checkmarkBtn.addEventListener('click', () => {
  // does nothing for now
});
