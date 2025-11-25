uv pip install --force-reinstall \
  torch==2.7.1+cu128 \
  torchaudio==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  fairseq2n \
  --index-url https://download.pytorch.org/whl/cu128

uv pip install --force-reinstall torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

uv pip install omnilingual-asr --index-url https://download.pytorch.org/whl/cu128


uv pip uninstall torch torchvision torchaudio -y
uv pip install --force-reinstall torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128

uv pip install --force-reinstall fairseq2\
  --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.9.0/cu128


uv pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128



uv pip install --force-reinstall torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128

uv pip install fairseq2 datasets omnilingual-asr soundfile pandas jiwer huggingface-hub

