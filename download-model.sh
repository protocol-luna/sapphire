#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/model"

echo "[sapphire] downloading ONNX model from HuggingFace..."
pip install -q huggingface-hub 2>/dev/null

python3 -c "
from huggingface_hub import hf_hub_download
import os

repo = 'addyo07/distilbert-query-classifier'
dest = '$DIR/model'

files = [
    'model/onnx/model_quantized.onnx',
    'model/pytorch/tokenizer.json',
    'model/pytorch/tokenizer_config.json',
    'model/pytorch/config.json',
]

for f in files:
    path = hf_hub_download(repo, f)
    os.system(f'cp \"{path}\" \"{dest}/\"')
    print(f'  {os.path.basename(f)}')
"

echo "[sapphire] model files ready in $DIR/model"
