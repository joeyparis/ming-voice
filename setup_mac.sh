#!/usr/bin/env bash
# Native Apple Silicon setup. No CUDA, ROCm, Triton, or system Python changes.
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  printf '%s\n' 'This installer requires an Apple Silicon Mac.' >&2
  exit 1
fi
command -v uv >/dev/null
[ -d .venv ] || uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-mac.txt
if [ "${MING_SKIP_MODEL:-0}" != 1 ]; then
  # 8.8 GB total, including the Thinker, Talker, VAE, and speaker encoder.
  .venv/bin/hf download mlx-community/Ming-3.0-audio-tiny-4bit \
    --revision af0238886184f15fb81e28858df2a3bf43983145 \
    --local-dir "${MING_MODEL_DIR:-model-mlx}"
  .venv/bin/hf download pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx --local-dir model_extra
fi
printf '%s\n' 'Mac environment ready. Start with bash run.sh, then open http://localhost:8000.'
