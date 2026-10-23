#!/usr/bin/env bash
# Native Apple Silicon setup. No CUDA, ROCm, Triton, or system Python changes.
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  printf '%s\n' 'This installer requires an Apple Silicon Mac.' >&2
  exit 1
fi
if ! command -v uv >/dev/null; then
  printf '%s\n' 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/ and rerun bash setup_mac.sh.' >&2
  exit 1
fi
[ -d .venv ] || uv venv --python 3.13 .venv
bash setup_mac_frontend.sh .venv/bin/python
uv pip install --python .venv/bin/python -r requirements-mac.txt
if [ "${MING_SKIP_MODEL:-0}" != 1 ]; then
  # 8.8 GB total, including the Thinker, Talker, VAE, and speaker encoder.
  .venv/bin/hf download mlx-community/Ming-3.0-audio-tiny-4bit \
    --revision af0238886184f15fb81e28858df2a3bf43983145 \
    --local-dir "${MING_MODEL_DIR:-model-mlx}"
  .venv/bin/hf download pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx --local-dir model_extra
  # Only CPU frontend code/data, never the upstream CUDA model or inference package.
  for pattern in 'talker/front/*.py' 'talker/tts_prompt.py' 'talker/talker_tn/*' 'talker/data/*'; do
    .venv/bin/hf download inclusionAI/Ming-3.0-audio-tiny \
      --revision 3c9578f641b6745abf2d1239dc8419726f37c0e2 \
      --include "$pattern" --exclude '*/__pycache__/*' \
      --local-dir "${MING_FRONTEND_DIR:-model-frontend}"
  done
fi
printf '%s\n' 'Mac environment ready. Start with bash run.sh, then open http://localhost:8000.'
