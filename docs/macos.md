# Apple Silicon Mac

This fork adds a native MLX backend for Ming Voice. The Thinker and Talker run
on the integrated Apple GPU using unified memory. PyTorch remains CPU-only for
voice activity detection, resampling, token transport, and saved voice files.
The Linux/Windows CUDA and ROCm backend remains available.

## Setup

Use an Apple Silicon Mac with at least 16 GB unified memory, Python 3.13
(managed by `uv`), and approximately 13 GB free disk total, including the
8.8 GB four-bit model. A 32 GB or larger Mac gives other applications
more room. Install `uv` first if it is not already available.

```sh
bash setup_mac.sh
bash run.sh
```

Open `http://localhost:8000`, press **Start conversation**, and allow microphone
access. To use another port, run `PORT=8008 bash run.sh`.
Stop the server with Ctrl-C to release its unified memory.

The MLX Audio source revision and model revision are pinned in the installer.
Weights live in `model-mlx/`, not the upstream `model/` directory. Set
`MING_MODEL_DIR` during both setup and startup to use a different location.
Do not use the Linux/Windows installer on a Mac.

## Behavior

- Existing Ming Voice conversation history, tool execution, speculative
  rollback, and Home Assistant commands use the same policy as upstream.
- Text tokens stream as they are generated. Speech is decoded per text
  fragment, rather than using the upstream HIP/CUDA graph path. Expect different
  latency from the Radeon benchmarks.
- Native model operations share one persistent Metal thread. Request threads
  must not evaluate MLX graphs themselves; MLX streams belong to their creator.
- Built-in voices, voice anchors (including Jarvis), and newly cloned voices
  use the native MLX Talker. GPU-specific developer scripts still target CUDA.
- `/api/health` reports `backend: "mlx"`. Memory values describe MLX allocations
  and its recommended unified-memory working-set budget, not dedicated VRAM.
- The Home Assistant connection is configured separately using the upstream
  settings. Starting the Mac port does not connect it to Home Assistant.

## Checks

Run the Mac regression checks with:

```sh
uv run --python .venv/bin/python --with pytest pytest dev/test_mlx.py
```

The initial port was exercised on a 64 GB Apple M5 Pro. Startup used about
8.2 GiB active MLX memory. A short streamed chat produced its first text token
in 160 ms and its first spoken audio in about 4.2 seconds. Synthesizing a
three-second sentence took about 4.4 seconds. These are individual measurements,
not latency guarantees.

The four real-model cache/rollback tests and upstream offline tool checks pass.
API checks exercised speech synthesis, transcription, spoken questions, streamed
audio, and isolated chat history, including returning to a live conversation
after API requests. Voice cloning and playback of the clone also succeeded.
The upstream API suite finishes 13/14: one assertion expects the word
`five`; the four-bit model instead counts using digits. That assertion is
unchanged. Physical microphone pickup and speaker echo behavior require a local
listening test; recorded-audio round trips do not establish those.
