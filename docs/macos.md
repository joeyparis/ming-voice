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
- Text tokens stream as they are generated. Speech now emits PCM before a text
  fragment finishes synthesizing, using cached incremental MLX audio decoding.
  The decoder primes three latents, releases audio with one lookahead latent,
  then decodes three-latent batches. Final chunks flush the decoder; cancellation
  closes the generator and releases its request state on the Metal thread.
- The server compiles the flow sampler by default, including its random state.
  Set `MING_GRAPHS=0` to disable compilation. This is native MLX execution, not
  the upstream HIP/CUDA graph implementation.
- A 400 ms browser playback buffer worked well in local streaming checks.
  Longer replies can still exhaust that buffer when generation falls behind
  playback, particularly under load. First PCM and first audible playback are
  different measurements.
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
uv run --python .venv/bin/python --with pytest pytest dev/test_mlx_talker.py dev/test_mlx.py
```

The initial fragment-buffered port was exercised on a 64 GB Apple M5 Pro. Startup used about
8.2 GiB active MLX memory. A short streamed chat produced its first text token
in 160 ms and its first spoken audio in about 4.2 seconds. Synthesizing a
three-second sentence took about 4.4 seconds. These are individual measurements,
not latency guarantees.

The incremental implementation passes all 17 checks above, including decoder
continuity, final flushes, step limits, cancellation, Metal thread ownership,
random-state compilation, and the existing cache/rollback checks.
With the full model on the same Mac, three warm runs per prompt at ten flow
steps gave these median times to first PCM:

| Prompt | Buffered decoder | Incremental decoder |
| --- | --- | --- |
| Short | 1.926 s | 0.316 s |
| Longer | 2.946 s | 0.330 s |

These speech-only timings exclude language-model generation and browser
buffering. Longer streamed synthesis remained close to real time, with a median
3.359 s to generate 3.2 s of audio, so uninterrupted playback is not guaranteed.
Built-in and cloned voices produced finite audio, and HTTP streaming recovered
after cancellation. Two real browser replies each delivered 16 PCM chunks
without a scheduling gap at a 400 ms playback buffer. First scheduled sound was
2.67 s after the first typed request and 2.02 s on its follow-up, including text
generation and buffering. Captured browser PCM was also transcribed successfully.

During the initial port validation, upstream offline tool checks passed.
API checks exercised speech synthesis, transcription, spoken questions, streamed
audio, and isolated chat history, including returning to a live conversation
after API requests. Voice cloning and playback of the clone also succeeded.
That earlier upstream API suite finished 13/14: one assertion expects the word
`five`; the four-bit model instead counts using digits. That assertion is
unchanged; the full API suite was not rerun for incremental decoding. Subsequent
local listening was reported working. Physical microphone pickup and speaker
echo behavior remain untested; digital playback and recorded-audio round trips
do not establish those.
