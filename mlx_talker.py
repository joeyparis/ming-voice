"""Native MLX Talker, retaining upstream voice anchors and cloning files."""
from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Generator
from contextlib import ExitStack
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx_audio.sts.models.ming_v3_audio.talker import Talker as NativeTalker
from mlx_audio.tts.models.bailingmm.bailingmm import (
    MING3_MAX_FRAGMENT_LENGTH,
    _split_fragments,
    ming3_instruction,
)
from mlx_audio.tts.models.lingv3_tts.speaker import SpeakerEncoder
from mlx_audio.utils import resample_audio

from engine import MODEL_DIR, Talker as VoicePolicy, log
from metal_worker import on_metal


class Talker:
    SR = 44100

    @on_metal
    def __init__(self, model_dir: Path = MODEL_DIR, use_graphs: bool = False) -> None:
        started = time.time()
        self.t = NativeTalker.from_pretrained(model_dir / "talker")
        if use_graphs:
            self.t.flowloss.sample = mx.compile(
                self.t.flowloss.sample, inputs=mx.random.state, outputs=mx.random.state,
            )
        self.steps = int(os.environ.get("MING_CFM_STEPS", "10"))
        self.lock = threading.Lock()
        self.anchors = {}
        for path in sorted(VoicePolicy.ANCHOR_DIR.glob("*.pt")):
            anchor = torch.load(path, map_location="cpu", weights_only=True)
            if not anchor.get("instruction") and os.environ.get("MING_VOICE_ANCHORS", "1") != "1":
                continue
            self.anchors[path.stem] = {
                "text": anchor["text"],
                "latents": mx.array(anchor["latents"].float().numpy()),
                "instruction": anchor.get("instruction"),
            }
        log(f"MLX Talker loaded in {time.time() - started:.1f}s; incremental audio"
            f"{' (compiled flow)' if use_graphs else ''}")

    @property
    def designed_voices(self) -> list[str]:
        return [name for name, anchor in self.anchors.items() if anchor["instruction"]]

    def speaker_from_audio(self, samples: np.ndarray, sample_rate: int) -> torch.Tensor:
        with self.lock:
            return self._speaker_from_audio(samples, sample_rate)

    @on_metal
    def _speaker_from_audio(self, samples: np.ndarray, sample_rate: int) -> torch.Tensor:
        if self.t._speaker_encoder is None:
            self.t._speaker_encoder = SpeakerEncoder(self.t._speaker_path)
        resampled = resample_audio(samples, sample_rate, self.SR)
        embedding = self.t._speaker_encoder(mx.array(resampled), self.SR)
        return torch.from_numpy(np.array(embedding, dtype=np.float32))

    def synth_stream(self, text: str, voice: str | torch.Tensor = "US_Stella",
                     cancel: threading.Event | None = None,
                     max_steps: int = 750) -> Generator[np.ndarray, None, None]:
        cancel = cancel or threading.Event()
        # Wait for another voice request outside the worker, so it can keep advancing.
        with self.lock:
            for fragment in _split_fragments(text, MING3_MAX_FRAGMENT_LENGTH):
                if cancel.is_set():
                    return
                with ExitStack() as cleanup:
                    stream = self._synthesize(fragment, voice, cancel, max_steps)
                    cleanup.callback(on_metal(stream.close))

                    @on_metal
                    def advance() -> np.ndarray | None:
                        return next(stream, None)

                    while not cancel.is_set():
                        audio = advance()
                        if audio is None:
                            break
                        if not cancel.is_set():
                            yield audio

    def _synthesize(self, text: str, voice: str | torch.Tensor,
                    cancel: threading.Event, max_steps: int) -> Generator[np.ndarray, None, None]:
        """Advance only on the Metal worker; decoder state belongs to this utterance."""
        if cancel.is_set() or not text.strip():
            return
        cloned = isinstance(voice, torch.Tensor)
        anchor = None if cloned else self.anchors.get(voice)
        instruction = None if cloned else (
            anchor["instruction"] if anchor and anchor["instruction"] else ming3_instruction(ip=voice))
        if re.search(r"[\u4e00-\u9fff]", text):
            anchor = None  # Acoustic anchors are English clips; keep any voice-design instruction.
        speaker = mx.array(voice.float().numpy()) if cloned else None
        speaker = self.t._speaker_embeddings(speaker, not cloned)
        prompt = "Please generate speech based on the following description.\n"
        pending: list[mx.array] = []
        state, pkv, calls = (None, None, None), None, 0
        for step, (latent, last) in enumerate(self.t.sample(
            prompt, text, instruction=instruction,
            prompt_text=anchor["text"] if anchor else None,
            prompt_latent=anchor["latents"] if anchor else None,
            spk_emb=speaker, max_decode_steps=max_steps, flow_steps=self.steps,
        )):
            if cancel.is_set():
                return
            mx.eval(latent)
            pending.append(mx.array(latent))
            last = last or step + 1 >= max_steps
            # Prime three latents, then release them with one lookahead latent.
            if last or len(pending) >= (1 if calls == 1 else 3):
                audio, state, pkv = self.t.audio.decode(
                    mx.concatenate(pending, axis=1), past_key_values=pkv,
                    use_cache=True, stream_state=state, last_chunk=last,
                )
                audio = mx.array(audio)
                pending.clear()
                calls += 1
                mx.eval(audio)
                if audio.size and not cancel.is_set():
                    yield np.array(audio[0], dtype=np.float32).reshape(-1)
            if last:
                return

    def synthesize(self, text: str, voice: str | torch.Tensor = "US_Stella") -> np.ndarray:
        parts = list(self.synth_stream(text, voice))
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)
