"""Native MLX Talker, retaining upstream voice anchors and cloning files."""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx_audio.sts.models.ming_v3_audio.talker import Talker as NativeTalker
from mlx_audio.tts.models.bailingmm.bailingmm import ming3_instruction
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
        self.steps = int(os.environ.get("MING_CFM_STEPS", "10"))
        self.lock = threading.Lock()
        self.anchors = {}
        for path in sorted(VoicePolicy.ANCHOR_DIR.glob("*.pt")):
            anchor = torch.load(path, map_location="cpu", weights_only=True)
            self.anchors[path.stem] = {
                "text": anchor["text"],
                "latents": mx.array(anchor["latents"].float().numpy()),
                "instruction": anchor.get("instruction"),
            }
        log(f"MLX Talker loaded in {time.time() - started:.1f}s; sentence-fragment audio")

    @property
    def designed_voices(self) -> list[str]:
        return [name for name, anchor in self.anchors.items() if anchor["instruction"]]

    @on_metal
    def speaker_from_audio(self, samples: np.ndarray, sample_rate: int) -> torch.Tensor:
        with self.lock:
            if self.t._speaker_encoder is None:
                self.t._speaker_encoder = SpeakerEncoder(self.t._speaker_path)
            resampled = resample_audio(samples, sample_rate, self.SR)
            embedding = self.t._speaker_encoder(mx.array(resampled), self.SR)
            return torch.from_numpy(np.array(embedding, dtype=np.float32))

    def synth_stream(self, text: str, voice: str | torch.Tensor = "US_Stella",
                     cancel: threading.Event | None = None,
                     max_steps: int = 750) -> Iterator[np.ndarray]:
        audio = self._synthesize(text, voice, cancel, max_steps)
        if audio is not None:
            yield audio

    @on_metal
    def _synthesize(self, text: str, voice: str | torch.Tensor,
                    cancel: threading.Event | None, max_steps: int) -> np.ndarray | None:
        cancel = cancel or threading.Event()
        if cancel.is_set() or not text.strip():
            return
        with self.lock:
            cloned = isinstance(voice, torch.Tensor)
            anchor = None if cloned else self.anchors.get(voice)
            instruction = None if cloned else (
                anchor["instruction"] if anchor and anchor["instruction"] else ming3_instruction(ip=voice))
            speaker = mx.array(voice.float().numpy()) if cloned else None
            speaker = self.t._speaker_embeddings(speaker, not cloned)
            prompt = "Please generate speech based on the following description.\nTarget language: English.\n"
            latents = []
            for latent, last in self.t.sample(
                prompt, text, instruction=instruction,
                prompt_text=anchor["text"] if anchor else None,
                prompt_latent=anchor["latents"] if anchor else None,
                spk_emb=speaker, max_decode_steps=max_steps, flow_steps=self.steps,
            ):
                if cancel.is_set():
                    return
                mx.eval(latent)
                latents.append(latent)
                if last:
                    break
            if latents:
                audio, _, _ = self.t.audio.decode(mx.concatenate(latents, axis=1),
                                                  use_cache=False, last_chunk=True)
                mx.eval(audio)
                if not cancel.is_set():
                    return np.array(audio[0], dtype=np.float32).reshape(-1)
        return None

    def synthesize(self, text: str, voice: str | torch.Tensor = "US_Stella") -> np.ndarray:
        parts = list(self.synth_stream(text, voice))
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)
