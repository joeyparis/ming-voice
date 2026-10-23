"""Native Metal inference with Ming Voice's existing conversation/tool policy."""
from __future__ import annotations

import copy
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

import mlx.core as mx
import numpy as np
import torch
from mlx_audio.lm.models.cache import ArraysCache, KVCache
from mlx_audio.sts.models.ming_v3_audio import Model
from mlx_audio.sts.utils import load_model
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from engine import DEFAULT_SYSTEM_PROMPT, MODEL_DIR, Thinker as Conversation, log, pad_to_bucket
from metal_worker import on_metal

Cache = list[KVCache | ArraysCache]
Saved = tuple[int, Cache]


class Snapshot(TypedDict):
    saved: Saved
    pending: str | None
    n_hist: int
    epoch: int


class Turn(TypedDict, total=False):
    role: str
    user: str | None
    content: str
    interrupted: bool


class AudioFeatures(TypedDict, total=False):
    audio: mx.array


class Thinker:
    # Share conversation policy, not CUDA construction or cache implementation.
    reset = Conversation.reset
    _prefill_system = Conversation._prefill_system
    _rollover_text = Conversation._rollover_text
    _system_block = Conversation._system_block
    set_tools = Conversation.set_tools
    ctx_len = Conversation.ctx_len
    needs_compact = Conversation.needs_compact
    render_history = Conversation.render_history
    stream = Conversation.stream

    @on_metal
    def __init__(self, model_dir: Path = MODEL_DIR,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        started = time.time()
        model = load_model(model_dir, strict=True)
        if not isinstance(model, Model):
            raise TypeError(type(model).__name__)
        self.model = model
        self.tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
        self.system_prompt = system_prompt
        self.tools = None
        self._tools_sig = None
        self._sys_cache = {}
        self.max_ctx = int(os.environ.get("MING_MAX_CTX", "8192"))
        self.cache = self.model.make_cache()
        self.cur_len = 0
        self.epoch = 1
        self.pending: str | None = None
        self.history: list[Turn] = []
        self.rebuild = False
        log(f"MLX Thinker loaded in {time.time() - started:.1f}s; native Metal, quantized experts")

    @contextmanager
    def isolated(self) -> Iterator[Thinker]:
        saved = self._save_state()
        meta = (self.pending, list(self.history), self.system_prompt, self.epoch,
                self.rebuild, self.tools, self._tools_sig)
        try:
            yield self
        finally:
            self._load_state(saved)
            (self.pending, self.history, self.system_prompt, self.epoch,
             self.rebuild, self.tools, self._tools_sig) = meta

    @on_metal
    def _clear_state(self) -> None:
        self.cache = self.model.make_cache()

    @on_metal
    def _save_state(self) -> Saved:
        # Copy both recurrent state and KV offsets. MLX arrays are copy-on-write.
        cache = copy.deepcopy(self.cache)
        return self.cur_len, cache

    @on_metal
    def _load_state(self, saved: Saved) -> None:
        self.cur_len, cache = saved
        self.cache = copy.deepcopy(cache)

    def snapshot(self) -> Snapshot:
        return Snapshot(saved=self._save_state(), pending=self.pending,
                        n_hist=len(self.history), epoch=self.epoch)

    def restore(self, snap: Snapshot) -> None:
        history = self.history[:snap["n_hist"]]
        if snap["epoch"] != self.epoch:
            self.reset()
            self.history, self.rebuild = history, bool(history)
            return
        self._load_state(snap["saved"])
        self.pending = snap["pending"]
        self.history = history

    @on_metal
    def _audio_input(self, wav16k: np.ndarray) -> tuple[str, AudioFeatures]:
        features = self.model.prepare_audio(pad_to_bucket(wav16k))
        audio = self.model.encode_audio(features)
        mx.eval(audio)
        human = "<|audio_start|>" + "<|audio_pad|>" * audio.shape[1] + "<|audio_end|>"
        return human, {"audio": audio}

    @on_metal
    def _prefill(self, ids: torch.Tensor, feats: AudioFeatures) -> int:
        tokens = ids[0].tolist()
        if "audio" in feats:
            _, embeds = self.model.inject_audio(tokens, feats["audio"])
        else:
            embeds = self.model.model.word_embeddings(mx.array([tokens], dtype=mx.int32))
        hidden = self.model.model(inputs_embeds=embeds[:, :256], cache=self.cache)
        mx.eval(hidden)
        for start in range(256, len(tokens), 256):
            hidden = self.model.model(inputs_embeds=embeds[:, start:start + 256], cache=self.cache)
            mx.eval(hidden)
        logits = self.model.lm_head(hidden[:, -1])
        token = int(np.array(mx.argmax(logits, axis=-1), dtype=np.int32)[0])
        self.cur_len += len(tokens)
        return token

    @on_metal
    def _step(self, tok: int) -> int:
        logits = self.model(mx.array([[tok]], dtype=mx.int32), cache=self.cache)
        token = int(np.array(mx.argmax(logits[:, -1], axis=-1), dtype=np.int32)[0])
        self.cur_len += 1
        return token

    def transcribe(self, wav16k: np.ndarray, max_new_tokens: int = 200,
                   cancel: threading.Event | None = None) -> str:
        if cancel is not None and cancel.is_set():
            return ""
        from mlx_audio.stt.models.ming_v3_asr.ming_v3_asr import ASR_INSTRUCTION, decode_transcript

        human, feats = self._audio_input(wav16k)
        prompt = (f"<role>SYSTEM</role>You are a speech recognition engine.<|role_end|>"
                  f"<role>HUMAN</role>{human}{ASR_INSTRUCTION}<|role_end|>"
                  "<role>ASSISTANT</role>\n<think></think>")
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        request = copy.copy(self)
        request._clear_state()
        request.cur_len = 0
        tok = request._prefill(ids, feats)
        tokens = []
        limit = min(max_new_tokens, int(len(wav16k) / 16000 * 7) + 16)
        for _ in range(limit):
            if tok in {156892, 156895} or cancel is not None and cancel.is_set():
                break
            tokens.append(tok)
            tok = request._step(tok)
        if cancel is not None and cancel.is_set():
            return ""
        text, _ = decode_transcript(self.tok.decode(tokens, skip_special_tokens=True))
        return text

    def compact(self) -> None:
        text = self._rollover_text(budget=min(self.max_ctx // 2, 4096))
        history = self.history
        self.reset()
        self.history = history
        self._prefill_system()
        if text:
            ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids
            self._prefill(ids, {})
