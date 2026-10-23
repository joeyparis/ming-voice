"""Small real Metal models exercise rollback without downloading full weights."""
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx_audio.sts.models.ming_v3_audio import Model, ModelConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_thinker import AudioFeatures, Thinker
from metal_worker import on_metal


@pytest.fixture
@on_metal
def thinker() -> Thinker:
    mx.random.seed(17)
    model = Model(ModelConfig(
        text_config={
            "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 4,
            "vocab_size": 32, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 32, "q_lora_rank": 4, "kv_lora_rank": 4,
            "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8,
            "num_experts": 4, "num_experts_per_tok": 2,
            "moe_intermediate_size": 8, "moe_shared_expert_intermediate_size": 8,
            "n_group": 2, "topk_group": 1,
        },
        audio_config={
            "n_mels": 4, "n_audio_ctx": 16, "n_audio_state": 8,
            "n_audio_head": 2, "n_audio_layer": 1, "whisper_chunk_size": 16,
        },
        audio_token_id=5,
    ))
    model.eval()
    th = Thinker.__new__(Thinker)
    th.model = model
    th.system_prompt = "test"
    th.tools = None
    th._tools_sig = None
    th.reset()
    return th


@on_metal
def test_cached_prefill_matches_native_forward(thinker: Thinker) -> None:
    # Given a real tiny Ming backbone and a fixed prompt.
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = thinker.model(mx.array(ids.numpy()))
    # When the adapter prefills the prompt and takes one cached step.
    token = thinker._prefill(ids, {})
    following = thinker._step(token)
    complete = thinker.model(mx.array([[1, 2, 3, 4, token]]))
    # Then both predictions agree with uncached native inference.
    assert token == int(np.argmax(np.array(expected[:, -1]), axis=-1)[0])
    assert following == int(np.argmax(np.array(complete[:, -1]), axis=-1)[0])
    assert thinker.ctx_len() == 5


@on_metal
def test_snapshot_can_be_reused_after_interrupted_decode(thinker: Thinker) -> None:
    # Given cached attention and recurrent state at a turn boundary.
    thinker._prefill(torch.tensor([[1, 2, 3]]), {})
    snap = thinker.snapshot()
    thinker._step(4)
    expected = thinker._save_state()
    # When another branch is decoded, then rolled back twice to the same snapshot.
    thinker._step(9)
    thinker.history.append({"content": "interrupted"})
    for _ in range(2):
        thinker.restore(snap)
        thinker._step(4)
        actual = thinker._save_state()
        # Then neither KV offsets nor recurrent arrays leak from the discarded branch.
        assert actual[0] == expected[0]
        assert thinker.history == []
        for left, right in zip(actual[1], expected[1], strict=True):
            for a, b in zip(left.state, right.state, strict=True):
                if a is not None:
                    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-6)


@on_metal
def test_isolated_request_restores_cache_and_conversation(thinker: Thinker) -> None:
    # Given an existing conversation with a pending role terminator.
    thinker._prefill(torch.tensor([[1, 2]]), {})
    thinker.pending = "<|role_end|>"
    thinker.history = [{"user": "original", "content": "reply"}]
    expected = thinker.snapshot()
    # When an API-style isolated request resets and decodes unrelated input.
    with thinker.isolated():
        thinker.reset("temporary")
        thinker._prefill(torch.tensor([[7, 8, 9]]), {})
    # Then live metadata and the next prediction match the original state.
    assert thinker.ctx_len() == 2
    assert thinker.pending == "<|role_end|>"
    assert thinker.history == [{"user": "original", "content": "reply"}]
    assert thinker.system_prompt == "test"
    actual = thinker._prefill(torch.tensor([[3, 4, 5]]), {})
    thinker.restore(expected)
    assert thinker._prefill(torch.tensor([[3, 4, 5]]), {}) == actual


@on_metal
def test_snapshot_restores_history_after_context_reset(thinker: Thinker) -> None:
    # Given a snapshot before the active cache is replaced by a rollover.
    thinker.history = [{"user": "keep", "content": "this"}]
    snap = thinker.snapshot()
    thinker.reset()
    thinker.history = [{"user": "keep", "content": "this"}, {"content": "discard"}]
    # When the speculative turn is rejected after that reset.
    thinker.restore(snap)
    # Then surviving text is queued for rebuilding instead of restoring stale KV state.
    assert thinker.history == [{"user": "keep", "content": "this"}]
    assert thinker.rebuild
    assert thinker.ctx_len() == 0


@pytest.fixture
@on_metal
def transcription(thinker: Thinker, monkeypatch: pytest.MonkeyPatch) -> Thinker:
    # Keep real model/cache execution, with a tiny matching tokenizer vocabulary.
    class Tokenizer:
        def __call__(self, _text: str, **_kwargs: bool | str) -> SimpleNamespace:
            return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

        def decode(self, tokens: list[int], **_kwargs: bool) -> str:
            return " ".join(map(str, tokens))

    monkeypatch.setattr(thinker, "tok", Tokenizer(), raising=False)
    monkeypatch.setattr(thinker, "_audio_input", lambda _wav: ("", {}))
    thinker.model.lm_head.weight = mx.zeros_like(thinker.model.lm_head.weight)
    return thinker


@on_metal
def test_transcription_never_exposes_temporary_conversation_state(
    transcription: Thinker, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given live history that unlocked health/hello readers can observe during ASR.
    th = transcription
    th._prefill(torch.tensor([[1, 2, 3]]), {})
    th.history = [{"user": "keep", "content": "visible"}]
    th.pending = "<|role_end|>"
    expected = (th.ctx_len(), list(th.history), th.pending, th.epoch)
    seen = []
    prefill = Thinker._prefill

    def observe(request: Thinker, ids: torch.Tensor, feats: AudioFeatures) -> int:
        seen.append((th.ctx_len(), list(th.history), th.pending, th.epoch))
        return prefill(request, ids, feats)

    monkeypatch.setattr(Thinker, "_prefill", observe)
    # When transcription prefills and decodes on the real tiny model.
    th.transcribe(np.zeros(16000, np.float32), max_new_tokens=2)
    # Then readers see the live state throughout, not an empty ASR conversation.
    assert seen == [expected]
    assert (th.ctx_len(), th.history, th.pending, th.epoch) == expected


def test_startup_skips_voices_from_an_incompatible_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    # Given raw Mac and projected CUDA voice files, and no full model construction.
    torch.save(torch.zeros(1, 192), tmp_path / "mac.pt")
    torch.save(torch.zeros(16), tmp_path / "cuda.pt")
    talker = SimpleNamespace(
        designed_voices=[],
        t=SimpleNamespace(spk_head=SimpleNamespace(weight=torch.empty(16, 192))),
    )
    monkeypatch.setattr(server, "ENGINE", {})
    monkeypatch.setattr(server, "VOICES", ["US_Stella"])
    monkeypatch.setattr(server, "VOICE_DIR", tmp_path)
    monkeypatch.setattr(server, "Thinker", lambda: None)
    monkeypatch.setattr(server, "Talker", lambda **_kwargs: talker)
    monkeypatch.setattr(server, "_warmup", lambda: None)
    monkeypatch.setattr(server, "load_conversation", lambda _thinker: None)
    monkeypatch.setenv("MING_SMART_TURN", "0")
    # When startup loads the saved voices.
    with torch.set_grad_enabled(torch.is_grad_enabled()):
        server._load()
    # Then only a usable voice is advertised, and neither saved file is removed.
    assert list(server.ENGINE["custom_voices"]) == ["mac"]
    assert len(list(tmp_path.glob("*.pt"))) == 2
