"""Exercise real cached audio decoding with a deterministic latent producer."""
import sys
import threading
import re
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx_audio.tts.models.bailingmm.bailingmm import AudioVAE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metal_worker import on_metal
from mlx_talker import NativeTalker, Talker
from speech_frontend import SpeechFrontend


@pytest.fixture(scope="session")
def frontend() -> SpeechFrontend:
    return SpeechFrontend()


class RandomFlow:
    def sample(self, hidden: mx.array, _history: mx.array) -> mx.array:
        return hidden + mx.random.normal(hidden.shape)


class LatentSampler:
    """Replace only latent generation; keep the real decoder and its streaming state."""

    def __init__(self, audio: AudioVAE, latents: mx.array) -> None:
        self.audio = audio
        self.flowloss = RandomFlow()
        self.latents = latents
        self.consumed = 0
        self.owner = threading.current_thread()
        self.producer_threads: list[threading.Thread] = []
        self.closed_on: threading.Thread | None = None
        self.texts: list[str] = []
        self.references: list[tuple[str | None, mx.array | None]] = []

    def _speaker_embeddings(self, _speaker: mx.array | None, _zero: bool) -> list[mx.array]:
        return []

    def sample(self, _prompt: str, _text: str, *, instruction: str | None,
               prompt_text: str | None, prompt_latent: mx.array | None,
               spk_emb: list[mx.array], max_decode_steps: int,
               flow_steps: int) -> Iterator[tuple[mx.array, bool]]:
        self.consumed = 0
        self.closed_on = None
        self.texts.append(_text)
        self.references.append((prompt_text, prompt_latent))
        try:
            for index in range(min(max_decode_steps, self.latents.shape[1])):
                self.producer_threads.append(threading.current_thread())
                self.consumed += 1
                yield self.latents[:, index:index + 1], index + 1 == self.latents.shape[1]
        finally:
            self.closed_on = threading.current_thread()

    sample_anchored = sample


@pytest.fixture
@on_metal
def speech(monkeypatch: pytest.MonkeyPatch, frontend: SpeechFrontend) -> tuple[Talker, LatentSampler]:
    mx.random.seed(17)
    backbone = {
        "model_type": "qwen2", "rms_norm_eps": 1e-6,
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 1,
        "max_position_embeddings": 512, "rope_theta": 1000000.0,
        "is_causal": True, "use_sliding_window": True, "sliding_window": 64,
    }
    audio = AudioVAE({
        "sample_rate": 400, "patch_size": 4,
        "enc_kwargs": {"backbone": backbone, "input_dim": 8, "hop_size": 8, "latent_dim": 4},
        "dec_kwargs": {"backbone": backbone, "output_dim": 8, "latent_dim": 4},
    })
    audio.eval()
    # Isolate streaming state from float32 GEMV/GEMM projection precision differences.
    audio.decoder.fc1.weight = mx.pad(mx.eye(4), [(0, 12), (0, 0)])
    audio.decoder.fc1.bias = mx.zeros((16,))
    latents = mx.random.normal((1, 21, 4)).astype(mx.bfloat16).astype(mx.float32)
    mx.eval(audio.parameters(), latents)
    sampler = LatentSampler(audio, latents)
    talker = Talker.__new__(Talker)
    monkeypatch.setattr(talker, "t", sampler, raising=False)
    monkeypatch.setattr("mlx_talker.SpeechFrontend", lambda: frontend)
    talker.frontend = frontend
    talker.lock = threading.Lock()
    talker.steps = 10
    talker.anchors = {}
    return talker, sampler


@on_metal
def whole_audio(sampler: LatentSampler, count: int) -> np.ndarray:
    waveform, _, _ = sampler.audio.decode(
        sampler.latents[:, :count], use_cache=False, last_chunk=True,
    )
    waveform = mx.array(waveform)
    mx.eval(waveform)
    return np.array(waveform[0], dtype=np.float32)


def test_audio_arrives_before_latent_generation_finishes(speech: tuple[Talker, LatentSampler]) -> None:
    # Given a longer deterministic utterance and the real streaming decoder.
    talker, sampler = speech
    with closing(talker.synth_stream("hello")) as stream:
        # When the consumer requests its first PCM chunk.
        first = next(stream)
        # Then generation is still in progress and ran on the Metal-owning thread.
        assert first.size > 0
        assert np.isfinite(first).all()
        assert sampler.consumed < sampler.latents.shape[1]
        assert sampler.closed_on is None
        assert all(thread is sampler.owner for thread in sampler.producer_threads)
        assert sampler.owner is not threading.current_thread()


@pytest.mark.parametrize("max_steps", [1, 2, 5, 7, 21, 750])
def test_stream_matches_whole_decode_including_final_tail(
    speech: tuple[Talker, LatentSampler], max_steps: int,
) -> None:
    # Given fixed latents, including sequences long enough to cross the attention window.
    talker, sampler = speech
    expected = whole_audio(sampler, min(max_steps, sampler.latents.shape[1]))
    # When the utterance ends naturally or reaches its generation limit.
    chunks = list(talker.synth_stream("hello", max_steps=max_steps))
    actual = np.concatenate(chunks)
    # Then no samples are lost, repeated, or discontinuous at chunk boundaries.
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert np.isfinite(actual).all()
    assert sampler.closed_on is sampler.owner


def test_cancel_stops_sampling_and_cleans_up_on_metal(speech: tuple[Talker, LatentSampler]) -> None:
    # Given a stream whose first chunk has already reached the consumer.
    talker, sampler = speech
    cancel = threading.Event()
    with closing(talker.synth_stream("hello", cancel=cancel)) as stream:
        next(stream)
        consumed = sampler.consumed
        # When cancellation arrives before the next chunk is requested.
        cancel.set()
        remaining = list(stream)
        # Then it emits nothing else, does not sample ahead, and closes on the owner.
        assert remaining == []
        assert sampler.consumed == consumed
        assert sampler.closed_on is sampler.owner
    assert not talker.lock.locked()


def test_closing_partial_stream_releases_the_voice_request(speech: tuple[Talker, LatentSampler]) -> None:
    # Given an active speech request.
    talker, sampler = speech
    # When its consumer leaves after the first chunk.
    with closing(talker.synth_stream("hello")) as stream:
        next(stream)
    # Then worker-side resources and the outer request lock have both been released.
    assert sampler.closed_on is sampler.owner
    assert not talker.lock.locked()


def test_next_utterance_has_fresh_decoder_state_after_cancellation(
    speech: tuple[Talker, LatentSampler],
) -> None:
    # Given a cancelled utterance with partially filled decoder buffers.
    talker, sampler = speech
    cancel = threading.Event()
    with closing(talker.synth_stream("hello", cancel=cancel)) as stream:
        next(stream)
        cancel.set()
    expected = whole_audio(sampler, sampler.latents.shape[1])
    # When another request uses the same Talker.
    actual = talker.synthesize("hello")
    # Then it starts from clean state rather than retaining the cancelled audio tail.
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_long_input_is_spoken_in_complete_bounded_fragments(
    speech: tuple[Talker, LatentSampler],
) -> None:
    # Given text longer than one native Talker fragment.
    talker, sampler = speech
    text = " ".join(f"word{chr(97 + index // 26)}{chr(97 + index % 26)}" for index in range(120))
    expected = whole_audio(sampler, sampler.latents.shape[1])
    # When the complete request is synthesized.
    actual = talker.synthesize(text)
    # Then every input word reaches a bounded request with fresh decoder state.
    assert len(sampler.texts) == 3
    assert all(len(piece.split()) <= 50 for piece in sampler.texts)
    assert re.findall("[A-Za-z]+", " ".join(sampler.texts)) == text.split()
    np.testing.assert_allclose(actual, np.tile(expected, 3), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(("text", "cancelled"), [("", False), ("hello", True)])
def test_empty_or_pre_cancelled_request_does_not_sample(
    speech: tuple[Talker, LatentSampler], text: str, cancelled: bool,
) -> None:
    # Given an empty request or cancellation before generation starts.
    talker, sampler = speech
    cancel = threading.Event()
    if cancelled:
        cancel.set()
    # When the request is consumed.
    chunks = list(talker.synth_stream(text, cancel=cancel))
    # Then it emits no audio and never advances the latent producer.
    assert chunks == []
    assert sampler.consumed == 0


def test_compiled_flow_preserves_seed_and_advances_random_state(
    speech: tuple[Talker, LatentSampler], monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given the real constructor and a small native sampling function.
    _, sampler = speech
    monkeypatch.setattr("mlx_talker.NativeTalker.from_pretrained", lambda _path: sampler)
    Talker(use_graphs=True)

    @on_metal
    def draw_sequence() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden = mx.zeros((1, 4))
        mx.random.seed(19)
        first = sampler.flowloss.sample(hidden, hidden)
        mx.eval(first)
        following = sampler.flowloss.sample(hidden, hidden)
        mx.eval(following)
        mx.random.seed(19)
        repeated = sampler.flowloss.sample(hidden, hidden)
        mx.eval(repeated)
        return np.array(first), np.array(following), np.array(repeated)

    # When compiled sampling advances and then the same seed is restored.
    first, following, repeated = draw_sequence()
    # Then it neither freezes random noise nor ignores a reset seed.
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, following)


@pytest.mark.parametrize("enabled", [False, True])
def test_anchor_setting_keeps_designed_voices_when_builtin_anchors_are_disabled(
    speech: tuple[Talker, LatentSampler], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, enabled: bool,
) -> None:
    # Given one built-in anchor and one explicitly designed voice.
    _, sampler = speech
    for name, instruction in (("builtin", None), ("designed", "custom voice")):
        torch.save(
            {"text": name, "latents": torch.zeros(1, 1, 4), "instruction": instruction},
            tmp_path / f"{name}.pt",
        )
    monkeypatch.setattr("mlx_talker.NativeTalker.from_pretrained", lambda _path: sampler)
    monkeypatch.setattr("mlx_talker.VoicePolicy.ANCHOR_DIR", tmp_path)
    monkeypatch.setenv("MING_VOICE_ANCHORS", "1" if enabled else "0")
    # When the real constructor loads anchors with that setting.
    talker = Talker()
    # Then disabling ordinary anchors preserves only the designed voice.
    assert set(talker.anchors) == ({"builtin", "designed"} if enabled else {"designed"})
    assert talker.designed_voices == ["designed"]


@pytest.mark.parametrize(("text", "anchored"), [
    ("Hello there.", True),
    ("\u4f60\u597d\u3002", False),
    ("Hello\uff0c\u4f60\u597d\u3002", False),
])
def test_english_acoustic_anchor_is_used_only_without_chinese_text(
    speech: tuple[Talker, LatentSampler], text: str, anchored: bool,
) -> None:
    # Given an English acoustic reference for the selected voice.
    talker, sampler = speech
    talker.anchors["US_Stella"] = {
        "text": "Hello.", "latents": sampler.latents, "instruction": None,
    }
    # When the request contains English, Chinese, or mixed text.
    audio = talker.synthesize(text)
    # Then only English-only text is conditioned on the English reference.
    reference_text, reference_latents = sampler.references[0]
    assert (reference_text is not None) is anchored
    assert (reference_latents is not None) is anchored
    assert audio.size > 0
    assert np.isfinite(audio).all()


@pytest.fixture
@on_metal
def native_sampler(monkeypatch: pytest.MonkeyPatch) -> NativeTalker:
    from mlx_audio.sts.models.ming_v3_audio.talker import TalkerFlowLoss

    # Use real attention, latent projection, diffusion, and stopping with tiny weights.
    mx.random.seed(17)
    backbone = {
        "model_type": "qwen2", "rms_norm_eps": 1e-6,
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 8,
        "max_position_embeddings": 512, "rope_theta": 1000000.0,
        "is_causal": True, "use_sliding_window": True, "sliding_window": 64,
    }
    flow = {"hidden_size": 16, "depth": 1, "num_heads": 2,
            "patch_size": 1, "history_patch_size": 1}
    model = NativeTalker({
        "model_type": "dense", "llm_config": backbone,
        "audio_tokenizer_config": {
            "sample_rate": 400, "patch_size": 4,
            "enc_kwargs": {"backbone": backbone, "input_dim": 8, "hop_size": 8, "latent_dim": 4},
            "dec_kwargs": {"backbone": backbone, "output_dim": 8, "latent_dim": 4},
        },
        "ditar_config": flow,
        "aggregator_config": {"hidden_size": 16, "depth": 1, "num_heads": 2},
    })
    monkeypatch.setattr(model, "flowloss", TalkerFlowLoss(4, 16, flow))
    model.stop_head.weight = mx.zeros_like(model.stop_head.weight)
    model.stop_head.bias = mx.array([-10.0, 10.0])

    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return [2 if text == "<audio>" else 1]

        def convert_tokens_to_ids(self, _token: str) -> int:
            return 2

    monkeypatch.setattr(model, "tokenizer", Tokenizer())
    model.eval()
    return model


@on_metal
@pytest.mark.parametrize(("anchored", "limit", "count"), [
    (False, 40, 5), (True, 40, 17), (True, 7, 7),
])
def test_native_sampler_honors_anchor_minimum_without_exceeding_limit(
    native_sampler: NativeTalker, anchored: bool, limit: int, count: int,
) -> None:
    # Given a real tiny native sampler whose stop head always requests an early stop.
    text = "hello " * 10
    anchor = mx.zeros((1, 2, 4)) if anchored else None
    # When generation uses an acoustic anchor, no anchor, or a shorter explicit limit.
    sample = native_sampler.sample_anchored if anchored else native_sampler.sample
    output = list(sample(
        "prompt", text, prompt_text="reference" if anchored else None,
        prompt_latent=anchor, max_decode_steps=limit, flow_steps=1,
    ))
    # Then anchored speech waits for the upstream minimum, but respects the hard limit.
    assert len(output) == count
    assert all(np.isfinite(np.array(latent)).all() for latent, _last in output)


def test_anchored_speech_flushes_the_native_sampler_tail(
    speech: tuple[Talker, LatentSampler], native_sampler: NativeTalker,
) -> None:
    # Given the public adapter with real tiny native sampling and audio decoding.
    talker, _sampler = speech
    text = " ".join(["hello"] * 10)

    @on_metal
    def configure() -> int:
        talker.t = native_sampler
        talker.steps = 1
        talker.anchors["US_Stella"] = {
            "text": "reference", "latents": mx.zeros((1, 2, 4)), "instruction": None,
        }
        expected, _, _ = native_sampler.audio.decode(
            mx.zeros((1, 17, 4)), use_cache=False, last_chunk=True,
        )
        expected = mx.array(expected)
        mx.eval(expected)
        return expected.shape[1]

    expected_samples = configure()
    # When the adapter streams the anchored request through its Metal worker.
    audio = np.concatenate(list(talker.synth_stream(text, max_steps=40)))
    # Then it includes the full minimum-length utterance and final decoder tail.
    assert audio.size == expected_samples
    assert np.isfinite(audio).all()


@pytest.mark.parametrize("cloned", [False, True])
def test_frontend_strips_language_tag_and_normalizes_english_numbers(
    frontend: SpeechFrontend, cloned: bool,
) -> None:
    # Given direct TTS input that does not pass through the conversation Fragmenter.
    text = "English\tI have 21 apples."
    # When the upstream CPU frontend prepares either a built-in or cloned voice.
    fragments = list(frontend.prepare(text, cloned=cloned))
    # Then only normalized speech text reaches the model.
    words = re.findall("[A-Za-z]+", " ".join(part for part, _prompt in fragments))
    assert words == ["I", "have", "twenty", "one", "apples"]


def test_frontend_normalizes_chinese_numbers(frontend: SpeechFrontend) -> None:
    # Given Chinese speech text containing an Arabic number.
    text = "\u6211\u670921\u4e2a\u82f9\u679c\u3002"
    # When the pinned Chinese FST frontend prepares the text.
    fragments = list(frontend.prepare(text, cloned=False))
    # Then the number is verbalized instead of being left to the speech model.
    normalized = "".join(part for part, _prompt in fragments)
    assert "21" not in normalized
    assert "\u4e8c\u5341\u4e00" in normalized


def test_frontend_preserves_paralinguistic_tags(frontend: SpeechFrontend) -> None:
    # Given a model control token mixed with text requiring number normalization.
    text = "[Laughter] I have 21 apples."
    # When the text crosses both English number and FST normalization.
    fragments = list(frontend.prepare(text, cloned=False))
    # Then the exact control token survives and no internal placeholder leaks out.
    normalized = " ".join(part for part, _prompt in fragments)
    assert "[Laughter]" in normalized
    assert "MINGPARALINGPLACEHOLDER" not in normalized
    assert "21" not in normalized


def test_frontend_detects_foreign_language_without_destructive_normalization(
    frontend: SpeechFrontend,
) -> None:
    # Given a complete German request with characters outside the zh/en normalizer.
    text = "Guten Morgen, wie geht es Ihnen heute? Ich freue mich auf einen sch\u00f6nen Tag."
    # When FastText detects the request's language before fragmentation.
    fragments = list(frontend.prepare(text, cloned=False))
    # Then the parsed language control is German and the accented text is preserved.
    assert all(frontend.policy.target_language_from_prompt(prompt) == "German"
               for _part, prompt in fragments)
    assert "sch\u00f6nen" in " ".join(part for part, _prompt in fragments)
