"""Exercise real cached audio decoding with a deterministic latent producer."""
import sys
import threading
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_audio.tts.models.bailingmm.bailingmm import AudioVAE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metal_worker import on_metal
from mlx_talker import Talker


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

    def _speaker_embeddings(self, _speaker: mx.array | None, _zero: bool) -> list[mx.array]:
        return []

    def sample(self, _prompt: str, _text: str, *, instruction: str | None,
               prompt_text: str | None, prompt_latent: mx.array | None,
               spk_emb: list[mx.array], max_decode_steps: int,
               flow_steps: int) -> Iterator[tuple[mx.array, bool]]:
        self.consumed = 0
        self.closed_on = None
        try:
            for index in range(min(max_decode_steps, self.latents.shape[1])):
                self.producer_threads.append(threading.current_thread())
                self.consumed += 1
                yield self.latents[:, index:index + 1], index + 1 == self.latents.shape[1]
        finally:
            self.closed_on = threading.current_thread()


@pytest.fixture
@on_metal
def speech(monkeypatch: pytest.MonkeyPatch) -> tuple[Talker, LatentSampler]:
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
