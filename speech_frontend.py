"""CPU-only preparation from the pinned upstream Ming Talker frontend."""
from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path


class SpeechFrontend:
    """Load frontend assets without importing the upstream CUDA inference package."""

    def __init__(self) -> None:
        root = Path(os.environ.get(
            "MING_FRONTEND_DIR", Path(__file__).resolve().parent / "model-frontend",
        )).expanduser() / "talker"
        name = "_ming_voice_frontend"
        spec = importlib.util.spec_from_file_location(
            name, root / "front" / "__init__.py", submodule_search_locations=[str(root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError("Ming frontend assets are missing; run bash setup_mac.sh")
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
        self.policy = importlib.import_module(f"{name}.tts_prompt")
        self.numbers = importlib.import_module(f"{name}.front.number_en")
        self.segments = importlib.import_module(f"{name}.front.text_segment_cut")
        self.toolkit = importlib.import_module(f"{name}.front.toolkit")
        self.normalizer = importlib.import_module(f"{name}.talker_tn.talker_tn").TalkerTN()

    def prepare(self, text: str, *, cloned: bool) -> Iterator[tuple[str, str]]:
        """Match engine.Talker.prepare's language, tag, and text normalization."""
        text = re.sub(r"^[A-Za-z]{1,16}\t", "", text)
        language, text = self.policy.resolve_request_language(
            text, language=None, taskname="TTS" if cloned else "IP",
        )
        prompt = self.policy.append_optional_target_language(
            self.policy.DEFAULT_TTS_PROMPT, language,
        )
        skip_tn = self.policy.preserve_multilingual_text(
            self.policy.target_language_from_prompt(prompt),
        )
        for fragment in self._fragments(text):
            protected = self.policy.contains_paralinguistic_tag(fragment)
            if protected:
                fragment = self.policy.protect_paralinguistic_tags(fragment)
            if not skip_tn:
                if not self.segments.is_chinese(fragment):
                    fragment = self.numbers.normalize_numbers(fragment)
                fragment = self.toolkit.normalize_text_kit(
                    self.normalizer.normalize(fragment),
                ).lstrip("\uFF0C")
            if protected:
                fragment = self.policy.restore_paralinguistic_tags(fragment)
            if fragment.strip():
                yield fragment, prompt

    def _fragments(self, text: str) -> Iterator[str]:
        """Port MingTalker.text_fragments from upstream revision 3c9578f (MIT)."""
        buffered: list[str] = []
        count = 0
        for token in self.toolkit.tokenize_mixed_text_iterator(text):
            if not token:
                continue
            process = False
            if token[-1] in "\uFF01\uFF1F\u3002!?" and (
                len(buffered) >= 12 or count > 0 and len(buffered) >= 8
            ):
                process = True
                buffered.append(token)
            elif token[-1] in "\uFF0C," and len(buffered) >= 36:
                process = True
                buffered.append(token)
            elif token[-1] == "." and (
                len(buffered) >= 12 or count > 0 and len(buffered) >= 8
            ) and not re.search("[0-9]", buffered[-1][-1]):
                process = True
                buffered.append(token)
            elif token[-1] == "\n":
                if buffered:
                    if re.search(r"[\u4e00-\u9fff]", "".join(buffered)):
                        if re.search(r"[\u4e00-\u9fff]", buffered[-1][-1]):
                            buffered.append("\uFF0C")
                    else:
                        token = token[:-1] + "." if len(token) > 1 and re.search(
                            "[a-zA-Z]", token[-2],
                        ) else token[:-1]
                        buffered.append(token)
                if len(buffered) >= 12 or count > 0 and len(buffered) >= 8:
                    process = True
            else:
                buffered.append(token)
                continue
            if process:
                cleaned = self.toolkit.preprosess_text("".join(buffered))
                fragments = self.segments.cut_text_by_semantic_length(cleaned, 50)["fragments"]
                count += len(fragments)
                yield from fragments
                buffered = []
        if buffered and re.search(r"[a-zA-Z\u4e00-\u9fff1-9]", "".join(buffered)):
            cleaned = self.toolkit.preprosess_text("".join(buffered))
            cleaned = self.toolkit.add_period(cleaned)
            yield from self.segments.cut_text_by_semantic_length(cleaned, 50)["fragments"]
