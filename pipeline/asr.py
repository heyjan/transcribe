"""Granite Speech wrapper: ASR + speech translation, block at a time."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .models import GRANITE_MODEL as DEFAULT_MODEL, default_revision

# Prompts are taken verbatim from the model card's "Preferred prompt by task"
# table. Granite falls back to plain transcription on unfamiliar prompts, so
# deviating here degrades silently rather than erroring - don't improvise.
ASR_PROMPT = "transcribe the speech with proper punctuation and capitalization."
# Second try for a block that came back unpunctuated (see transcribe()).
ASR_RETRY_PROMPT = ASR_PROMPT + " Write complete sentences."
AST_PROMPT = "translate the speech to {language} with proper punctuation and capitalization."

# Punctuated AST only accepts the English prompt form, per the card.
LANGUAGE_NAMES = {
    "en": "English", "de": "German", "fr": "French",
    "es": "Spanish", "pt": "Portuguese", "ja": "Japanese",
    "it": "Italian", "zh": "Mandarin",
}


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def detect_repetition(text: str, min_repeats: int = 4) -> bool:
    """Flag the degenerate loop LLM-based ASR falls into on near-silent audio.

    We only report it; collapsing the text automatically would quietly discard
    legitimately repeated phrasing.
    """
    words = text.split()
    if len(words) < min_repeats * 2:
        return False
    for size in range(1, 11):
        if size * min_repeats > len(words):
            break
        for start in range(len(words) - size * min_repeats + 1):
            chunk = words[start:start + size]
            if all(
                words[start + k * size:start + (k + 1) * size] == chunk
                for k in range(1, min_repeats)
            ):
                return True
    return False


def punctuation_rate(text: str) -> float:
    """Sentence punctuation marks per word."""
    words = len(text.split())
    return len(re.findall(r"[.,?!;:]", text)) / words if words else 0.0


def detect_unpunctuated(text: str, min_words: int = 15) -> bool:
    """A block of real speech with essentially no punctuation.

    Normal speech here runs 8-25 marks per 100 words; the failure mode is a
    flat 0. Below 1 per 100 words over 15+ words is unambiguous.
    """
    return len(text.split()) >= min_words and punctuation_rate(text) < 0.01


_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def detect_untranslated(source: str, target: str, threshold: float = 0.6) -> bool:
    """Flag a translation that is really the source text copied through.

    Granite occasionally answers an AST prompt with the transcript instead --
    most often on blocks that open with a name or a product noun.

    Compared on word *sequences* rather than vocabulary overlap: a shared set
    of tokens only tells you both sides mention Anthropic and OpenAI, and a
    short brand-heavy sentence scores 0.63 on set overlap despite being a
    perfectly good translation. Alignment order is what actually distinguishes
    a copy. Measured over the test course, passthrough scores >= 0.98, genuine
    translations <= 0.21, and that same brand-heavy sentence 0.56.
    """
    a = [w.lower() for w in _WORD.findall(source)]
    b = [w.lower() for w in _WORD.findall(target)]
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= threshold


@dataclass
class Segment:
    start: float
    end: float


class GraniteSpeech:
    def __init__(self, model_name: str = DEFAULT_MODEL, revision: str | None = None,
                 device: str | None = None, dtype: torch.dtype = torch.bfloat16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.revision = revision or default_revision(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name, revision=self.revision)
        self.tokenizer = self.processor.tokenizer
        # transformers>=5 renamed torch_dtype -> dtype; keep both paths working.
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name, revision=self.revision, device_map=self.device, dtype=dtype,
            )
        except TypeError:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name, revision=self.revision, device_map=self.device,
                torch_dtype=dtype,
            )
        self.model.eval()

    def _generate(self, wav: np.ndarray, instruction: str, max_new_tokens: int) -> str:
        tensor = torch.from_numpy(wav).unsqueeze(0)  # [1, N], mono as the card requires
        chat = [{"role": "user", "content": f"<|audio|>{instruction}"}]
        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            prompt, tensor, device=self.device, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,      # greedy: transcription is not a creative task
                num_beams=1,
            )
        n_in = inputs["input_ids"].shape[-1]
        text = self.tokenizer.batch_decode(
            out[0, n_in:].unsqueeze(0), add_special_tokens=False, skip_special_tokens=True
        )[0]
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _budget(seconds: float, wordy: bool = False) -> int:
        """Token budget for a block. Speech tops out near 3 words/s; German
        runs longer than the English it came from, hence the multiplier.
        Scales with the block, so a longer --target is never truncated."""
        est = seconds * 8.0 * (1.3 if wordy else 1.0)
        return int(max(64, est + 64))

    def transcribe(self, wav: np.ndarray, seconds: float) -> str:
        """Granite 5.0 sometimes ignores the punctuation half of the prompt
        and returns a bare word stream (4 of 17 blocks on a conversational
        clip; flaky, not tied to the audio). One retry with a reworded prompt
        recovers some; whatever stays bare is flagged by detect_unpunctuated."""
        budget = self._budget(seconds)
        text = self._generate(wav, ASR_PROMPT, budget)
        if detect_unpunctuated(text):
            retry = self._generate(wav, ASR_RETRY_PROMPT, budget)
            if punctuation_rate(retry) > punctuation_rate(text):
                text = retry
        return text

    def translate(self, wav: np.ndarray, seconds: float, lang: str) -> str:
        language = LANGUAGE_NAMES.get(lang, lang)
        return self._generate(
            wav, AST_PROMPT.format(language=language), self._budget(seconds, wordy=True)
        )
