"""Which Granite checkpoint we run, pinned, and how to fetch it.

Kept free of torch/transformers imports so the CLI can read the defaults
without paying for a CUDA import.
"""

from __future__ import annotations

GRANITE_MODEL = "ibm-granite/granite-speech-5.0-2b"
# A commit on the Hub, not `main`: the repo is still being updated after
# release, and an unpinned download on another machine would silently get
# different weights than the ones the benchmarks were measured with.
# Bump deliberately: change it, `make model`, re-run a known video, compare.
GRANITE_REVISION = "c075cf19e041a2e4d16c5ee24763e5b037d8f479"

# What transformers actually loads. The repo also carries a 411 MB
# out_llm.safetensors, a sample WAV and benchmark images we never read.
WEIGHT_FILES = ["*.json", "*.jinja", "*.txt", "*.safetensors.index.json",
                "model*.safetensors"]


def default_revision(model: str) -> str | None:
    """The pinned commit for our default model; `main` for anything else."""
    return GRANITE_REVISION if model == GRANITE_MODEL else None


def download(model: str = GRANITE_MODEL, revision: str | None = None) -> str:
    """Fetch the weights into the Hugging Face cache; return the local path.

    The model is public and ungated, so no token is needed. Set HF_TOKEN
    anyway on a shared IP to avoid anonymous rate limits.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model,
        revision=revision or default_revision(model),
        allow_patterns=WEIGHT_FILES,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Download the Granite weights.")
    p.add_argument("--model", default=GRANITE_MODEL)
    p.add_argument("--revision", default=None, help="default: the pinned commit")
    a = p.parse_args()
    print(download(a.model, a.revision))
