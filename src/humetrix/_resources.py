"""Download and load the external resources used by the metrics."""

from pathlib import Path
from typing import Iterable, Optional

import safetensors.numpy
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download

from .configs import EXTERNAL_EMBEDDINGS, RESOURCES_REPO, RESOURCES_REVISION


def download_resource(
    filenames: Iterable[str],
    subfolder: Optional[str] = None,
    repo_id: str = RESOURCES_REPO,
    revision: str = RESOURCES_REVISION,
) -> Path:
    """Download files from the Hugging Face Hub repository and return their directory."""

    paths = [
        hf_hub_download(repo_id, filename, subfolder=subfolder, revision=revision)
        for filename in filenames
    ]

    if not paths:
        raise ValueError("At least one filename is required.")
    return Path(paths[0]).parent


def download_embeddings(language: str) -> Path:
    """Download the default glove embeddings for a language."""
    filenames = ["embeddings.safetensors", "vocab.txt"]
    if language in EXTERNAL_EMBEDDINGS:
        repo_id, revision = EXTERNAL_EMBEDDINGS[language]
        return download_resource(filenames, repo_id=repo_id, revision=revision)
    return download_resource(filenames, subfolder=f"glove/{language}")


def load_embeddings(directory: Path) -> KeyedVectors:
    """Load embeddings from `embeddings.safetensors` + `vocab.txt` files."""
    tensors = safetensors.numpy.load_file(directory / "embeddings.safetensors")
    vectors = tensors["embeddings"]

    with open(directory / "vocab.txt", encoding="utf-8", newline="\n") as f:
        vocab = f.read().split("\n")
    if vocab[-1] == "":
        vocab.pop()

    if len(vocab) != len(vectors):
        raise ValueError(
            f"{directory}: vocab.txt has {len(vocab)} words, "
            f"but the matrix has {len(vectors)} rows."
        )

    embeddings = KeyedVectors(vector_size=vectors.shape[1])
    embeddings.add_vectors(vocab, vectors)
    return embeddings
