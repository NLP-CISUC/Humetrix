from pathlib import Path

from gensim.models import KeyedVectors
from safetensors.numpy import save_file

EMBEDDINGS_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("data/hub/glove")
LANGUAGES = ["en", "es", "fr", "zh"]

for language in LANGUAGES:
    kv = KeyedVectors.load(str(EMBEDDINGS_DIR / language / "glove_s300.gensim"))
    out = OUTPUT_DIR / language
    out.mkdir(parents=True, exist_ok=True)

    save_file({"embeddings": kv.vectors}, out / "embeddings.safetensors")
    out_vocab_path = out / "vocab.txt"
    with open(out_vocab_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(kv.index_to_key) + "\n")

    with open(out_vocab_path, encoding="utf-8", newline="\n") as f:
        vocab = f.read().split("\n")[:-1]
    assert vocab == kv.index_to_key, f"{language}: vocabulary mismatch"
    print(language, kv.vectors.shape, "OK")
