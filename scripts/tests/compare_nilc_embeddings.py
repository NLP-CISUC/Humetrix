import numpy as np
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

REPO_ID = "nilc-nlp/glove-300d"
LOCAL_PATH = "data/embeddings/pt/glove_s300.gensim"

local = KeyedVectors.load(LOCAL_PATH)

path = hf_hub_download(REPO_ID, "embeddings.safetensors")
nilc_vectors = load_file(path)["embeddings"]

vocab_path = hf_hub_download(REPO_ID, "vocab.txt")
with open(vocab_path) as f:
    nilc_vocab = [w.rstrip("\n") for w in f]

print(
    f"NILC: {len(nilc_vocab)} words, shape {nilc_vectors.shape}, type {nilc_vectors.dtype}"
)
print(
    f"Local: {len(local.index_to_key)} words, shape {local.vectors.shape}, type {local.vectors.dtype}"
)

if nilc_vocab == local.index_to_key:
    print("Same vocabulary, same order")
    max_diff = np.abs(nilc_vectors - local.vectors).max()
    print(f"Identical vectors: {np.array_equal(nilc_vectors, local.vectors)}")
    print(f"Max abs difference: {max_diff}")
else:
    common = set(nilc_vocab) & set(local.key_to_index)
    print(f"Different vocabulary or order. Common words: {len(common)}")
    print(f"Different words: {set(nilc_vocab) - set(local.key_to_index)}")
