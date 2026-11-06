from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SKIPGRAM_DIR = Path("results/skipgram")
OUTPUT_DIR = Path("data/hub/skipgram")
LANGUAGES = ["en", "es", "fr", "pt", "zh"]

for language in LANGUAGES:
    out = OUTPUT_DIR / language
    out.mkdir(parents=True, exist_ok=True)

    state_dict = torch.load(SKIPGRAM_DIR / f"{language}.pt", map_location="cpu")
    save_file(state_dict, out / "model.safetensors")

    reloaded = load_file(out / "model.safetensors")
    assert reloaded.keys() == state_dict.keys(), f"{language}: keys mismatch"
    assert all(torch.equal(reloaded[k], state_dict[k]) for k in state_dict)
    shapes = {k: tuple(v.shape) for k, v in state_dict.items()}
    print(language, shapes, "ok")
