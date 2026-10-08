import json
from pathlib import Path

import pandas as pd
import polars as pl
from ollama import Client
from pydantic import BaseModel
from tqdm import tqdm


class PunAnnotation(BaseModel):
    location: str
    interpretation: str


client = Client("http://crai-04.dei.uc.pt:8080")

SYS_PROMPT = """
A pun is a joke that exploits a word or expression with two meanings.
Given a pun, identify:
- "location": the exact word or expression in the text that carries the double meaning, copied as it appears in the text.
- "interpretation": the alternative word or expression evoked by it. If the pun relies on two meanings of the same word (homographic pun), repeat the same word.

Answer only with JSON.
"""

MODELS = ["llama4", "gemma3:27b"]
SEED = 42

FEW_SHOT = {
    "en": ["cup.34", "cup.148", "cup.149", "cup.2795"],
    "pt": [
        "puntuguese.1.1.H",
        "puntuguese.2.3.H",
        "puntuguese.2.2.H",
        "puntuguese.2.16.H",
    ],
}

CORPORA_NAMES = {"en": "cup", "pt": "puntuguese"}
CORPORA = {
    language: pd.read_json(f"data/humor_interpretation/{name}.json", orient="index")
    for language, name in CORPORA_NAMES.items()
}

RESULTS_PATH = Path("data/humor_interpretation/llm")


def build_messages(text, language):
    messages = [{"role": "system", "content": SYS_PROMPT}]
    corpus = CORPORA[language]
    for idx in FEW_SHOT[language]:
        location, interpretation, fs_text = corpus.loc[
            idx, ["location", "interpretation", "text"]
        ]
        example_answer = json.dumps(
            {"location": location, "interpretation": interpretation}, ensure_ascii=False
        )

        messages.append({"role": "user", "content": fs_text})
        messages.append({"role": "assistant", "content": example_answer})
    messages.append({"role": "user", "content": text})
    return messages


def get_test_data(language, seed=SEED):
    corpus = CORPORA[language]
    corpus = corpus.loc[~corpus.index.isin(FEW_SHOT[language])]
    return corpus.sample(n=100, random_state=seed)


def annotate(text, model, language, seed=SEED):
    response = client.chat(
        model=model,
        messages=build_messages(text, language),
        format=PunAnnotation.model_json_schema(),
        options={"temperature": 0, "seed": seed},
    )
    return PunAnnotation.model_validate_json(response.message.content)


def load_done(path: Path):
    if not path.exists():
        return set()
    return set(pl.read_ndjson(path)["index"])


def run_test(model, language):
    savepath = RESULTS_PATH / model / f"{CORPORA_NAMES[language]}.jsonl"
    savepath.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(savepath)

    test_data = get_test_data(language)
    test_data = test_data.loc[~test_data.index.isin(done)]
    if test_data.empty:
        return

    records = []
    for idx, row in tqdm(
        test_data.iterrows(), total=len(test_data), desc=f"{model} | {language}"
    ):
        try:
            annotation = annotate(row["text"], model, language).model_dump()
        except Exception as e:
            annotation = {"location": None, "interpretation": None}
            print(f"{idx}: {e!r}")
        records.append({"index": idx, "text": row["text"], **annotation})

    df = pl.from_records(records)
    if savepath.exists():
        df = pl.concat(
            [
                pl.read_ndjson(savepath),
                df,
            ],
            how="vertical_relaxed",
        )
    df.write_ndjson(savepath)


if __name__ == "__main__":
    for model in MODELS:
        for language in CORPORA:
            run_test(model, language)
