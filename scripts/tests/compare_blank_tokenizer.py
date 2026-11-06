import pandas as pd
import spacy

from humetrix.configs import SPACY_MODELS

PUNCT_CHARS = list(spacy.pipeline.Sentencizer.default_punct_chars) + [",", ";"]
LIGHT_MODELS = {"zh": "zh_core_web_sm"}


def build_pipeline(nlp):
    nlp.add_pipe("sentencizer", config={"punct_chars": PUNCT_CHARS})
    return nlp


def segment(nlp, texts):
    with nlp.select_pipes(enable="sentencizer"):
        return [
            [[tok.lower_ for tok in sent] for sent in doc.sents]
            for doc in nlp.pipe(texts)
        ]


corpora = {
    "en": ["semeval", "humicroedit", "joker_clef_en", "expunations", "cup"],
    "fr": ["joker_clef_fr"],
    "es": ["joker_clef_es", "HAHA@IberLEF2021", "HUHU@IberLEF2023"],
    "pt": ["clemencio", "puntuguese", "joker_clef_pt"],
    "zh": ["chumor"],
}
paths = {
    "semeval": "data/humor_recognition/semeval.json",
    "humicroedit": "data/humor_recognition/humicroedit.json",
    "joker_clef_en": "data/humor_recognition/joker_clef_en.json",
    "expunations": "data/humor_recognition/expunations.json",
    "cup": "data/humor_interpretation/cup.json",
    "joker_clef_fr": "data/humor_recognition/joker_clef_fr.json",
    "joker_clef_es": "data/humor_recognition/joker_clef_es.json",
    "HAHA@IberLEF2021": "data/humor_recognition/HAHA@IberLEF2021.json",
    "HUHU@IberLEF2023": "data/humor_recognition/HUHU@IberLEF2023.json",
    "clemencio": "data/humor_recognition/clemencio.json",
    "puntuguese": "data/humor_recognition/puntuguese.json",
    "joker_clef_pt": "data/humor_recognition/joker_clef_pt.json",
    "chumor": "data/humor_recognition/chumor.json",
}

for language, datasets in corpora.items():
    full = build_pipeline(spacy.load(SPACY_MODELS[language]))
    if language in LIGHT_MODELS:
        light = build_pipeline(spacy.load(LIGHT_MODELS[language]))
    else:
        light = build_pipeline(spacy.blank(language))

    for corpus in datasets:
        texts = pd.read_json(paths[corpus], orient="index")["text"].tolist()
        full_segs = segment(full, texts)
        blank_segs = segment(light, texts)

        mismatches = [
            i for i, (a, b) in enumerate(zip(full_segs, blank_segs)) if a != b
        ]

        print(f"{corpus}: {len(mismatches)}/{len(texts)} mismatches")

        for i in mismatches[:3]:
            print(f"  text :{texts[i]}")
            print(f"  full :{full_segs[i]}")
            print(f"  blank :{blank_segs[i]}")
