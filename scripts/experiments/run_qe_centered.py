from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from humetrix import HumorAnalyzer, QuantumUncertainty

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
glove = {
    "en": "data/embeddings/en/glove_s300.gensim",
    "es": "data/embeddings/es/glove_s300.gensim",
    "fr": "data/embeddings/fr/glove_s300.gensim",
    "pt": "data/embeddings/pt/glove_s300.gensim",
    "zh": "data/embeddings/zh/glove_s300.gensim",
}
xlmr = "FacebookAI/xlm-roberta-base"
mean_vectors_path = Path("results/mean_vectors")
results_path = Path("results/quantum_entropy_centered")
results_path.mkdir(exist_ok=True, parents=True)


def centered_uncertainty(text, embeddings, spacy_model, mean_vector):
    scorer = QuantumUncertainty(
        sentence=text,
        embeddings=embeddings,
        spacy_model=spacy_model,
        mean_vector=mean_vector,
    )
    return scorer.score()


for language, datasets in corpora.items():
    analyzer = HumorAnalyzer(
        language, embeddings_path=glove[language], transformer_model_name=xlmr
    )
    spacy_model = analyzer._load_model("spacy")
    # (column name, mean vector name, embedding model)
    models = [
        ("GloVe", "GloVe", analyzer._load_model("embeddings")),
        ("HF", "XLM-R", analyzer._load_model("transformer_pipeline")),
    ]

    for corpus in datasets:
        print("- " * 10 + corpus + " -" * 10)
        df = pd.read_json(paths[corpus], orient="index").reset_index()

        for column, mean_name, embeddings in models:
            mean_vector = torch.load(mean_vectors_path / f"{corpus}_{mean_name}.pt")
            tqdm.pandas(desc=f"Uncertainty + {column} (centered)")
            df[f"QE-U + {column} (centered)"] = df["text"].progress_apply(
                centered_uncertainty,
                embeddings=embeddings,
                spacy_model=spacy_model,
                mean_vector=mean_vector,
            )

        savepath = (results_path / corpus).with_suffix(".jsonl")
        df.to_json(savepath, orient="records", lines=True, force_ascii=False)
