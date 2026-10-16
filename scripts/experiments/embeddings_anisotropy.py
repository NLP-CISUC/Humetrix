import random
from pathlib import Path

import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F

from humetrix import HumorAnalyzer, QuantumUncertainty

NUM_SAMPLE = 1000
NUM_PAIRS = 10_000
SEED = 42
random.seed(SEED)

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
results_path = Path("results/quantum_entropy")
results_path.mkdir(exist_ok=True, parents=True)

mean_vectors_path = Path("results/mean_vectors")
mean_vectors_path.mkdir(exist_ok=True, parents=True)


def setup_embeddings(texts, embedding_model, spacy_model):
    embeddings = []
    for text in texts:
        qeu = QuantumUncertainty(
            sentence=text,
            embeddings=embedding_model,
            spacy_model=spacy_model,
        )
        setup_toks, _ = qeu.tokenize_sentence()
        setup_embeddings = qeu.token_embeddings(setup_toks)
        if len(setup_embeddings) >= 2:
            embeddings.append(setup_embeddings.float())
    return embeddings


def intra_similarity(embeddings):
    means = []
    for sentence_embeddings in embeddings:
        normalized_embeds = F.normalize(sentence_embeddings)
        similarities = normalized_embeds @ normalized_embeds.T
        mask = torch.tril(torch.ones_like(similarities), diagonal=-1).bool()
        means.append(torch.mean(similarities[mask]).item())
    return sum(means) / len(means)


def inter_similarity(embeddings, n_pairs=NUM_PAIRS):
    total = 0.0
    for _ in range(n_pairs):
        sent1, sent2 = random.sample(embeddings, 2)
        emb1 = F.normalize(random.choice(sent1), dim=0)
        emb2 = F.normalize(random.choice(sent2), dim=0)
        total += (emb1 @ emb2).item()
    return total / n_pairs


def mean_embedding(embeddings):
    flatten = torch.cat(embeddings, dim=0)
    return torch.mean(flatten, dim=0)


rows = []
for language, datasets in corpora.items():
    analyzer = HumorAnalyzer(
        language, embeddings_path=glove[language], transformer_model_name=xlmr
    )
    embedding_models = {
        "GloVe": analyzer._load_model("embeddings"),
        "XLM-R": analyzer._load_model("transformer_pipeline"),
    }
    spacy_model = analyzer._load_model("spacy")

    for corpus in datasets:
        df = pd.read_json(paths[corpus], orient="index").reset_index()
        sample = df.sample(n=min(len(df), NUM_SAMPLE), random_state=SEED)

        for model_name in embedding_models:
            texts = sample["text"]
            embeddings = setup_embeddings(
                texts, embedding_models[model_name], spacy_model
            )

            intra, inter = intra_similarity(embeddings), inter_similarity(embeddings)
            rows.append(
                {
                    "corpus": corpus,
                    "embedding model": model_name,
                    "intrasentence similarity": intra,
                    "intersentence similarity": inter,
                    "intra-inter difference": intra - inter,
                }
            )

            mean_vector = mean_embedding(embeddings)
            torch.save(mean_vector, mean_vectors_path / f"{corpus}_{model_name}.pt")

            embeddings_centered = [X - mean_vector for X in embeddings]
            intra_centered, inter_centered = (
                intra_similarity(embeddings_centered),
                inter_similarity(embeddings_centered),
            )
            centered_difference = intra_centered - inter_centered
            rows.append(
                {
                    "corpus": corpus,
                    "embedding model": f"{model_name} (centered)",
                    "intrasentence similarity": intra_centered,
                    "intersentence similarity": inter_centered,
                    "intra-inter difference": centered_difference,
                }
            )


df = pl.from_records(rows)
df.write_ndjson("results/anisotropy.jsonl")
