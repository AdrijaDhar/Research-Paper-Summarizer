"""Initialize node features with Qwen3-Embedding text embeddings.

Model choice and usage pattern (SentenceTransformer wrapper) verified against
Qwen3-Embedding-0.6B's own Hugging Face model card and sentence-transformers
integration docs, cross-checked against independent third-party write-ups
showing the same pattern. Used purely as a frozen feature extractor — its
weights are never updated; only the hypergraph encoder trains.
"""

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

TEXT_FIELD_BY_TYPE = {
    "paper": "title",
    "section": "title",
    "sentence": "text",
    "entity": "text",
    "table": "caption",
    "figure": "caption",
    "citation": "title",
}


def node_text(node: dict) -> str:
    preferred = TEXT_FIELD_BY_TYPE.get(node["type"])
    for field in (preferred, "text", "title", "caption", "original_caption"):
        value = node.get(field) if field else None
        if value:
            return str(value)
    authors = node.get("authors")
    if authors:
        return " ".join(authors)
    return node["type"]


def embed_nodes(nodes: list[dict], model: SentenceTransformer | None = None) -> torch.Tensor:
    model = model or SentenceTransformer(MODEL_NAME)
    texts = [node_text(n) for n in nodes]
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    return embeddings.float()
