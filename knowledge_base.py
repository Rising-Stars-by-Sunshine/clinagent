import math
from collections import Counter
from typing import List, Tuple

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None


PSYCHIATRIC_REFERENCES = [
    {
        "id": "phq8_interest",
        "text": "Anhedonia reflects diminished interest or pleasure in activities and is scored according to symptom frequency over the assessment period.",
    },
    {
        "id": "phq8_mood",
        "text": "Depressed mood includes feeling down, hopeless, sad, or emotionally low, with severity determined by persistence and frequency.",
    },
    {
        "id": "phq8_sleep",
        "text": "Sleep disturbance includes insomnia, hypersomnia, early awakening, fragmented sleep, or difficulty maintaining restorative sleep.",
    },
    {
        "id": "phq8_energy",
        "text": "Fatigue or low energy should be scored when the transcript indicates persistent tiredness, lack of strength, or reduced capacity for daily activities.",
    },
    {
        "id": "phq8_appetite",
        "text": "Appetite change includes reduced appetite, overeating, weight-related eating changes, or clinically relevant shifts in food intake.",
    },
    {
        "id": "phq8_self_esteem",
        "text": "Low self-esteem includes guilt, worthlessness, self-blame, perceived failure, or negative self-evaluation.",
    },
    {
        "id": "phq8_concentration",
        "text": "Concentration difficulty includes trouble focusing, indecisiveness, slowed thinking, or difficulty sustaining attention.",
    },
    {
        "id": "phq8_movement",
        "text": "Psychomotor symptoms include observable slowing, restlessness, agitation, or behavioral changes noticed by others.",
    },
]


def _tokenize(text: str) -> List[str]:
    return [
        token.strip(".,;:!?()[]{}\"'").lower()
        for token in text.split()
        if token.strip(".,;:!?()[]{}\"'")
    ]


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _lexical_score(query: str, document: str) -> float:
    query_counts = Counter(_tokenize(query))
    document_counts = Counter(_tokenize(document))
    shared_terms = set(query_counts).intersection(document_counts)
    return float(sum(query_counts[term] * document_counts[term] for term in shared_terms))


class KnowledgeRetriever:
    def __init__(self, top_k: int = 3, embedding_model: str = "nomic-embed-text") -> None:
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.embedding_client = OllamaEmbeddings(model=embedding_model) if OllamaEmbeddings else None

    def retrieve(self, query: str) -> Tuple[str, List[dict]]:
        ranked_references = self._semantic_rank(query)
        if not ranked_references:
            ranked_references = self._lexical_rank(query)

        selected = ranked_references[: self.top_k]
        context = "\n".join(f"- {item['text']}" for item in selected)
        return context, selected

    def _semantic_rank(self, query: str) -> List[dict]:
        if self.embedding_client is None:
            return []

        try:
            query_embedding = self.embedding_client.embed_query(query)
            document_embeddings = self.embedding_client.embed_documents(
                [item["text"] for item in PSYCHIATRIC_REFERENCES]
            )
        except Exception:
            return []

        scored = []
        for reference, embedding in zip(PSYCHIATRIC_REFERENCES, document_embeddings):
            item = dict(reference)
            item["retrieval_score"] = _cosine_similarity(query_embedding, embedding)
            item["retrieval_method"] = self.embedding_model
            scored.append(item)

        return sorted(scored, key=lambda item: item["retrieval_score"], reverse=True)

    def _lexical_rank(self, query: str) -> List[dict]:
        scored = []
        for reference in PSYCHIATRIC_REFERENCES:
            item = dict(reference)
            item["retrieval_score"] = _lexical_score(query, reference["text"])
            item["retrieval_method"] = "lexical_fallback"
            scored.append(item)

        return sorted(scored, key=lambda item: item["retrieval_score"], reverse=True)
