from typing import Dict, List, Any, Optional

from ..models import VllmEmbedding, VllmReranker
from ..databases import FaissIndex


class RecommendationManager:
    def __init__(
        self,
        embedding: VllmEmbedding,
        reranker: VllmReranker,
        index: FaissIndex,
        lab_id_column_name: str,
        category_column_name: str,
        target_column_name: str,
        score_column_name: str,
        rerank_top_k: int,
    ) -> None:
        self.embedding = embedding
        self.reranker = reranker

        self.index = index
        self.index.load()

        self.lab_id_column_name = lab_id_column_name
        self.category_column_name = category_column_name
        self.target_column_name = target_column_name
        self.score_column_name = score_column_name

        self.rerank_top_k = rerank_top_k

    def retrieve(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedding(query=query)
        candidates = self.index.search(query_embedding=query_embedding)
        return candidates

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        category_value: Optional[str],
    ) -> Optional[List[Dict[str, Any]]]:
        if category_value is not None:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.get(self.category_column_name) == category_value
            ]

        if not candidates:
            return None

        target_candidates = [
            candidate[self.target_column_name] for candidate in candidates
        ]
        scores = self.reranker(
            query=query,
            candidates=target_candidates,
        )

        for candidate, score in zip(candidates, scores):
            candidate[self.score_column_name] = float(score)

        candidates.sort(
            key=lambda x: x[self.score_column_name],
            reverse=True,
        )
        return candidates[: self.rerank_top_k]

    def recommend_and_summarize(
        self,
        lab_id: str,
        category_value: Optional[str],
    ) -> str:
        query = self.index.df[self.index.df[self.lab_id_column_name] == lab_id][
            self.target_column_name
        ]
        candidates = self.retrieve(query=query)
        if candidates is None:
            return "No matching lab_id found."

        reranked_candidates = self.rerank(
            query=query,
            candidates=candidates,
            category_value=category_value,
        )
        if reranked_candidates is None:
            return "No matching lab_id found."

        lines = [
            f"{i+1}. {reranked_candidate[self.lab_id_column_name]} (score: {reranked_candidate[self.score_column_name]:.3f})"
            for i, reranked_candidate in enumerate(reranked_candidates)
        ]
        return "\n".join(lines)
