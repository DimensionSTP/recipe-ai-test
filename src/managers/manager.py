from typing import Dict, List, Any, Optional

import pandas as pd

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
        category_name_column_name: str,
        target_column_name: str,
        amount_column_name: str,
        score_column_name: str,
        rerank_top_k: int,
        input_mode: Dict[str, int],
        is_table: bool,
    ) -> None:
        self.embedding = embedding
        self.reranker = reranker

        self.index = index
        self.index.load()

        self.lab_id_column_name = lab_id_column_name
        self.category_column_name = category_column_name
        self.category_name_column_name = category_name_column_name
        self.target_column_name = target_column_name
        self.amount_column_name = amount_column_name
        self.score_column_name = score_column_name

        self.rerank_top_k = rerank_top_k
        self.input_mode = input_mode
        self.is_table = is_table

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

    def recommend(
        self,
        input_value: str,
        input_type: str,
        category_value: Optional[str],
    ) -> Optional[List[Dict[str, Any]]]:
        if input_type == self.input_mode.lab_id:
            series = self.index.df[
                self.index.df[self.lab_id_column_name] == input_value
            ][self.target_column_name]
            if series.empty:
                return None
            query = series.iloc[0]
        elif input_type == self.input_mode.ingredients:
            query = input_value
        else:
            raise ValueError(
                f"Invalid input_type. Use {self.input_mode.lab_id} or {self.input_mode.ingredients}."
            )

        candidates = self.retrieve(query=query)
        if candidates is None:
            return None

        reranked_candidates = self.rerank(
            query=query,
            candidates=candidates,
            category_value=category_value,
        )
        if reranked_candidates is None:
            return None

        return reranked_candidates

    def create_html_table(
        self,
        reranked_candidates: List[Dict[str, Any]],
    ) -> str:
        if not reranked_candidates:
            return "<p>No candidates available.</p>"

        html_blocks = []
        for candidate in reranked_candidates:
            lab_id = str(candidate.get(self.lab_id_column_name))
            ingredients_str = str(candidate.get(self.target_column_name))
            amounts_str = str(candidate.get(self.amount_column_name))
            score_value = candidate.get(self.score_column_name)
            category_name_str = str(candidate.get(self.category_name_column_name))

            if pd.isna(ingredients_str) or pd.isna(amounts_str):
                html_blocks.append(
                    f"<strong>{lab_id}</strong><br/><p>No data available.</p>"
                )
                continue

            ingredients = str(ingredients_str).split("|")
            amounts = str(amounts_str).split("|")

            if len(ingredients) != len(amounts):
                html_blocks.append(
                    f"<strong>{lab_id}</strong><br/><p>Error: Mismatch between ingredient and amount counts.</p>"
                )
                continue

            table_data: List[Dict[str, Any]] = []
            for ingredient, amount in zip(ingredients, amounts):
                try:
                    amount_str = f"{float(amount):.4f}"
                except (ValueError, TypeError):
                    amount_str = str(amount)
                table_data.append(
                    {
                        "성분 명칭": ingredient,
                        "함량(%)": amount_str,
                    }
                )

            table_df = pd.DataFrame(table_data)
            html_table = (
                table_df.style.set_properties(**{"text-align": "left"})
                .set_table_styles(
                    [
                        {"selector": "th", "props": [("text-align", "left")]},
                        {"selector": "td", "props": [("text-align", "left")]},
                    ]
                )
                .hide(axis="index")
                .to_html()
            )

            if score_value is None or pd.isna(score_value):
                score_text = ""
            else:
                try:
                    score_text = f" (confidence score: {float(score_value):.3f})"
                except (ValueError, TypeError):
                    score_text = f" (confidence score: {score_value})"

            if category_name_str is None or pd.isna(category_name_str):
                category_text = "category: -"
            else:
                category_name = str(category_name_str).replace("|", ", ")
                category_text = f"category: {category_name}"

            html_block = (
                f"<strong>{lab_id}{score_text}</strong><br/>"
                f"<strong>{category_text}</strong><br/>"
                f"<details><summary>성분표 보기</summary>{html_table}</details>"
            )

            html_blocks.append(html_block)

        return "\n<br/><br/>\n".join(html_blocks)

    def recommend_and_summarize(
        self,
        input_value: str,
        input_type: str,
        category_value: Optional[str],
    ) -> str:
        reranked_candidates = self.recommend(
            input_value=input_value,
            input_type=input_type,
            category_value=category_value,
        )
        if reranked_candidates is None:
            return "No matching lab_id found."

        if self.is_table:
            return self.create_html_table(reranked_candidates)
        else:
            lines = [
                f"{i+1}. {reranked_candidate[self.lab_id_column_name]} (score: {reranked_candidate[self.score_column_name]:.3f})"
                for i, reranked_candidate in enumerate(reranked_candidates)
            ]
            return "\n".join(lines)
