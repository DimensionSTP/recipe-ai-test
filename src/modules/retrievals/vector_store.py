from typing import Dict, List, Any
import os

import numpy as np
import pandas as pd

import faiss


class FaissIndex:
    def __init__(
        self,
        data_path: str,
        indices_name: str,
        items_name: str,
        dim: int,
        retrieval_top_k: int,
        distance_column_name: str,
    ) -> None:
        self.data_path = data_path
        self.indices_name = indices_name
        self.items_name = items_name
        self.indices_path = os.path.join(
            self.data_path,
            self.indices_name,
        )
        self.items_path = os.path.join(
            self.data_path,
            self.items_name,
        )

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.df = None

        self.retrieval_top_k = retrieval_top_k
        self.distance_column_name = distance_column_name

    def add(
        self,
        embedding: np.ndarray,
    ) -> None:
        self.index.add(embedding)

    def search(
        self,
        query_embedding: np.ndarray,
    ) -> List[Dict[str, Any]]:
        distances, indices = self.index.search(
            query_embedding,
            k=self.retrieval_top_k,
        )

        candidates = []
        for idx, distance in zip(indices[0], distances[0]):
            row = self.df.iloc[idx].to_dict()
            row[self.distance_column_name] = float(distance)
            candidates.append(row)
        return candidates

    def save(self) -> None:
        faiss.write_index(
            self.index,
            self.indices_path,
        )

    def load(self) -> None:
        if not os.path.exists(self.indices_path):
            raise FileNotFoundError(f"Missing indices: {self.indices_path}")
        if not os.path.exists(self.items_path):
            raise FileNotFoundError(f"Missing items: {self.items_path}")

        index = faiss.read_index(self.indices_path)
        df = pd.read_csv(self.items_path)

        self.index = index
        self.df = df
