import dotenv

dotenv.load_dotenv(
    override=True,
)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from ..modules.retrievals import FaissIndex
from ..modules.models import VllmEmbedding


@hydra.main(
    config_path="../../configs/",
    config_name="main.yaml",
)
def set_vector_store(
    config: DictConfig,
) -> None:
    index: FaissIndex = instantiate(
        config.vector_store,
    )
    embedding: VllmEmbedding = instantiate(
        config.model.embedding,
    )

    queries = index.df[index.target_column_name].tolist()
    embeddings = [embedding(query=query) for query in queries]
    index.add(embeddings=embeddings)
    index.save()


if __name__ == "__main__":
    set_vector_store()
