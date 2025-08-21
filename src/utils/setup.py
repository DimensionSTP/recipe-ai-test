from omegaconf import DictConfig
from hydra.utils import instantiate

from ..modules.manager import RecommendationManager


class SetUp:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config

    def get_recommendation_manager(self) -> RecommendationManager:
        recommendation_manager: RecommendationManager = instantiate(
            self.config.recommendation_manager,
        )
        return recommendation_manager
