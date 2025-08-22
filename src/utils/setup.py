from typing import Union

from omegaconf import DictConfig
from hydra.utils import instantiate

from ..managers import RecommendationManager


class SetUp:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config

    def get_manager(self) -> Union[RecommendationManager, None]:
        manager: RecommendationManager = instantiate(
            self.config.manager[self.config.manager_type],
        )
        return manager
