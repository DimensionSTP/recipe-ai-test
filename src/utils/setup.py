import os

from omegaconf import DictConfig
from hydra.utils import instantiate

from ..modules.search_manager import SearchManager


class SetUp:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config

    def get_search_manager(self) -> SearchManager:
        search_manager: SearchManager = instantiate(
            self.config.search_manager,
        )
        return search_manager
