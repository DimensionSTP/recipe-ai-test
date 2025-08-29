from ..models import VllmGenerator


class ReportManager:
    def __init__(
        self,
        generator: VllmGenerator,
        is_table: bool,
    ) -> None:
        self.generator = generator
        self.is_table = is_table

    def generate(
        self,
        recommendation: str,
        is_table: bool,
    ) -> str:
        report = self.generator(
            recommendations=recommendation,
            is_table=is_table,
        )
        return report
