from omegaconf import DictConfig

from ..utils import SetUp


def pipeline(
    config: DictConfig,
) -> None:
    setup = SetUp(config)

    manager = setup.get_manager()

    print("Recipe AI Demo")

    option = "Lab number based recommendation"
    print(f"Recommendation mode: {option}")

    if option == "Lab number based recommendation":
        while True:
            lab_id = input(
                "Enter the lab number to recommend for (or 'q', 'quit', 'exit' to quit): "
            ).strip()

            if not lab_id:
                print("Warning: Enter the lab number to recommend for.")
                continue

            if lab_id.lower() in {"q", "quit", "exit"}:
                print("Exiting. Bye!")
                break

            category_value = input(
                "Enter a category to restrict by (press Enter to 'all' to include all categories, or 'q', 'quit', 'exit' to quit): "
            ).strip()

            if category_value.lower() in {"q", "quit", "exit"}:
                print("Exiting. Bye!")
                break

            if category_value.lower() == "all":
                category_value = None

            results = manager.recommend_and_summarize(
                lab_id=lab_id,
                category_value=category_value,
            )
            print("\nSummary recommendation results")
            print(results)

    else:
        raise ValueError("Invalid search mode")
