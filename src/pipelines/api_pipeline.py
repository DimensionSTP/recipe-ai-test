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
        lab_id = input("Enter the lab number to recommend for: ")

        if not lab_id:
            print("Warning: Enter the lab number to recommend for.")
        else:
            results = manager.recommend_and_summarize(lab_id=lab_id)
            print("\nSummary recommendation results")
            print(results)

    else:
        raise ValueError("Invalid search mode")
