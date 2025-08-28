import streamlit as st

from omegaconf import DictConfig

from ..utils import SetUp


def pipeline(
    config: DictConfig,
) -> None:
    @st.cache_resource(show_spinner=True)
    def get_cached_manager():
        setup = SetUp(config)
        manager = setup.get_manager()
        return manager

    manager = get_cached_manager()

    st.title("Recipe AI Demo")

    option = st.selectbox(
        "Please select the recommendation mode",
        [
            "Lab number based recommendation",
        ],
    )

    if option == "Lab number based recommendation":
        lab_id = st.text_input("Enter the lab number to recommend for:")

        if st.button("recommend"):
            if not lab_id:
                st.warning("Enter the lab number to recommend for.")
            else:
                results = manager.recommend_and_summarize(lab_id=lab_id)
                st.subheader("Summary recommendation results")
                st.write(results)

    else:
        raise ValueError("Invalid search mode")
