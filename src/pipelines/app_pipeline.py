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
        try:
            categories = (
                manager.index.df[manager.category_column_name]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            categories = sorted(categories)
        except Exception:
            categories = []
        category_options = ["ALL"] + categories
        category_value = st.selectbox(
            "Select a category (use 'ALL' to include all categories):",
            options=category_options,
            index=0,
        )

        recommend_column, reset_column = st.columns([1, 1])
        with recommend_column:
            run = st.button("recommend", type="primary")
        with reset_column:
            reset = st.button("reset")

        if reset:
            st.session_state.clear()
            st.rerun()

        if run:
            lab_id = lab_id.strip()
            category_value = category_value.strip()

            if not lab_id:
                st.warning("Enter the lab number to recommend for.")
            else:
                if isinstance(category_value, str) and category_value.lower() == "all":
                    category_value = None

                with st.spinner("Recommendation in progress..."):
                    try:
                        results = manager.recommend_and_summarize(
                            lab_id=lab_id,
                            category_value=category_value,
                        )
                    except Exception as e:
                        st.error(f"Error during recommendation: {e}")
                    else:
                        st.subheader("Summary recommendation results")
                        if isinstance(results, (dict, list)):
                            st.json(results)
                        elif isinstance(results, str) and (
                            "<br/>" in results
                            or "<strong>" in results
                            or "<p>" in results
                        ):
                            st.markdown(results, unsafe_allow_html=True)
                        else:
                            st.write(results)
    else:
        raise ValueError("Invalid search mode")
