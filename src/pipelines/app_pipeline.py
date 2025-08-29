import streamlit as st

from omegaconf import DictConfig

from ..utils import SetUp


def pipeline(
    config: DictConfig,
) -> None:
    @st.cache_resource(show_spinner=True)
    def get_cached_manager(manager_type: str):
        setup = SetUp(config)
        manager = setup.get_manager(manager_type=manager_type)
        return manager

    recommendation_manager = get_cached_manager(manager_type="recommendation")

    st.title("Recipe AI Demo")

    lab_number_recommendation_mode = "Lab number based recommendation"
    ingredients_recommendation_mode = "Ingredients based recommendation"

    option = st.selectbox(
        "Please select the recommendation mode",
        [
            lab_number_recommendation_mode,
            ingredients_recommendation_mode,
        ],
    )

    if option == lab_number_recommendation_mode:
        lab_id = st.text_input("Enter the lab number to recommend for:")
        try:
            categories = (
                recommendation_manager.index.df[
                    recommendation_manager.category_column_name
                ]
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
                        recommendations = recommendation_manager.recommend(
                            input_value=lab_id,
                            input_type=config.input_mode.lab_id,
                            category_value=category_value,
                        )
                    except Exception as e:
                        st.error(f"Error during recommendation: {e}")
                    else:
                        st.subheader("Summary of AI recommendations")
                        if isinstance(recommendations, (dict, list)):
                            st.json(recommendations)
                        elif isinstance(recommendations, str) and (
                            "<br/>" in recommendations
                            or "<strong>" in recommendations
                            or "<p>" in recommendations
                        ):
                            st.markdown(
                                recommendations,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.write(recommendations)
    elif option == ingredients_recommendation_mode:
        st.write("Enter ingredients one per line:")
        ingredients_input = st.text_area(
            "Ingredients list",
            height=180,
            placeholder="A\nB\nC\n...",
        )

        try:
            categories = (
                recommendation_manager.index.df[
                    recommendation_manager.category_column_name
                ]
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
            if isinstance(category_value, str) and category_value.lower() == "all":
                category_value = None

            lines = [
                line.strip() for line in ingredients_input.split("\n") if line.strip()
            ]
            ingredients_query = "|".join(lines)

            with st.spinner("Recommendation in progress..."):
                try:
                    recommendations = recommendation_manager.recommend(
                        input_value=ingredients_query,
                        input_type=config.input_mode.ingredients,
                        category_value=category_value,
                    )
                except Exception as e:
                    st.error(f"Error during recommendation: {e}")
                else:
                    st.subheader("Summary of AI recommendations")
                    if isinstance(recommendations, (dict, list)):
                        st.json(recommendations)
                    elif isinstance(recommendations, str) and (
                        "<br/>" in recommendations
                        or "<strong>" in recommendations
                        or "<p>" in recommendations
                    ):
                        st.markdown(
                            recommendations,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write(recommendations)
    else:
        raise ValueError("Invalid input mode")
