import pandas as pd
import streamlit as st

from omegaconf import DictConfig

from ..utils.setup import SetUp


def pipeline(
    config: DictConfig,
) -> None:
    setup = SetUp(config)

    search_manager = setup.get_search_manager()

    st.title("Recipe AI Demo")

    option = st.selectbox(
        "Please select the search mode",
        [
            "Single-ingredient search",
            "Multi-ingredient search",
            "CSV upload-based search",
            "Table input-based search",
        ],
    )

    if option == "Single-ingredient search":
        ingredient = st.text_input("Enter the ingredients to search for:")
        category = st.text_input(
            "Enter product classification (e.g. skincare, moisturizing cream, etc.):"
        )

        if st.button("search"):
            if not ingredient or not category:
                st.warning("Enter ingredients and product classification.")
            else:
                search_query = f"{category} {ingredient}"
                search_results = search_manager.search_and_summarize(search_query)
                st.subheader("Summary search results")
                st.write(search_results)

    elif option == "Multi-ingredient search":
        ingredients = st.text_area(
            "Enter the ingredients you want to search for as a new line:"
        ).split("\n")
        category = st.text_input(
            "Enter product classification (e.g. skincare, moisturizing cream, etc.):"
        )

        if st.button("search"):
            if not ingredients or not category:
                st.warning("Enter ingredients and product classification.")
            else:
                search_query = f"{category} {' '.join(ingredients)}"
                search_results = search_manager.search_and_summarize(search_query)
                st.subheader("Summary search results")
                st.write(search_results)

    elif option == "CSV upload-based search":
        uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])
        category = st.text_input(
            "Enter product classification (e.g. skincare, moisturizing cream, etc.):"
        )
        if uploaded_file:
            input_data = pd.read_csv(uploaded_file)
            st.write("Uploaded data:")
            st.dataframe(input_data)

            if st.button("search"):
                ingredients = []
                for _, row in input_data.iterrows():
                    ingredient = str(row[config.table_column_name])
                    ingredients.append(ingredient)
                search_query = f"{category} {' '.join(ingredients)}"
                search_results = search_manager.search_and_summarize(search_query)
                st.subheader(f"Summary search results ({category})")
                st.write(search_results)

    elif option == "Table input-based search":
        st.write("Enter directly in the table below:")
        columns = [config.table_column_name]
        input_data = pd.DataFrame(columns=columns)

        st.write("Table of ingredients")
        input_data = st.data_editor(
            input_data,
            num_rows="dynamic",
            use_container_width=True,
        )

        category = st.text_input(
            "Enter product classification (e.g. skincare, moisturizing cream, etc.):"
        )

        if st.button("search"):
            ingredients = []
            for _, row in input_data.iterrows():
                ingredient = str(row[config.table_column_name])
                ingredients.append(ingredient)
            search_query = f"{category} {' '.join(ingredients)}"
            search_results = search_manager.search_and_summarize(search_query)
            st.subheader(f"Summary search results ({category})")
            st.write(search_results)

    else:
        raise ValueError("Invalid search mode")
