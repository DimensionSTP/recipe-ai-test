from typing import Dict

from langchain import LLMChain, PromptTemplate
from langchain_openai import OpenAI
from langchain_community.utilities import SerpAPIWrapper


class SearchManager:
    def __init__(
        self,
        openai_api_key: str,
        serpapi_api_key: str,
        temperature: float,
        search_template: str,
        sorting_template: str,
        serpapi_params: Dict[str, str],
    ) -> None:
        self.openai_api_key = openai_api_key
        self.serpapi_api_key = serpapi_api_key
        self.temperature = temperature
        self.search_template = search_template
        self.sorting_template = sorting_template
        self.serpapi_params = serpapi_params

        self.llm = OpenAI(
            api_key=self.openai_api_key,
            temperature=self.temperature,
        )

        self.search = SerpAPIWrapper(
            serpapi_api_key=self.serpapi_api_key,
            params=self.serpapi_params,
        )

    def search_and_summarize(
        self,
        query: str,
    ) -> str:
        prompt = PromptTemplate(
            template=self.search_template,
            input_variables=["query"],
        )
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
        )

        search_results = self.search.run(query)
        return llm_chain.run(query=search_results)

    def refine_and_sort(
        self,
        raw_results: str,
    ) -> str:
        prompt = PromptTemplate(
            template=self.sorting_template,
            input_variables=["results"],
        )
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
        )
        return llm_chain.run(results=raw_results)
