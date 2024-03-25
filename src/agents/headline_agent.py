from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class HeadlineAgent:
    def __init__(self, client):
        self.client = client
        headline_agent_template = """ You are an agent with the task of identifying the 
        whether the headline {headline} is aligned with
        the body of the news {news}.
        you will generate a Json output:
        
      (
        "label": Aligned or not Aligned,
      )

        """
        self.prompt_template = PromptTemplate(template=headline_agent_template, input_variables=["headline", "news"])
        self.llm_chain = LLMChain(prompt=self.prompt_template, llm=self.client)  # Assuming LLLMChain was a typo and should be LLMChain

    def analyze_alignment(self, headline, news):
        """
        Analyzes the alignment between a given headline and the body of the news.

        Parameters:
        - headline (str): The news headline.
        - news (str): The full text of the news article.

        Returns:
        - A dictionary with the analysis results, including whether the headline is aligned with the news body and any relevant analysis details.
        """
        try:
            output = self.llm_chain.run({'headline': headline, 'news': news})
            return output
        except Exception as e:
            print(e)
            return {"error": "Error in headline alignment analysis layer"}


