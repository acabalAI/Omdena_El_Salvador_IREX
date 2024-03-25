from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class ClassAgent:
    def __init__(self, client):
        self.client = client
        class_agent_template = """ You are an agent with the task of analysing a headline {headline} .
        you will identify the subject, the event, and the field the news belongs to either Politics, Economics, or Social.
        you will provide a JSON Structure:
          (
          "subject": subject of the news,
          "event": event described,
          "topic": field the news belongs to Politics, Economics, or Social
          )
        """
        self.prompt_template = PromptTemplate(template=class_agent_template, input_variables=["headline"])
        self.llm_chain = LLMChain(prompt=self.prompt_template, llm=self.client)

    def classify(self, headline):
        try:
            output = self.llm_chain.run({'headline': headline})
            return output
        except Exception as e:
            print(e)
            return "Error in classification layer"
