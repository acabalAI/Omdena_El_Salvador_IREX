
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class FilterAgent:
    def __init__(self, client):
        self.client = client
        filter_agent_template = """ You are an agent with the task of counting in how many entries of a context {context},
        a text extract {headline} can be found identically and literally word by word.
        You will review each entry and see if the extract {headline} can be found exactly the same within each entry,
        not just similar semantically but word by word.
        
        your job is to generate a JSON structure with the number of entries where this happens:
        
              (
                "times": number of entries where the headline is found exactly and literally word by word,
              )
        """
        self.prompt_template = PromptTemplate(template=filter_agent_template, input_variables=["headline","context"])
        self.llm_chain = LLMChain(prompt=self.prompt_template, llm=self.client)

    def filter_context(self, headline, context):
        try:
            output = self.llm_chain.run({'headline': headline, 'context': context})
            return output
        except Exception as e:
            print(e)
            return "Error in filtering layer"