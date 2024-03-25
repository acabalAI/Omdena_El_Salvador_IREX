# decision_agent.py

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class DecisionAgent:
    def __init__(self, client):
        self.client = client
        decision_agent_template = """you are information verification agent in 2024,
        You will be presented with a piece of news {news} and information gathered from the internet {filtered_context}.
        Your task is to evaluate whether the news is real or fake, based solely on:
        
        - How the {news} corresponds to the information retrieved {filtered_context}, considering the reliability of the sources.
        - Probability of the news {probability} being real.
        - Alignment of the headline and the news {alignment},Not aligment is a sign of fake news .
        - Number of times the exact headline is found in other media outlets {times} which could indicate a misinformation campaign.
        
        Based on these criteria provided in order of importance,
        produced a reasoned argumentation whether the news is Fake or real.
        You answer strictly as a single JSON string. Don't include any other verbose texts and don't include the markdown syntax anywhere.
        
          (
        "category": Fake or Real,
        "reasoning": Your reasoning here.
           )  
        provide your answers in Spanish
        """
        self.prompt_template = PromptTemplate(template=decision_agent_template, input_variables=["news","filtered_context","probability","alignment","times"])
        self.llm_chain = LLMChain(prompt=self.prompt_template, llm=self.client)

    def make_decision(self, news, filtered_context, probability, alignment, times):
        try:
            output = self.llm_chain.run({'news': news, 'filtered_context': filtered_context, 'probability': probability, 'alignment': alignment, 'times': times})
            return output
        except Exception as e:
            print(e)
            return "Error in decision layer"
