import streamlit as st
from src.agents.class_agent import ClassAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.filter_agent import FilterAgent
from src.agents.headline_agent import HeadlineAgent
# Assuming load_pipeline is implemented to load the model
from src.ml_models.ml_pipeline import build_and_train_pipeline
from src.utils.info_extraction import *
from src.utils.token_controler import *

from dotenv import load_dotenv
import os
import pandas as pd
from langchain.llms import OpenAI
import json

# Load environment variables from .env file
load_dotenv()

# Access the API keys securely
open_ai_api_key = os.environ.get("OPENAI_API_KEY")
google_search_api_key = os.environ.get("GOOGLE_SEARCH_API_KEY")
client = OpenAI(api_key=open_ai_api_key, model='gpt-3.5-turbo-instruct')

# Initialize agents
class_agent = ClassAgent(client=client)
headline_agent = HeadlineAgent(client=client)
filter_agent = FilterAgent(client=client)
decision_agent = DecisionAgent(client=client)

# Streamlit UI Code
st.title("Analisis de Veracidad de Noticias ")
st.title("El Salvador")

# Sample input for demonstration
headline = st.text_input("Titular", "")
news = st.text_area("Cuerpo de la Noticia", "")

headline = limit_tokens(headline)
news = limit_tokens(news)

if st.button("Verificar"):
    # Begin verification process
    st.write("Verificando...")
    file_path_train = './src/data_processing/train_dataset.xlsx'

    df_train = pd.read_excel(file_path_train)
    df_train["total_text"] = df_train['Text']
    ml_pipeline = build_and_train_pipeline(df_train["total_text"], df_train["Category"])
    
    predicted_category = ml_pipeline.predict([news])  # Use the news directly or another appropriate variable
    predicted_probability = ml_pipeline.predict_proba([news])[0][1]
    st.write("Predicion basada en estilo:", predicted_category)
    st.write("Probabilidad de veracidad basada en el estilo:", predicted_probability)
    
    
    
    # Class agent
    class_result = class_agent.classify(headline=headline)
    st.write("Class result:", class_result)  # Modification here
    data = json.loads(class_result)
    subject = data["subject"]
    event = data["event"]
    


    # Info Extraction
    context = info_extraction(headline)
    st.write("Contexto-Fuentes utilizadas:", context)  # Modification here

    # Headline Alignment
    alignment_result = headline_agent.analyze_alignment(headline=headline, news=news)
    st.write("Alineamiento Titular-Noticia:", alignment_result)  # Modification here
    data_alignment = json.loads(alignment_result)
    alignment_label = data_alignment["label"]

    


    # Misinformation Campaign Filter
    filter_result = filter_agent.filter_context(headline, context)
    st.write("Difusion del Titular (times)", filter_result)  # Modification here
    filter_data = json.loads(filter_result)
    times = filter_data["times"]
    
    # Display Filter results

    # Decision Making Agent
    decision_result = decision_agent.make_decision(news, context, predicted_probability, alignment_label, times)
    
    # Final decision display
    st.write("Decision Final", decision_result)  # Modification here



