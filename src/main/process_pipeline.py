# text_process_pipeline.py

from text_utils import preprocess_text, tokenize_text
from model_pipeline import create_pipeline, train_model, evaluate_model
from info_extraction import fetch_information, summarize_information
from data_loading import load_and_preprocess_data
import pandas as pd

class TextProcessPipeline:
    def __init__(self, model_pipeline=None):
        """
        Initializes the TextProcessPipeline with an optional machine learning pipeline.
        
        Parameters:
            model_pipeline: A scikit-learn pipeline object for text classification (optional).
        """
        self.model_pipeline = model_pipeline if model_pipeline is not None else create_pipeline()

    def preprocess_and_tokenize(self, text: str) -> list:
        """
        Preprocesses and tokenizes the input text.
        
        Parameters:
            text (str): The text to be processed.
            
        Returns:
            list: A list of tokens extracted from the text.
        """
        preprocessed_text = preprocess_text(text)
        tokens = tokenize_text(preprocessed_text)
        return tokens

    def classify_text(self, text: str):
        """
        Classifies the input text using the machine learning model pipeline.
        
        Parameters:
            text (str): The text to classify.
            
        Returns:
            The output of the model pipeline's predict method.
        """
        return self.model_pipeline.predict([text])[0]

    def extract_and_summarize_information(self, query: str):
        """
        Extracts and summarizes information based on the given query.
        
        Parameters:
            query (str): The search query to fetch and summarize information for.
            
        Returns:
            A summary of the information extracted from the internet.
        """
        fetched_info = fetch_information(query)
        summarized_info = summarize_information(fetched_info)
        return summarized_info

# Example usage
if __name__ == "__main__":
    # Example data loading and model training
    df = pd.read_csv('your_dataset.csv')  # Adjust the path according to your dataset
    X_train, X_test, y_train, y_test = load_and_preprocess_data(df, ['text_column'], 'label_column')
    model_pipeline = create_pipeline()
    trained_pipeline = train_model(model_pipeline, X_train, y_train)
    evaluation_metrics = evaluate_model(trained_pipeline, X_test, y_test)
    print("Model Evaluation Metrics:", evaluation_metrics)
    
    # Initializing the text processing pipeline
    text_pipeline = TextProcessPipeline(model_pipeline=trained_pipeline)
    
    # Example text classification
    example_text = "This is an example text to classify."
    print("Classified as:", text_pipeline.classify_text(example_text))
    
    # Example information extraction and summarization
    query = "Latest tech trends 2023"
    print("Information Summary:", text_pipeline.extract_and_summarize_information(query))
