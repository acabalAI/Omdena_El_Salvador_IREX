from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from src.data_processing.text_preprocessor import *
import joblib
from src.utils.info_extraction import *
from src.utils.token_controler import *
import pandas as pd
import os




def build_and_train_pipeline(X_train, y_train, remove_digits=True, max_features=5000, n_estimators=100):
    """
    Build the machine learning pipeline and train it on the provided dataset.

    Parameters:
    X_train: The training data (text).
    y_train: The target labels.
    remove_digits (bool): Whether to remove digits in the preprocessing step.
    max_features (int): The maximum number of features for TF-IDF vectorization.
    n_estimators (int): The number of trees in the random forest classifier.

    Returns:
    pipeline: The trained machine learning pipeline.
    """
    # Define the machine learning pipeline
    pipeline = make_pipeline(
        TextPreprocessor(remove_digits=remove_digits),
        TfidfVectorizer(max_features=max_features),
        RandomForestClassifier(n_estimators=n_estimators)
    )

    # Train the pipeline on the provided data
    pipeline.fit(X_train, y_train)

    return pipeline

def save_pipeline(pipeline, filename='ml_model_pipeline.joblib'):
    """
    Save the trained machine learning pipeline to a file.

    Parameters:
    pipeline: The trained machine learning pipeline.
    filename (str): The filename to save the pipeline to.

    Returns:
    None
    """
    joblib.dump(pipeline, filename)

def load_pipeline(filename='ml_model_pipeline.joblib'):
    """
    Load a machine learning pipeline from a file.

    Parameters:
    filename (str): The filename to load the pipeline from.

    Returns:
    pipeline: The loaded machine learning pipeline.
    """
    return joblib.load(filename)


def main():
    file_path_train = './src/data_processing/train_dataset.xlsx'

    df_train = pd.read_excel(file_path_train)
    df_train["total_text"] = df_train['Text']
    ml_pipeline = build_and_train_pipeline(df_train["total_text"], df_train["Category"])


    # Save the model
    model_folder = "saved_models"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_path = os.path.join(model_folder, "ml_model_pipeline.joblib")
    save_pipeline(ml_pipeline, filename=model_path)


if __name__ == "__main__":
    main()
