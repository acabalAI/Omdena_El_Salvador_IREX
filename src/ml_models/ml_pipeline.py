from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from src.data_processing.text_preprocessor import TextPreprocessor
import joblib

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
