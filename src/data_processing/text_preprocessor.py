import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies text preprocessing.
    This includes converting text to lowercase, removing special characters,
    and optionally removing digits.
    """
    def __init__(self, remove_digits=True):
        """
        Initialize the TextPreprocessor.

        Parameters:
        remove_digits (bool): Whether to remove digits from the text. Defaults to True.
        """
        self.remove_digits = remove_digits

    def fit(self, X, y=None):
        """
        Fit method. Does nothing in this case as no fitting is required.

        Parameters:
        X: The data to fit.
        y: The target variables (ignored in this case).

        Returns:
        self: The instance itself.
        """
        return self

    def transform(self, X, y=None):
        """
        Apply the text preprocessing steps to the input data.

        Parameters:
        X: The data to transform. Can be a pandas Series or a list.
        y: The target variables (ignored in this case).

        Returns:
        transformed_X: The transformed version of X after text preprocessing.
        """
        if isinstance(X, pd.Series):
            return X.apply(lambda x: self.preprocess_text(x, self.remove_digits))
        else:  # Handle other types like list
            return [self.preprocess_text(x, self.remove_digits) for x in X]

    def preprocess_text(self, text, remove_digits=True):
        """
        Apply text preprocessing steps to a single text string.

        Parameters:
        text (str): The text to preprocess.
        remove_digits (bool): Whether to remove digits from the text. Defaults to True.

        Returns:
        processed_text (str): The processed text string.
        """
        text = re.sub(r'\W', ' ', str(text))
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        if remove_digits:
            text = re.sub(r'\d', ' ', text)
        return text.strip()
