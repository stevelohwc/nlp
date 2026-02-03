"""
Text Classification Module
Provides text classification capabilities using various machine learning algorithms.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


class TextClassifier:
    """
    A class for text classification using machine learning.
    """

    def __init__(self, classifier_type: str = 'naive_bayes',
                 vectorizer_type: str = 'tfidf',
                 max_features: int = 5000):
        """
        Initialize the TextClassifier.

        Args:
            classifier_type (str): Type of classifier ('naive_bayes', 'logistic', 'svm').
            vectorizer_type (str): Type of vectorizer ('tfidf' or 'count').
            max_features (int): Maximum number of features for vectorization.
        """
        self.classifier_type = classifier_type
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        
        # Initialize vectorizer
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features)
        else:
            self.vectorizer = CountVectorizer(max_features=max_features)
        
        # Initialize classifier
        if classifier_type == 'naive_bayes':
            self.classifier = MultinomialNB()
        elif classifier_type == 'logistic':
            self.classifier = LogisticRegression(max_iter=1000)
        elif classifier_type == 'svm':
            self.classifier = SVC(kernel='linear', probability=True)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        self.is_trained = False

    def train(self, texts: List[str], labels: List[any],
              test_size: float = 0.2) -> Dict[str, any]:
        """
        Train the text classifier.

        Args:
            texts (List[str]): List of text documents.
            labels (List[any]): List of corresponding labels.
            test_size (float): Proportion of data to use for testing.

        Returns:
            Dict: Training results including accuracy and classification report.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        
        # Vectorize texts
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_vec, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'test_size': len(X_test)
        }

    def predict(self, texts: List[str]) -> List[any]:
        """
        Predict labels for new texts.

        Args:
            texts (List[str]): List of text documents to classify.

        Returns:
            List: Predicted labels.
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction.")
        
        X_vec = self.vectorizer.transform(texts)
        return self.classifier.predict(X_vec)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probability distribution over labels for new texts.

        Args:
            texts (List[str]): List of text documents to classify.

        Returns:
            np.ndarray: Probability distributions.
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction.")
        
        X_vec = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X_vec)

    def get_top_features(self, label: any = None, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top features (words) for a given label.

        Args:
            label (any): Label to get features for (required for logistic/svm).
            n (int): Number of top features to return.

        Returns:
            List[Tuple[str, float]]: List of (feature, importance) tuples.
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained first.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        if self.classifier_type == 'naive_bayes':
            # For Naive Bayes, use feature log probabilities
            if label is not None:
                label_idx = list(self.classifier.classes_).index(label)
                importances = self.classifier.feature_log_prob_[label_idx]
            else:
                importances = self.classifier.feature_log_prob_[0]
        else:
            # For linear classifiers, use coefficients
            if label is not None:
                label_idx = list(self.classifier.classes_).index(label)
                importances = self.classifier.coef_[label_idx]
            else:
                importances = self.classifier.coef_[0]
        
        # Get top n features
        top_indices = np.argsort(importances)[-n:][::-1]
        return [(feature_names[i], importances[i]) for i in top_indices]

    def save_model(self, filepath: str):
        """
        Save the trained model to a file.

        Args:
            filepath (str): Path to save the model.
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'classifier_type': self.classifier_type,
            'vectorizer_type': self.vectorizer_type
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str):
        """
        Load a trained model from a file.

        Args:
            filepath (str): Path to the saved model.
        """
        model_data = joblib.load(filepath)
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.classifier_type = model_data['classifier_type']
        self.vectorizer_type = model_data['vectorizer_type']
        self.is_trained = True


if __name__ == "__main__":
    # Example usage
    print("Text Classification Example\n")
    
    # Sample data
    texts = [
        "I love this movie, it's fantastic!",
        "This movie is terrible and boring.",
        "Great film, highly recommended!",
        "Waste of time, very disappointing.",
        "Amazing story and great acting.",
        "Poor quality, not worth watching.",
        "Excellent movie, loved every minute!",
        "Bad script and poor direction.",
    ]
    
    labels = ['positive', 'negative', 'positive', 'negative',
              'positive', 'negative', 'positive', 'negative']
    
    # Create and train classifier
    classifier = TextClassifier(classifier_type='naive_bayes')
    results = classifier.train(texts, labels, test_size=0.25)
    
    print(f"Training completed!")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"\nClassification Report:\n{results['classification_report']}")
    
    # Test predictions
    test_texts = [
        "This is a wonderful film!",
        "I didn't like this movie at all."
    ]
    
    predictions = classifier.predict(test_texts)
    probabilities = classifier.predict_proba(test_texts)
    
    print("\nPredictions:")
    for text, pred, proba in zip(test_texts, predictions, probabilities):
        print(f"\nText: {text}")
        print(f"Prediction: {pred}")
        print(f"Confidence: {max(proba):.3f}")
