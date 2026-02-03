# NLP Group Assignment - Part A

## Project Overview
This is a Natural Language Processing (NLP) group assignment project developed by up to three team members. The project focuses on implementing various NLP techniques for text analysis, preprocessing, and classification.

## Team Members
- Student 1: [Name]
- Student 2: Mohamed Said Aly
- Student 3: [Name]

## Project Structure
```
nlp/
├── src/                    # Source code
│   ├── preprocessing.py    # Text preprocessing utilities
│   ├── sentiment.py        # Sentiment analysis module
│   ├── classification.py   # Text classification module
│   └── utils.py           # Utility functions
├── data/                   # Data directory
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── notebooks/             # Jupyter notebooks for analysis
├── models/                # Saved models
├── tests/                 # Unit tests
├── examples/              # Example scripts
├── outputs/               # Output files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features
The project implements the following NLP functionalities:

### 1. Text Preprocessing
- Tokenization
- Lowercasing
- Punctuation removal
- Stop words removal
- Stemming and Lemmatization
- Text cleaning and normalization

### 2. Sentiment Analysis
- Sentiment classification (positive/negative/neutral)
- Polarity scoring
- Emotion detection

### 3. Text Classification
- Document classification
- Topic modeling
- Feature extraction (TF-IDF, Bag of Words)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/stevelohwc/nlp.git
cd nlp
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
```

## Usage

### Text Preprocessing
```python
from src.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
text = "This is an example text with punctuation!"
cleaned_text = preprocessor.clean_text(text)
tokens = preprocessor.tokenize(cleaned_text)
print(tokens)
```

### Sentiment Analysis
```python
from src.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
text = "I love this project! It's amazing."
sentiment = analyzer.analyze(text)
print(f"Sentiment: {sentiment['label']}, Score: {sentiment['score']}")
```

### Text Classification
```python
from src.classification import TextClassifier

classifier = TextClassifier()
# Train the classifier
classifier.train(train_texts, train_labels)
# Predict
predictions = classifier.predict(test_texts)
```

## Examples
Check the `examples/` directory for complete usage examples and the `notebooks/` directory for Jupyter notebooks demonstrating various NLP techniques.

## Running Tests
```bash
python -m pytest tests/
```

## Contributing
This is a group project. All team members should follow these guidelines:

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

3. Push to the repository:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request for review

## Development Guidelines
- Write clean, documented code
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Write unit tests for new features
- Update documentation as needed

## Assignment Details
- **Assessment Type**: Group Assignment (Part A)
- **Group Size**: Up to 3 members
- **Contribution**: Each member expected to contribute equally
- **Note**: Part B requires individual screen recording demonstration of this system

## Resources
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## License
This project is for educational purposes as part of an NLP course assignment.

## Contact
For questions or issues, please contact team members or create an issue in the repository.