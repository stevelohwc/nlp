# Setup Guide for NLP Group Project

This guide will help you set up the NLP project on your local machine.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/stevelohwc/nlp.git
cd nlp
```

### 2. Create a Virtual Environment (Recommended)

Creating a virtual environment helps isolate project dependencies from your system Python.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt when the virtual environment is activated.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- NLTK for natural language processing
- scikit-learn for machine learning
- pandas and numpy for data manipulation
- matplotlib and seaborn for visualization
- pytest for testing

### 4. Download NLTK Data

Some NLTK features require additional data to be downloaded:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
```

Or you can run this Python script:

```python
import nltk

# Download required NLTK data
nltk.download('punkt')        # Tokenizer
nltk.download('stopwords')    # Stopwords list
nltk.download('wordnet')      # WordNet for lemmatization
nltk.download('vader_lexicon') # VADER for sentiment analysis
```

### 5. Verify Installation

Run the tests to verify everything is set up correctly:

```bash
python -m pytest tests/ -v
```

All tests should pass. If you encounter any errors, check that:
- All dependencies are installed
- NLTK data is downloaded
- You're using Python 3.8 or higher

### 6. Try the Examples

Run one of the example scripts to see the NLP functionality in action:

```bash
# Text preprocessing example
python examples/preprocessing_example.py

# Sentiment analysis example
python examples/sentiment_example.py

# Text classification example
python examples/classification_example.py
```

### 7. Launch Jupyter Notebook (Optional)

If you want to explore the Jupyter notebook:

```bash
jupyter notebook notebooks/nlp_demo.ipynb
```

This will open the notebook in your web browser.

## Project Structure

```
nlp/
├── src/                    # Source code
│   ├── __init__.py
│   ├── preprocessing.py    # Text preprocessing
│   ├── sentiment.py        # Sentiment analysis
│   ├── classification.py   # Text classification
│   └── utils.py           # Utilities
├── tests/                 # Unit tests
│   ├── test_preprocessing.py
│   ├── test_sentiment.py
│   └── test_classification.py
├── examples/              # Example scripts
│   ├── preprocessing_example.py
│   ├── sentiment_example.py
│   └── classification_example.py
├── notebooks/             # Jupyter notebooks
│   └── nlp_demo.ipynb
├── data/                  # Data directory
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
├── models/               # Saved models
├── outputs/              # Output files
├── requirements.txt      # Dependencies
├── README.md            # Project documentation
├── CONTRIBUTING.md      # Contribution guidelines
└── SETUP.md            # This file
```

## Common Issues and Solutions

### Issue: `ModuleNotFoundError: No module named 'nltk'`
**Solution:** Make sure you've activated your virtual environment and installed dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: NLTK data not found errors
**Solution:** Download the required NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
```

### Issue: Import errors when running examples
**Solution:** Make sure you're running from the project root directory:
```bash
cd /path/to/nlp
python examples/preprocessing_example.py
```

### Issue: Test failures
**Solution:** Ensure all NLTK data is downloaded and you have the correct Python version (3.8+).

## Development Workflow

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and test them:
   ```bash
   python -m pytest tests/ -v
   ```

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

4. **Push to GitHub**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub for review.

## Getting Help

If you encounter issues:
1. Check this setup guide
2. Review the README.md and CONTRIBUTING.md
3. Ask your team members
4. Create an issue on GitHub

## Next Steps

- Read the README.md for project overview
- Check CONTRIBUTING.md for development guidelines
- Explore the example scripts in `examples/`
- Try the Jupyter notebook in `notebooks/`
- Start contributing to the project!

## Updating Dependencies

If new dependencies are added to the project:

```bash
git pull origin main
pip install -r requirements.txt
```

## Deactivating Virtual Environment

When you're done working on the project:

```bash
deactivate
```

Good luck with the project!
