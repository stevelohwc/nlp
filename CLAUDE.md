# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic NLP assignment for CT052-3-M-NLP with two main components:

1. **Spelling Correction System** - A probabilistic spelling error detection and correction system with GUI
2. **Text Classification Model** - A supervised learning model for tasks like sentiment analysis, spam detection, etc.

The project is implemented in Python and requires both Part A (group report, 60 marks) and Part B (individual demonstration, 40 marks).

## Repository Structure

```
nlp/
├── Code/                  # All implementation code
├── Documentations/        # Assignment specifications and marking schemes
└── CLAUDE.md             # This file
```

The Code/ directory should contain:
- Spelling correction system with GUI implementation
- Text classification model building scripts
- Dataset files and preprocessing utilities
- Any supporting libraries or utilities

## Assignment Requirements

### Part A: Spelling Correction System (30 marks)

**Core Requirements:**
- Corpus: Minimum 100,000 words from any scientific field
- Detect and correct both non-words and real-word contextual errors
- Implement Edit Distance algorithms with variations
- Implement Bigram models for context-aware correction
- Build GUI with text editor (500 character limit)
- Provide dictionary word list with search functionality
- Make misspelled words clickable with correction suggestions
- Display minimum edit distance for each suggestion

### Part A: Text Classification Model (30 marks)

**Tasks:**
- Select dataset (sentiment, spam, authorship, fake reviews, or cyberbullying)
- Perform exploratory data analysis (EDA)
- Build multiple supervised classification models
- Tune hyperparameters using grid search or random search
- Deploy model as standalone webpage
- Evaluate and compare with prior work

### Part B: Demonstration (40 marks)

**Requirements:**
- 10-15 minute screen cam recording
- Demonstrate both systems
- Discuss strengths, limitations, and improvements
- Cover POS tagging, Information Retrieval, and semantic improvements

## Development Guidelines

**Key Constraints:**
- Report must be 7,000 words in English
- Final deliverables: Report (.doc/.docx), Python code (.py/.ipynb), datasets, screen recording
- Submission deadline: 27.03.2026

**Implementation Focus:**
- The spelling correction GUI should handle exactly 500 characters
- Edit distance calculations must be shown to users
- Text classification requires model comparison and deployment
- Both systems need clear documentation for demonstration

## Architecture Notes

Since this is a new project with no existing code:

**Spelling Correction System:**
- Will need separate modules for: corpus processing, edit distance calculation, bigram model, dictionary management, GUI
- GUI framework selection (tkinter, PyQt, or web-based) should support clickable word highlighting
- Consider separating core NLP logic from presentation layer

**Text Classification Model:**
- Standard ML pipeline: data loading → preprocessing → feature extraction → model training → evaluation → deployment
- Multiple models required for comparison (e.g., Naive Bayes, SVM, Random Forest, Neural Networks)
- Web deployment suggests Flask/Django or Streamlit for the standalone webpage
- Hyperparameter tuning will require significant compute time

**Suggested Project Structure:**
```
Code/
├── spelling_correction/
│   ├── corpus.py          # Corpus loading and processing
│   ├── edit_distance.py   # Edit distance algorithms
│   ├── bigram_model.py    # Bigram language model
│   ├── dictionary.py      # Dictionary management
│   ├── corrector.py       # Main correction logic
│   └── gui.py             # GUI implementation
├── text_classification/
│   ├── data_loader.py     # Dataset loading
│   ├── preprocessing.py   # Text preprocessing
│   ├── models.py          # Model definitions
│   ├── train.py           # Training scripts
│   ├── evaluate.py        # Evaluation metrics
│   └── deploy.py          # Web deployment
├── datasets/              # Raw and processed datasets
├── models/                # Saved model files
└── requirements.txt       # Python dependencies
```

## Common Commands

Since no build system exists yet, commands will depend on the implementation approach chosen. Standard Python development practices apply:

```bash
# Install dependencies (once requirements.txt is created)
pip install -r requirements.txt

# Run spelling correction system (example)
python Code/spelling_correction/gui.py

# Train text classification model (example)
python Code/text_classification/train.py

# Deploy classification model (example)
python Code/text_classification/deploy.py

# Run tests (if test suite is created)
pytest Code/tests/
```

## Academic Context

This is coursework with strict requirements. Ensure all implementations:
- Meet the minimum corpus size (100,000 words)
- Include all required NLP techniques (Edit Distance, Bigrams, etc.)
- Provide proper evaluation metrics for classification models
- Are well-documented for the demonstration video
- Follow academic integrity guidelines
