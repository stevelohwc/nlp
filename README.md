# NLP Assignment - CT052-3-M-NLP

Academic coursework project implementing a spelling correction system and text classification model for natural language processing tasks.

## Project Overview

This repository contains two main components developed for the CT052-3-M-NLP course:

1. **Spelling Correction System** - A probabilistic error detection and correction system with GUI that handles both non-word and real-word contextual errors
2. **Text Classification Model** - A supervised learning model for tasks such as sentiment analysis, spam detection, or similar NLP classification problems

## Project Components

### Part A: Spelling Correction System (30 marks)

The spelling correction system implements:
- Corpus-based correction using minimum 100,000 words from scientific texts
- Edit Distance algorithms with variations for non-word error detection
- Bigram language models for context-aware real-word error correction
- Interactive GUI with:
  - Text editor supporting up to 500 characters
  - Dictionary word list with search functionality
  - Clickable misspelled words with correction suggestions
  - Minimum edit distance display for each suggestion

### Part A: Text Classification Model (30 marks)

The classification system includes:
- Dataset selection (sentiment, spam, authorship, fake reviews, or cyberbullying)
- Exploratory Data Analysis (EDA)
- Multiple supervised classification models with performance comparison
- Hyperparameter tuning using grid search or random search
- Model deployment as standalone webpage
- Evaluation against prior research benchmarks

### Part B: Demonstration (40 marks)

A 10-15 minute screen recording demonstrating:
- Both systems in operation
- Discussion of strengths and limitations
- Proposed improvements
- Coverage of POS tagging, Information Retrieval, and semantic enhancements

## Prerequisites

- **Python**: 3.8 or higher (tested on 3.12.2)
- **pip**: Python package manager
- **Virtual environment**: Recommended for dependency isolation
- **GUI framework**: TBD based on implementation (tkinter, PyQt, or web-based)
- **Web framework**: For model deployment (Flask, Django, or Streamlit)

## Installation & Setup

### 1. Navigate to Repository

```bash
cd nlp
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate    # On Windows
```

### 3. Install Dependencies

```bash
# Once requirements.txt is created
pip install -r requirements.txt
```

## Project Structure

```
nlp/
├── Code/                           # All implementation code
│   ├── spelling_correction/        # Spelling correction system
│   │   ├── corpus.py              # Corpus loading and processing
│   │   ├── edit_distance.py       # Edit distance algorithms
│   │   ├── bigram_model.py        # Bigram language model
│   │   ├── dictionary.py          # Dictionary management
│   │   ├── corrector.py           # Main correction logic
│   │   └── gui.py                 # GUI implementation
│   ├── text_classification/        # Text classification models
│   │   ├── data_loader.py         # Dataset loading
│   │   ├── preprocessing.py       # Text preprocessing
│   │   ├── models.py              # Model definitions
│   │   ├── train.py               # Training scripts
│   │   ├── evaluate.py            # Evaluation metrics
│   │   └── deploy.py              # Web deployment
│   ├── datasets/                   # Raw and processed datasets
│   └── models/                     # Saved model files
├── Documentations/                 # Assignment specifications
├── CLAUDE.md                       # Architecture and development guide
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## Quick Start

### Running Spelling Correction System

```bash
# Once implemented
python Code/spelling_correction/gui.py
```

### Training Text Classification Model

```bash
# Once implemented
python Code/text_classification/train.py
```

### Deploying Classification Model

```bash
# Once implemented
python Code/text_classification/deploy.py
```

## Development Workflow

### Key Requirements

- **Corpus Size**: Minimum 100,000 words from scientific domain
- **GUI Constraint**: Text editor limited to 500 characters
- **Model Comparison**: Multiple classification models required
- **Hyperparameter Tuning**: Grid search or random search implementation
- **Deployment**: Standalone webpage for classification model

### Implementation Phases

1. **Phase 1**: Spelling Correction Core
   - Collect and process corpus data
   - Implement edit distance algorithms
   - Build bigram language model
   - Create dictionary management system

2. **Phase 2**: Spelling Correction GUI
   - Design text editor interface
   - Implement word highlighting and click detection
   - Integrate correction suggestion display
   - Add edit distance visualization

3. **Phase 3**: Text Classification
   - Select and analyze dataset
   - Perform exploratory data analysis
   - Implement preprocessing pipeline
   - Train and compare multiple models

4. **Phase 4**: Model Deployment & Evaluation
   - Deploy classification model as web application
   - Evaluate against benchmarks
   - Document performance metrics

5. **Phase 5**: Demonstration
   - Prepare screen recording script
   - Record system demonstrations
   - Document limitations and improvements

## Assignment Deliverables

### Submission Requirements

- **Report**: 7,000 words in English (.doc or .docx format)
- **Code**: Python implementation files (.py or .ipynb)
- **Datasets**: All training and test data used
- **Screen Recording**: 10-15 minute demonstration video
- **Deadline**: 27.03.2026

### Report Structure

- Part A: Spelling Correction System (30 marks)
- Part A: Text Classification Model (30 marks)
- Part B: Demonstration and Discussion (40 marks)

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Detailed architecture, implementation guidelines, and development instructions
- **[Documentations/](Documentations/)** - Assignment specifications and marking schemes

## Next Steps

### Immediate Actions

1. **Environment Setup**
   - Create virtual environment
   - Generate requirements.txt with necessary dependencies
   - Set up Code/ directory structure

2. **Spelling Correction System**
   - Collect corpus data (100,000+ words from scientific texts)
   - Implement edit distance calculation algorithms
   - Build bigram language model
   - Develop GUI with clickable word highlighting

3. **Text Classification Model**
   - Select appropriate dataset (sentiment/spam/authorship/fake reviews/cyberbullying)
   - Perform exploratory data analysis
   - Implement multiple classification models
   - Compare model performance with hyperparameter tuning

4. **Integration & Deployment**
   - Deploy classification model as standalone webpage
   - Integrate both systems for demonstration
   - Prepare documentation for screen recording

### Recommended Setup

Create a `.gitignore` file for Python projects:
```
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
*.log
.DS_Store
```

Start with a basic `requirements.txt`:
```
numpy
pandas
scikit-learn
nltk
matplotlib
seaborn
jupyter
# Add GUI framework (e.g., tkinter, PyQt5)
# Add web framework (e.g., flask, streamlit)
```

## License

This is an academic project for CT052-3-M-NLP coursework. All work must adhere to academic integrity guidelines.

## Academic Context

This project is coursework with specific evaluation criteria. All implementations must:
- Meet minimum corpus requirements (100,000+ words)
- Include all required NLP techniques (Edit Distance, Bigrams, etc.)
- Provide comprehensive evaluation metrics
- Be well-documented for demonstration purposes
- Follow academic integrity and citation guidelines
