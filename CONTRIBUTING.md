# Contributing to NLP Group Project

Thank you for contributing to this NLP group assignment! This document provides guidelines for contributing to the project.

## Team Collaboration

This is a group assignment where each member is expected to contribute equally. Please communicate with your team members regularly.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/stevelohwc/nlp.git
   cd nlp
   ```

2. **Set up your development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a branch for your work**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### 1. Before Making Changes

- Pull the latest changes from the main branch
- Make sure all existing tests pass
- Communicate with team members about what you're working on

### 2. Making Changes

- Write clean, well-documented code
- Follow Python PEP 8 style guidelines
- Add docstrings to functions and classes
- Include type hints where appropriate
- Update documentation if needed

### 3. Testing Your Changes

- Test your code thoroughly before committing
- Run existing tests to ensure you didn't break anything
- Add new tests for new functionality

### 4. Committing Changes

- Write clear, descriptive commit messages
- Use present tense ("Add feature" not "Added feature")
- Reference issues if applicable

Example:
```bash
git add .
git commit -m "Add sentiment analysis feature"
```

### 5. Pushing Changes

```bash
git push origin feature/your-feature-name
```

### 6. Creating a Pull Request

- Create a pull request from your branch to the main branch
- Describe what changes you made and why
- Request review from team members
- Address any feedback from reviewers

## Code Style Guidelines

### Python Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Keep functions focused and small
- Add comments for complex logic

Example:
```python
def preprocess_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Preprocess text for NLP tasks.
    
    Args:
        text (str): Input text to preprocess.
        remove_stopwords (bool): Whether to remove stopwords.
    
    Returns:
        List[str]: Preprocessed tokens.
    """
    # Implementation here
    pass
```

### Documentation

- Add docstrings to all public functions and classes
- Update README.md when adding new features
- Include usage examples in documentation

## Project Structure

When adding new code, place it in the appropriate directory:

- `src/` - Core functionality modules
- `examples/` - Example scripts demonstrating usage
- `notebooks/` - Jupyter notebooks for analysis
- `tests/` - Unit tests
- `data/` - Data files (not committed to repo)
- `models/` - Saved models (not committed to repo)

## Testing

### Running Tests

```bash
python -m pytest tests/
```

### Writing Tests

- Write tests for new functionality
- Place tests in the `tests/` directory
- Use descriptive test names

Example:
```python
def test_text_preprocessing():
    preprocessor = TextPreprocessor()
    text = "Hello World!"
    result = preprocessor.clean_text(text)
    assert result == "hello world"
```

## Common Tasks

### Adding a New Feature

1. Create a branch: `git checkout -b feature/feature-name`
2. Implement the feature in the appropriate module
3. Add tests for the feature
4. Update documentation
5. Create a pull request

### Fixing a Bug

1. Create a branch: `git checkout -b fix/bug-description`
2. Fix the bug
3. Add a test that would have caught the bug
4. Create a pull request

### Adding Documentation

1. Update relevant files (README.md, docstrings, etc.)
2. Ensure examples are clear and working
3. Create a pull request

## Communication

- Use GitHub issues to track tasks and bugs
- Communicate regularly with team members
- Ask questions if you're unsure about something
- Review each other's pull requests

## Assignment Requirements

Remember that this is Part A of a group assignment:
- Each member should contribute equally
- Document your contributions
- Part B will require individual demonstration of this system

## Questions?

If you have questions or need help:
1. Check the existing documentation
2. Ask your team members
3. Create an issue in the repository

## Resources

- [Python PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

Thank you for contributing to this project!
