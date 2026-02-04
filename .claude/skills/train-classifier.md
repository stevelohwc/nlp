---
skill: train-classifier
description: Train the text classification models with hyperparameter tuning
---

# Train Text Classification Model

Train and compare multiple text classification models with hyperparameter tuning.

## Steps

1. Check if training script exists at Code/text_classification/train.py
2. Verify dataset is available in Code/datasets/
3. Check if virtual environment is activated
4. Run training script with appropriate parameters
5. Monitor training progress
6. Report training results, metrics, and best model performance
7. Verify model files are saved to Code/models/

## Expected Behavior

- Train multiple models (e.g., Naive Bayes, SVM, Random Forest, Neural Networks)
- Perform hyperparameter tuning using grid search or random search
- Compare model performance
- Save best models for deployment
- Generate evaluation metrics (accuracy, precision, recall, F1-score)

## Notes

- Training may take significant time depending on dataset size
- Accept optional arguments for specific model types or hyperparameters
- Display progress and intermediate results
