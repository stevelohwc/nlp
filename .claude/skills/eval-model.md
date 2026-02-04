---
skill: eval-model
description: Evaluate text classification model performance with detailed metrics
---

# Evaluate Model Performance

Evaluate the trained text classification models and generate comprehensive performance reports.

## Steps

1. Check if evaluation script exists at Code/text_classification/evaluate.py
2. Verify trained models exist in Code/models/
3. Check if test dataset is available
4. Run evaluation on all trained models
5. Generate metrics:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrix
   - ROC curve and AUC (if applicable)
   - Classification report
6. Compare with prior work/benchmarks if available
7. Display results in formatted tables and visualizations
8. Save evaluation report

## Expected Output

- Performance metrics for each model
- Comparison table across models
- Confusion matrices
- Recommendations for best model
- Insights on model strengths and weaknesses

## Notes

- Compare multiple models if available
- Include visualizations (matplotlib/seaborn)
- Reference benchmark performance from literature
- Generate report suitable for academic submission
