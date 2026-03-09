# UI Robustness (Streamlit)

## Startup checks
- Requires `models/best_model_bundle.pkl`.
- Validates required bundle keys.
- Validates scikit-learn major/minor compatibility with current environment.

## UX structure
- Tab 1: Single Prediction
- Tab 2: Benchmark Validation (holdout split)
- Tab 3: Live URL Validation

## Input validation
- Empty text blocked
- Minimum and maximum text length checks
- URL format validation (`http/https`)

## Error handling
- Graceful errors for:
  - missing artifacts
  - incompatible versions
  - fetch/extraction failures
  - inference exceptions
- Runtime errors logged to `reports/app_runtime.log`

## Caution policy
All predictions include a caution statement:
- classifier output is a signal, not authoritative fact verification.
