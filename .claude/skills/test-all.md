---
skill: test-all
description: Run all test suites for the NLP project
---

# Run All Tests

Execute all test suites for both spelling correction and text classification systems.

## Steps

1. Check if test files exist (pytest, unittest, or custom test scripts)
2. Verify virtual environment is activated
3. Check if test framework is installed (pytest recommended)
4. Run spelling correction tests if available
5. Run text classification tests if available
6. Display test results with coverage statistics
7. Report any failures with details

## Test Coverage

- **Spelling Correction Tests:**
  - Edit distance calculation
  - Bigram model functionality
  - Dictionary operations
  - Correction suggestion accuracy

- **Text Classification Tests:**
  - Data preprocessing
  - Model training pipeline
  - Prediction accuracy
  - Deployment functionality

## Expected Output

- Number of tests run
- Pass/fail count
- Coverage percentage
- Failed test details if any
- Performance metrics

## Notes

- If no tests exist, suggest creating test suite
- Use pytest if available, otherwise unittest
- Run with verbose output for detailed results
