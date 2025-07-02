# Description

This PR introduces a new feature: the `train_model` function located in `src/classifier.py`.  
It provides a reusable training pipeline for binary classification using logistic regression, including cross-validation (via GridSearchCV), validation evaluation, and logging to a file.

This enhancement is designed to make the model training process modular, testable, and production-ready, with clear logging and validation logic.

Dependencies:  
- Uses `pandas` for data preprocessing and manipulation (may be replaced with `polars` in future iterations for performance improvements).
- Requires `scikit-learn` for training and persistence.
- Logging is written to `classifier.log`.

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [X] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)

## Related issues

This PR is a foundational part of the Titanic classification system.  
Additional changes may follow, such as saving predictions to CSV, extending model options, or exporting model performance summaries.

# How has this been tested?

- A `pytest` fixture was added to train the model using preprocessed [Titanic data from Kaggle](https://www.kaggle.com/competitions/titanic/data).
- A test case (`test_classifier`) verifies the following:
  - A valid model object is returned.
  - The model implements the `predict` method.
  - The model achieves at least 80% accuracy on a hold-out validation split.

**Steps to reproduce**:
- Download the Titanic dataset from [Kaggle](https://www.kaggle.com/competitions/titanic/data).
- Extract `train.csv` and `test.csv` into a `data/` directory at the root of the project.
- Ensure that the file paths in `test_classifier.py` point to `data/train.csv` and `data/test.csv`.
- Run tests with `poetry run pytest -v`.
- Verify that `classifier.log` is created and contains model training and evaluation details.
- Confirm that the dataset is preprocessed correctly and that feature consistency is maintained between training and test sets.

# Checklist

- [X] Code has been formatted
- [X] Existing tests passed
- [X] New tests passed and were accepted
- [x] Logging is correctly set up
- [X] Linting checks are clean
