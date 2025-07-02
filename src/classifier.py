import logging

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

# Set up a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("classifier.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def train_model(X, y):
    """
    Trains a logistic regression model using GridSearchCV
    and evaluates it on a validation set.

    Parameters:
    - X: pd.DataFrame, feature matrix
    - y: pd.Series or np.array, target labels

    Returns:
    - best_model: Trained sklearn LogisticRegression model with best hyperparameters
    """
    # Split the dataset into training and validation sets
    logger.info("Splitting data...")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=10
    )

    # Define hyperparameter grid for GridSearchCV
    logger.info("Starting GridSearchCV for LogisticRegression...")

    param_grid = {
        # Inverse of regularization strength
        # Smaller values mean stronger regularization
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        # Norm used in the penalization, 'l2' is Ridge-like
        'penalty': ['l2'],
        # Algorithm used to optimize the cost function
        'solver': ['liblinear', 'lbfgs'],
        # Maximum number of iterations taken for the solvers to converge
        'max_iter': [500, 1000, 2000]
    }

    # Perform grid search with 5-fold cross-validation
    grid = GridSearchCV(
        LogisticRegression(class_weight='balanced'),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit the model on the training set
    grid.fit(X_train, y_train)

    # Log best parameters and CV score
    logger.info(f"Best parameters found: {grid.best_params_}")
    logger.info(f"Best cross-validation score: {grid.best_score_:.4f}")

    # Get the best model from grid search
    best_model = grid.best_estimator_

    # Make predictions on the validation set
    y_val_preds = best_model.predict(X_val)

    # Log evaluation metrics
    logger.info("Evaluation on validation set:")
    logger.info(f"Accuracy: {accuracy_score(y_val, y_val_preds):.4f}")
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_val, y_val_preds))
    logger.info("Classification report:\n%s", classification_report(y_val, y_val_preds))

    return best_model


