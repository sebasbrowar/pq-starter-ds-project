import pandas as pd
from pytest import fixture
from sklearn.model_selection import train_test_split

from src.classifier import train_model

# ==========================================
# Paths to dataset (adjust if necessary)
# ==========================================
PROJECT_PATH = 'C:/Users/sbrxb/OneDrive/Documentos/GitHub/pq-starter-ds-project'
TRAIN_PATH = PROJECT_PATH + '/data/train.csv'
TEST_PATH = PROJECT_PATH + '/data/test.csv'

# ==========================================
# Load datasets
# ==========================================
ds_train = pd.read_csv(TRAIN_PATH)
ds_test = pd.read_csv(TEST_PATH)

# ==========================================
# Preprocessing - Training Data
# ==========================================

# Drop columns that are not useful for modeling
ds_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing Age values with median grouped by Sex and Pclass
ds_train['Age'] = ds_train.groupby(['Sex', 'Pclass'])['Age'].transform(
    lambda x: x.fillna(x.median())
)

# Encode 'Sex' as binary: male -> 0, female -> 1
ds_train['Sex'] = ds_train['Sex'].map({'male': 0, 'female': 1})

# Fill missing 'Embarked' values with mode and map to numerical values
ds_train['Embarked'] = ds_train['Embarked'].fillna(ds_train['Embarked'].mode()[0])
ds_train['Embarked'] = ds_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# ==========================================
# Preprocessing - Test Data
# ==========================================

# Drop unused columns in test set as well
ds_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing Age values similarly to training data
ds_test['Age'] = ds_test.groupby(['Sex', 'Pclass'])['Age'].transform(
    lambda x: x.fillna(x.median())
)

# Encode 'Sex' as binary
ds_test['Sex'] = ds_test['Sex'].map({'male': 0, 'female': 1})

# Fill missing 'Fare' values using the median Fare from the training set
ds_test['Fare'] = ds_test['Fare'].fillna(ds_train['Fare'].median())

# Fill and map 'Embarked' values
ds_test['Embarked'] = ds_test['Embarked'].fillna(ds_test['Embarked'].mode()[0])
ds_test['Embarked'] = ds_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# ==========================================
# Feature selection
# ==========================================

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = ds_train[features]
y = ds_train['Survived']

# ==========================================
# Pytest Fixtures and Tests
# ==========================================

@fixture
def model():
    """Fixture that trains and returns the model using the training data."""
    return train_model(X, y)


def test_model_output(model):
    """Test that the returned model is valid and performs decently on validation."""
    # Ensure model is returned and has the expected interface
    assert model is not None
    assert hasattr(model, 'predict')

    # Check if accuracy is above a minimum threshold
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=20
    )
    accuracy = model.score(X_val, y_val)

    # Assert that the model reaches a reasonable accuracy threshold
    assert accuracy > 0.8, f"Model accuracy too low: {accuracy:.2f}"


