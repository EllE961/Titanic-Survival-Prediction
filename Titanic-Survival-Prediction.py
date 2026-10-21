from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data (Kaggle notebook path, or the local copy in data/)
DATA_DIR = Path("/kaggle/input/titanic")
if not DATA_DIR.exists():
    DATA_DIR = Path(__file__).resolve().parent / "data"
train_data = pd.read_csv(DATA_DIR / "train.csv")
test_data = pd.read_csv(DATA_DIR / "test.csv")

# Define features and target
features = ["Pclass", "Sex", "SibSp", "Parch"]
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
y_train = train_data["Survived"]

# Train the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Prepare submission
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")
