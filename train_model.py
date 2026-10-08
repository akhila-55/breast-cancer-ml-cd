import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load your breast cancer dataset
data = pd.read_csv("breast_cancer_v1_100.csv")

# Remove unnamed columns if present
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert categorical target if necessary
if y.dtype == "object":
    y = y.astype("category").cat.codes

# Convert categorical features if present
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ML pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Save trained model
joblib.dump(model, "breast_cancer_model.pkl")

# Save metrics
metrics = {
    "accuracy": accuracy,
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model saved as breast_cancer_model.pkl")
print("Metrics saved as metrics.json")
