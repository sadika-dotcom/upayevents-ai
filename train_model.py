
import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# 1. Load the existing dataset
data = pd.read_csv("synthetic_data.csv")

# 2. Print the dataset shape
print("Dataset shape:", data.shape)

# 3. Separate features and target
X = data.drop(columns=["checked_in"])
y = data["checked_in"]

# Convert event_type into numerical format
X = pd.get_dummies(X, columns=["event_type"], dtype=int)

# Save the feature column names
feature_columns = X.columns.tolist()

# 4. Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 5. Train the XGBoost classifier
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed.")

# 6. Calculate and print ROC-AUC
y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)

print(f"\nTest ROC-AUC Score: {roc_auc:.4f}")

# 7. Print the top 5 feature importances
feature_importances = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

feature_importances = feature_importances.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 5 Feature Importances:")
print(feature_importances.head(5).to_string(index=False))

# 8. Save the trained model
joblib.dump(model, "model.pkl")
print("\nModel saved to model.pkl")

# 9. Save the feature column names
with open("feature_columns.json", "w") as file:
    json.dump(feature_columns, file, indent=4)

print("Feature columns saved to feature_columns.json")
