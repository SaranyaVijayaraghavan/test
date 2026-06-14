cd "C:/Users/Saranya/OneDrive/Documents/GitHub/test"
pip install ucimlrepo
pip install shap xgb
reticulate::install_miniconda()oost
from ucimlrepo import fetch_ucirepo

# Fetch Breast Cancer Wisconsin (Diagnostic) dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# Features and target
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# Display basic information
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nFirst 5 rows of features:")
print(X.head())

print("\nFirst 5 rows of target:")
print(y.head())
# Metadata
print(breast_cancer_wisconsin_diagnostic.metadata)

# Variable descriptions
print(breast_cancer_wisconsin_diagnostic.variables)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Convert target labels
y = y.squeeze()  # DataFrame -> Series

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=5000)

model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

for name, model in models.items():

    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
