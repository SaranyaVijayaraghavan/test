# Disease Prediction from Medical Data

## Objective

Predict the possibility of diseases from structured patient data using common classification algorithms.

## Approach

The main script, `disease finding.py`, loads a medical dataset from the local `Input data` folder, preprocesses numeric and categorical features, trains multiple classifiers, and compares their evaluation metrics.

Supported models:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Supported local dataset presets:

- Breast Cancer recurrence data from `Input data/breast+cancer`
- Heart Disease processed data from `Input data/heart+disease`
- Diabetes patient event records from `Input data/diabetes/diabetes-data.tar.Z`

The script also supports UCI fetching through `ucimlrepo` and any structured CSV dataset when you provide the target column.

## Project Files

- `disease finding.py`: Main disease prediction training and evaluation script.
- `Input data/`: Local medical datasets used by the project.
- `requirements.txt`: Python packages required to run the project.

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run Breast Cancer recurrence prediction from the local files:

```bash
python "disease finding.py" --dataset breast_cancer
```

Run Heart Disease prediction from the local processed files:

```bash
python "disease finding.py" --dataset heart_disease
```

Run Diabetes glucose-status prediction from the local archive:

```bash
python "disease finding.py" --dataset diabetes
```

Run Diabetes prediction from a separate CSV file:

```bash
python "disease finding.py" --dataset diabetes --source csv --csv-path data/diabetes.csv --target-column Outcome
```

Run any custom CSV dataset:

```bash
python "disease finding.py" --dataset breast_cancer --source csv --csv-path data/my_dataset.csv --target-column diagnosis
```

Run a UCI dataset through `ucimlrepo`:

```bash
python "disease finding.py" --dataset heart_disease --source uci
```

Save the best-performing model:

```bash
python "disease finding.py" --dataset breast_cancer --save-model models/breast_cancer_best.joblib
```

## Dataset Notes

- Breast Cancer target: `no-recurrence-events` or `recurrence-events`.
- Heart Disease target: original values `0` to `4`, converted to binary where `0` means no disease and values greater than `0` mean disease present.
- Diabetes target: the AIM-94 diabetes archive contains event logs for patients who already have diabetes, not a disease/no-disease diagnosis table. The script converts glucose readings into a classification target: `low` if glucose is below 70, `normal` from 70 to 180, and `high` above 180.

## Output

For each model, the script prints:

- Accuracy
- Confusion matrix
- Precision, recall, and F1-score
- A final model comparison table sorted by accuracy

## Notes

- UCI datasets require internet access the first time they are fetched.
- If XGBoost is not installed, the script skips XGBoost and still runs the other three models.
- For Heart Disease, the original multi-class disease severity target is converted into a binary label: `0` means no disease, and values greater than `0` mean disease present.
