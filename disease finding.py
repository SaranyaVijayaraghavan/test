"""Disease prediction from structured medical data.

This script trains and compares classification models for medical datasets
such as Heart Disease, Diabetes, and Breast Cancer. It supports the local
UCI-style files in the `Input data` folder, datasets from the UCI ML Repository
through `ucimlrepo`, and local CSV files supplied by the user.

Examples:
    python "disease finding.py" --dataset breast_cancer
    python "disease finding.py" --dataset heart_disease
    python "disease finding.py" --dataset diabetes
    python "disease finding.py" --dataset diabetes --source csv --csv-path data/diabetes.csv --target-column Outcome
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:  # pragma: no cover - handled at runtime with a clear message
    fetch_ucirepo = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - XGBoost is optional
    XGBClassifier = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    uci_id: Optional[int]
    target_column: Optional[str]
    positive_if_greater_than_zero: bool = False


DATASETS: dict[str, DatasetConfig] = {
    "breast_cancer": DatasetConfig(
        name="Breast Cancer Recurrence",
        uci_id=17,
        target_column=None,
    ),
    "heart_disease": DatasetConfig(
        name="Heart Disease",
        uci_id=45,
        target_column=None,
        positive_if_greater_than_zero=True,
    ),
    "diabetes": DatasetConfig(
        name="Diabetes",
        uci_id=None,
        target_column="Outcome",
    ),
}

HEART_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

BREAST_CANCER_COLUMNS = [
    "class",
    "age",
    "menopause",
    "tumor_size",
    "inv_nodes",
    "node_caps",
    "deg_malig",
    "breast",
    "breast_quad",
    "irradiat",
]

GLUCOSE_CODES = {48, 57, 58, 59, 60, 61, 62, 63, 64}
INSULIN_CODES = {33: "regular_insulin", 34: "nph_insulin", 35: "ultralente_insulin"}
MEAL_CODES = {66, 67, 68}
EXERCISE_CODES = {69, 70, 71}


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with older and newer sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_uci_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, pd.Series]:
    if config.uci_id is None:
        raise ValueError(
            f"{config.name} does not have a default UCI id in this project yet. "
            "Run with --source csv and provide --csv-path plus --target-column."
        )
    if fetch_ucirepo is None:
        raise ImportError("Install ucimlrepo first: pip install ucimlrepo")

    dataset = fetch_ucirepo(id=config.uci_id)
    features = dataset.data.features.copy()
    target = dataset.data.targets.squeeze().copy()
    target.name = target.name or "target"
    return features, target


def load_csv_dataset(csv_path: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(csv_path)
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {', '.join(data.columns)}"
        )
    features = data.drop(columns=[target_column])
    target = data[target_column]
    return features, target


def load_local_breast_cancer(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    data_path = data_dir / "breast+cancer" / "breast-cancer.data"
    data = pd.read_csv(data_path, names=BREAST_CANCER_COLUMNS, na_values="?")
    return data.drop(columns=["class"]), data["class"]


def load_local_heart_disease(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    heart_dir = data_dir / "heart+disease"
    files = [
        heart_dir / "processed.cleveland.data",
        heart_dir / "processed.hungarian.data",
        heart_dir / "processed.switzerland.data",
        heart_dir / "processed.va.data",
    ]
    frames = [
        pd.read_csv(path, names=HEART_COLUMNS, na_values="?")
        for path in files
        if path.exists()
    ]
    if not frames:
        raise FileNotFoundError(f"No processed heart disease files found in {heart_dir}")

    data = pd.concat(frames, ignore_index=True)
    return data.drop(columns=["num"]), data["num"]


def read_diabetes_archive_member(archive_path: Path, member_name: str) -> str:
    completed = subprocess.run(
        ["tar", "-xOf", str(archive_path), member_name],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def glucose_status(value: float) -> str:
    if value < 70:
        return "low"
    if value > 180:
        return "high"
    return "normal"


def load_local_diabetes(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    archive_path = data_dir / "diabetes" / "diabetes-data.tar.Z"
    rows = []

    for patient_number in range(1, 71):
        member = f"Diabetes-Data/data-{patient_number:02d}"
        content = read_diabetes_archive_member(archive_path, member)
        patient_data = pd.read_csv(
            StringIO(content),
            sep="\t",
            names=["date", "time", "code", "value"],
            dtype={"date": str, "time": str, "code": int, "value": str},
        )
        patient_data["timestamp"] = pd.to_datetime(
            patient_data["date"] + " " + patient_data["time"],
            errors="coerce",
        )
        patient_data["numeric_value"] = pd.to_numeric(patient_data["value"], errors="coerce")
        patient_data = patient_data.dropna(subset=["timestamp"]).sort_values("timestamp")

        previous_glucose = np.nan
        insulin_state = {name: 0.0 for name in INSULIN_CODES.values()}
        day_events: dict[str, dict[str, int]] = {}

        for record in patient_data.itertuples(index=False):
            day_key = record.timestamp.date().isoformat()
            day_events.setdefault(day_key, {"meal_events": 0, "exercise_events": 0})

            if record.code in MEAL_CODES:
                day_events[day_key]["meal_events"] += 1
            elif record.code in EXERCISE_CODES:
                day_events[day_key]["exercise_events"] += 1
            elif record.code in INSULIN_CODES and pd.notna(record.numeric_value):
                insulin_state[INSULIN_CODES[record.code]] = float(record.numeric_value)
            elif record.code in GLUCOSE_CODES and pd.notna(record.numeric_value):
                glucose_value = float(record.numeric_value)
                rows.append(
                    {
                        "patient_id": patient_number,
                        "glucose_code": record.code,
                        "hour": record.timestamp.hour,
                        "minute_of_day": record.timestamp.hour * 60 + record.timestamp.minute,
                        "previous_glucose": previous_glucose,
                        "regular_insulin": insulin_state["regular_insulin"],
                        "nph_insulin": insulin_state["nph_insulin"],
                        "ultralente_insulin": insulin_state["ultralente_insulin"],
                        "meal_events_today": day_events[day_key]["meal_events"],
                        "exercise_events_today": day_events[day_key]["exercise_events"],
                        "glucose_status": glucose_status(glucose_value),
                    }
                )
                previous_glucose = glucose_value

    if not rows:
        raise ValueError(f"No glucose measurement rows found in {archive_path}")

    data = pd.DataFrame(rows)
    return data.drop(columns=["glucose_status"]), data["glucose_status"]


def load_local_dataset(dataset: str, data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    if dataset == "breast_cancer":
        return load_local_breast_cancer(data_dir)
    if dataset == "heart_disease":
        return load_local_heart_disease(data_dir)
    if dataset == "diabetes":
        return load_local_diabetes(data_dir)
    raise ValueError(f"Unsupported local dataset: {dataset}")


def normalize_target(y: pd.Series, positive_if_greater_than_zero: bool) -> tuple[np.ndarray, LabelEncoder]:
    y = y.replace("?", np.nan).dropna()
    if positive_if_greater_than_zero:
        numeric_y = pd.to_numeric(y, errors="coerce")
        y = (numeric_y > 0).astype(int)

    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(y.astype(str))
    return encoded, label_encoder


def align_features_and_target(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    valid_target_rows = y.replace("?", np.nan).dropna().index
    X = X.loc[valid_target_rows].copy()
    y = y.loc[valid_target_rows].copy()
    X = X.replace("?", np.nan)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def build_models(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=random_state),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=random_state),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
        )
    else:
        print("XGBoost is not installed, so the XGBoost model will be skipped.")

    return models


def evaluate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    results = []
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, classifier in build_models(random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        results.append({"model": model_name, "accuracy": accuracy})
        trained_pipelines[model_name] = pipeline

        print(f"\n{model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions, zero_division=0))

    results_frame = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    return results_frame, trained_pipelines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train disease prediction classifiers.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="breast_cancer",
        help="Dataset preset to use.",
    )
    parser.add_argument(
        "--source",
        choices=["local", "uci", "csv"],
        default="local",
        help="Load from Input data, UCI through ucimlrepo, or a local CSV file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Input data",
        help="Folder containing breast+cancer, heart+disease, and diabetes subfolders.",
    )
    parser.add_argument("--csv-path", type=Path, help="Path to a CSV dataset.")
    parser.add_argument("--target-column", help="Target column for CSV datasets.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-model", type=Path, help="Optional path to save the best model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DATASETS[args.dataset]

    if args.source == "local":
        X, raw_y = load_local_dataset(args.dataset, args.data_dir)
    elif args.source == "uci":
        X, raw_y = load_uci_dataset(config)
    else:
        target_column = args.target_column or config.target_column
        if args.csv_path is None or target_column is None:
            raise ValueError("CSV source requires --csv-path and --target-column.")
        X, raw_y = load_csv_dataset(args.csv_path, target_column)

    X, raw_y = align_features_and_target(X, raw_y)
    y, label_encoder = normalize_target(raw_y, config.positive_if_greater_than_zero)

    print(f"Dataset: {config.name}")
    print(f"Rows: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Target classes: {list(label_encoder.classes_)}")

    results, trained_pipelines = evaluate_models(
        X=X,
        y=y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\nModel comparison:")
    print(results.to_string(index=False, formatters={"accuracy": "{:.4f}".format}))

    best_model_name = results.iloc[0]["model"]
    if args.save_model:
        artifact = {
            "dataset": config.name,
            "model_name": best_model_name,
            "pipeline": trained_pipelines[best_model_name],
            "label_encoder": label_encoder,
        }
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.save_model)
        print(f"\nSaved best model '{best_model_name}' to {args.save_model}")


if __name__ == "__main__":
    main()
