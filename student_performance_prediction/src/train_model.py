"""
train_model.py

Trains and compares multiple ML models for student pass/fail prediction.
Saves the best model + scaler + encoders to ../models/

Usage:
    python train_model.py
    python train_model.py --datasets student_data.csv
    python train_model.py --datasets student_data.csv "ai_student_impact_dataset (1).csv"
    python train_model.py --datasets all
"""

import argparse
import os
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from preprocess import load_data, prepare_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_PATH = DATA_DIR / "student_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def list_available_datasets():
    return sorted(DATA_DIR.glob("*.csv"))


def prompt_dataset_selection(available_paths):
    print("Multiple dataset files found in data/:")
    for idx, path in enumerate(available_paths, 1):
        print(f"  {idx}. {path.name}")
    selection = input(
        "Enter dataset numbers separated by commas, or type 'all' to use every dataset [all]: "
    ).strip()
    if not selection or selection.lower() == "all":
        return available_paths

    choices = [item.strip() for item in re.split(r"[\s,]+", selection) if item.strip()]
    selected_paths = []
    for item in choices:
        if item.isdigit():
            idx = int(item)
            if 1 <= idx <= len(available_paths):
                selected_paths.append(available_paths[idx - 1])
                continue
        raise ValueError(
            f"Invalid dataset selection '{item}'. Please choose numbers from 1 to {len(available_paths)}."
        )

    return list(dict.fromkeys(selected_paths))


def resolve_data_paths(dataset_args=None):
    available = list_available_datasets()
    if dataset_args:
        if len(dataset_args) == 1 and dataset_args[0].lower() in {"all", "*"}:
            if not available:
                raise FileNotFoundError(
                    f"No CSV dataset found in {DATA_DIR}. Run src/generate_data.py or add a CSV file there."
                )
            return available

        selected_paths = []
        for dataset_name in dataset_args:
            candidate = Path(dataset_name)
            if not candidate.is_absolute():
                candidate = DATA_DIR / dataset_name
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Dataset not found: {dataset_name}. Looked in {DATA_DIR} and as an absolute path."
                )
            selected_paths.append(candidate)
        return list(dict.fromkeys(selected_paths))

    if DEFAULT_DATA_PATH.exists() and len(available) == 1:
        return [DEFAULT_DATA_PATH]

    if not available:
        raise FileNotFoundError(
            f"No CSV dataset found in {DATA_DIR}. Run src/generate_data.py or add a CSV file there."
        )

    return prompt_dataset_selection(available)


def make_output_suffix(name):
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    return f"_{safe_name}" if safe_name else ""


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, preds


def train_dataset(df, dataset_label=None, file_suffix=""):
    if dataset_label:
        print(f"\n=== Training on dataset: {dataset_label} ===")
    else:
        print("\n=== Training dataset ===")

    X, y, encoders = prepare_features(df, target="passed", drop_grades=True)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        res, preds = evaluate_model(name, model, X_test_scaled, y_test)
        results.append(res)
        trained_models[name] = (model, preds)

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
    print("\n=== Model comparison ===")
    print(results_df.to_string(index=False))

    output_suffix = file_suffix or ""
    results_df.to_csv(OUTPUTS_DIR / f"model_comparison{output_suffix}.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model, best_preds = trained_models[best_name]
    print(f"\nBest model: {best_name}")

    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"],
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"confusion_matrix{output_suffix}.png")
    plt.close()

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False).head(15)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances.values, y=importances.index, color="steelblue")
        plt.title(f"Top 15 Feature Importances - {best_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / f"feature_importance{output_suffix}.png")
        plt.close()
        importances.to_csv(OUTPUTS_DIR / f"feature_importance{output_suffix}.csv")

    print("\nClassification report for best model:")
    print(classification_report(y_test, best_preds, target_names=["Fail", "Pass"]))

    joblib.dump(best_model, MODELS_DIR / f"best_model{output_suffix}.pkl")
    joblib.dump(scaler, MODELS_DIR / f"scaler{output_suffix}.pkl")
    joblib.dump(encoders, MODELS_DIR / f"encoders{output_suffix}.pkl")
    joblib.dump(feature_names, MODELS_DIR / f"feature_names{output_suffix}.pkl")
    joblib.dump(best_name, MODELS_DIR / f"best_model_name{output_suffix}.pkl")

    print(f"\nSaved best model ({best_name}) and preprocessing objects to {MODELS_DIR}/")
    print(f"Saved plots and comparison table to {OUTPUTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare models on selected student performance dataset(s)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help=(
            "Dataset filenames under data/ or absolute CSV paths. "
            "Use 'all' to load every CSV dataset in data/."
        ),
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available CSV datasets in data/ and exit.",
    )
    args = parser.parse_args()

    if args.list_datasets:
        available = list_available_datasets()
        if not available:
            print(f"No CSV datasets found in {DATA_DIR}.")
            return
        print("Available datasets:")
        for path in available:
            print(f" - {path.name}")
        return

    data_paths = resolve_data_paths(args.datasets)
    if len(data_paths) > 1:
        print("Loading selected datasets:")
        for path in data_paths:
            print(f" - {path}")

        dfs = [load_data(path) for path in data_paths]
        same_schema = all(set(df.columns) == set(dfs[0].columns) for df in dfs[1:])
        if same_schema:
            print("All selected datasets share the same schema. Combining them for training.")
            combined_df = pd.concat(dfs, ignore_index=True)
            train_dataset(combined_df, dataset_label="combined datasets", file_suffix="_combined")
        else:
            print(
                "Selected datasets have different schemas. Training separately on each file."
            )
            for path, df in zip(data_paths, dfs):
                suffix = make_output_suffix(path.stem)
                train_dataset(df, dataset_label=path.stem, file_suffix=suffix)
        return

    data_path = data_paths[0]
    print("Loading data...")
    print(f"Using dataset: {data_path}")
    df = load_data(data_path)
    train_dataset(df, dataset_label=data_path.stem)


if __name__ == "__main__":
    main()
