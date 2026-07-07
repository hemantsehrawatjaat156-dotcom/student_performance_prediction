"""
preprocess.py

Loads the raw student dataset and prepares it for modeling:
- encodes categorical columns
- splits features/target
- optionally drops G1/G2 to avoid "leaking" the answer
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]


def load_data(path="../data/student_data.csv"):
    return pd.read_csv(path)


def encode_categoricals(df):
    """Label-encode yes/no and small categorical columns.
    Returns encoded df and a dict of fitted encoders (so the app can reuse them)."""
    df = df.copy()
    encoders = {}
    categorical_cols = [
        col for col in CATEGORICAL_COLS
        if col in df.columns and df[col].dtype == "object"
    ]
    categorical_cols.extend(
        col for col in df.select_dtypes(include=["object", "bool", "category"]).columns
        if col not in categorical_cols
    )

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def prepare_features(df, target="passed", drop_grades=True):
    """
    target: 'passed' for classification, 'G3' for regression
    drop_grades: if True, drops G1/G2 (and G3 if not the target) to avoid
                 the model 'cheating' by relying on earlier grades.
                 Set False if you want max accuracy using G1/G2 as predictors.
    """
    df = df.copy()
    if target == "passed" and target not in df.columns:
        if "Post_Semester_GPA" not in df.columns:
            raise ValueError(
                "No 'passed' column found and 'Post_Semester_GPA' is unavailable "
                "to derive a pass/fail target."
            )
        df[target] = (df["Post_Semester_GPA"] >= 2.0).astype(int)

    df_enc, encoders = encode_categoricals(df)

    grade_cols = ["G1", "G2", "G3"]
    if "Post_Semester_GPA" in df_enc.columns and target != "Post_Semester_GPA":
        grade_cols.append("Post_Semester_GPA")

    cols_to_drop = ["passed"] if target != "passed" else []
    cols_to_drop += [c for c in grade_cols if c != target]
    cols_to_drop += [c for c in ["Student_ID"] if c in df_enc.columns]
    cols_to_drop = [c for c in cols_to_drop if c in df_enc.columns]

    if drop_grades:
        # remove all grade columns except the target itself
        X = df_enc.drop(columns=cols_to_drop + ([target] if target in df_enc else []))
    else:
        X = df_enc.drop(columns=[target] if target in df_enc else [])
        X = X.drop(columns=[c for c in ["passed"] if c in X.columns and target != "passed"])

    y = df[target] if target == "passed" else df_enc[target]

    return X, y, encoders


if __name__ == "__main__":
    df = load_data()
    X, y, encoders = prepare_features(df, target="passed", drop_grades=True)
    print("Feature columns used:", list(X.columns))
    print("X shape:", X.shape, "y shape:", y.shape)
