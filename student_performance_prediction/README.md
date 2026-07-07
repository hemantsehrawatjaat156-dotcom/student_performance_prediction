# Student Performance Prediction

A machine learning project that predicts whether a student will **pass or fail**
based on demographic, social, and study-related features — without using their
earlier grades (G1/G2) as a shortcut. This makes it a genuine prediction problem
rather than one that just copies prior grades forward.

## Project structure

```
student_performance_prediction/
├── data/
│   └── student_data.csv        # generated dataset (800 students)
├── src/
│   ├── generate_data.py        # creates the synthetic dataset
│   ├── preprocess.py           # cleaning + encoding logic
│   └── train_model.py          # trains & compares 4 models, saves the best one
├── models/                     # saved trained model + scaler + encoders (after running train_model.py)
├── outputs/                    # confusion matrix, feature importance, comparison table
├── app/
│   └── app.py                  # Streamlit app for live predictions
├── notebooks/                  # (optional) for exploratory analysis
├── requirements.txt
└── README.md
```

## Dataset

The dataset is synthetically generated (`src/generate_data.py`) but modeled closely
after the structure of the well-known UCI "Student Performance" dataset — same
feature set (school, family background, study time, failures, social habits, etc.)
and same grading scale (0–20, pass if G3 >= 10).

**Why synthetic instead of downloaded?** This keeps the project fully
self-contained and reproducible — anyone can regenerate the exact same data
without needing external downloads. You can swap in the real UCI dataset
(`student-mat.csv`) by dropping it into `data/` and adjusting the column names
if you'd rather use real-world data.

## Important design decision: dropping G1/G2

The dataset includes two earlier-period grades, G1 and G2, which are extremely
correlated with the final grade G3. Many tutorial projects use them as input
features and report ~95%+ accuracy — but that's barely a real prediction, since
G3 is just a slightly noisy version of G1/G2.

This project **drops G1 and G2** and predicts pass/fail using only behavioral
and demographic features (study time, failures, absences, family background,
etc.). This is a harder and more meaningful prediction task, with accuracy in
the realistic 80–85% range.

## How to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate the dataset:
   ```bash
   cd src
   python generate_data.py
   ```

3. Train and compare models:
   ```bash
   python train_model.py
   ```
   This prints accuracy/precision/recall/F1 for 4 models (Logistic Regression,
   Decision Tree, Random Forest, Gradient Boosting), saves the best one to
   `models/`, and saves plots + comparison table to `outputs/`.

4. Launch the interactive app:
   ```bash
   cd ../app
   streamlit run app.py
   ```
   Fill in a student's details in the form and get a live pass/fail prediction
   with confidence score.

## Results (on this generated dataset)

| Model               | Accuracy | Precision | Recall | F1   |
|----------------------|----------|-----------|--------|------|
| Random Forest         | ~0.84    | ~0.85     | ~0.97  | ~0.90|
| Logistic Regression    | ~0.83    | ~0.88     | ~0.90  | ~0.89|
| Gradient Boosting      | ~0.81    | ~0.87     | ~0.88  | ~0.88|
| Decision Tree          | ~0.80    | ~0.86     | ~0.89  | ~0.87|

Exact numbers depend on the random seed and will be saved in
`outputs/model_comparison.csv` after running `train_model.py`.

Feature importance analysis (see `outputs/feature_importance.png`) shows that
**past failures**, **study time**, and **absences** are typically the strongest
predictors — which matches real-world intuition about academic performance.

## Possible extensions

- Swap in the real UCI dataset for a true benchmark
- Add a regression mode to predict the exact final grade (G3) instead of pass/fail
- Try XGBoost/LightGBM for stronger performance
- Add SHAP values for per-prediction explainability
- Deploy the Streamlit app (Streamlit Community Cloud / Render) for a live demo link
