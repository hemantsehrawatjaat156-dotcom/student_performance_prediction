"""
app.py

Streamlit app to predict whether a student will pass or fail
based on the trained model.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    return model, scaler, encoders, feature_names, model_name


model, scaler, encoders, feature_names, model_name = load_artifacts()

st.title("🎓 Student Performance Predictor")
st.caption(f"Model in use: **{model_name}** (trained without G1/G2 grades, so it predicts purely from behavior & background)")

st.markdown("Fill in the student details below to predict whether they are likely to **pass** or **fail**.")

with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Address type", ["U", "R"], format_func=lambda x: "Urban" if x == "U" else "Rural")
        famsize = st.selectbox("Family size", ["LE3", "GT3"], format_func=lambda x: "<=3" if x == "LE3" else ">3")
        pstatus = st.selectbox("Parents' status", ["T", "A"], format_func=lambda x: "Living together" if x == "T" else "Apart")
        medu = st.slider("Mother's education (0=none, 4=higher)", 0, 4, 2)
        fedu = st.slider("Father's education (0=none, 4=higher)", 0, 4, 2)
        mjob = st.selectbox("Mother's job", ["teacher", "health", "services", "at_home", "other"])
        fjob = st.selectbox("Father's job", ["teacher", "health", "services", "at_home", "other"])
        reason = st.selectbox("Reason for choosing school", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
        traveltime = st.slider("Travel time to school (1=short, 4=long)", 1, 4, 1)
        studytime = st.slider("Weekly study time (1=low, 4=high)", 1, 4, 2)
        failures = st.slider("Past class failures", 0, 3, 0)

    with col2:
        schoolsup = st.selectbox("Extra school support", ["yes", "no"])
        famsup = st.selectbox("Family educational support", ["yes", "no"])
        paid = st.selectbox("Extra paid classes", ["yes", "no"])
        activities = st.selectbox("Extra-curricular activities", ["yes", "no"])
        nursery = st.selectbox("Attended nursery school", ["yes", "no"])
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        internet = st.selectbox("Internet access at home", ["yes", "no"])
        romantic = st.selectbox("In a relationship", ["yes", "no"])
        famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
        freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
        goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
        dalc = st.slider("Workday alcohol consumption (1-5)", 1, 5, 1)
        walc = st.slider("Weekend alcohol consumption (1-5)", 1, 5, 2)
        health = st.slider("Current health status (1-5)", 1, 5, 4)
        absences = st.slider("Number of school absences", 0, 30, 4)

    submitted = st.form_submit_button("Predict")

if submitted:
    raw_input = {
        "school": school, "sex": sex, "age": age, "address": address,
        "famsize": famsize, "Pstatus": pstatus, "Medu": medu, "Fedu": fedu,
        "Mjob": mjob, "Fjob": fjob, "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": dalc, "Walc": walc,
        "health": health, "absences": absences,
    }

    input_df = pd.DataFrame([raw_input])

    # apply the same label encoders used during training
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # ensure correct column order
    input_df = input_df[feature_names]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.divider()
    if prediction == 1:
        st.success(f"✅ Predicted result: **PASS** (confidence: {proba[1]*100:.1f}%)")
    else:
        st.error(f"⚠️ Predicted result: **FAIL** (confidence: {proba[0]*100:.1f}%)")

    st.progress(float(proba[1]))
    st.caption(f"Pass probability: {proba[1]*100:.1f}% | Fail probability: {proba[0]*100:.1f}%")

    st.info(
        "Note: This prediction is based on a synthetic dataset generated for this project "
        "and is for educational/demo purposes only, not a real academic assessment."
    )

st.divider()
st.caption("Built as a portfolio project — Student Performance Prediction")
