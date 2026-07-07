"""
generate_data.py

Generates a synthetic but realistic student performance dataset,
similar in spirit to the UCI Student Performance dataset.

Run this first to create data/student_data.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 800  # number of students

def generate_dataset(n=N):
    school = np.random.choice(["GP", "MS"], n, p=[0.65, 0.35])
    sex = np.random.choice(["F", "M"], n)
    age = np.random.randint(15, 20, n)
    address = np.random.choice(["U", "R"], n, p=[0.7, 0.3])  # urban/rural
    famsize = np.random.choice(["LE3", "GT3"], n, p=[0.4, 0.6])
    pstatus = np.random.choice(["T", "A"], n, p=[0.85, 0.15])  # together/apart

    medu = np.random.randint(0, 5, n)  # mother education 0-4
    fedu = np.random.randint(0, 5, n)  # father education 0-4

    mjob = np.random.choice(["teacher", "health", "services", "at_home", "other"], n)
    fjob = np.random.choice(["teacher", "health", "services", "at_home", "other"], n)

    reason = np.random.choice(["home", "reputation", "course", "other"], n)
    guardian = np.random.choice(["mother", "father", "other"], n, p=[0.65, 0.25, 0.10])

    traveltime = np.random.randint(1, 5, n)
    studytime = np.random.randint(1, 5, n)
    failures = np.random.choice([0, 1, 2, 3], n, p=[0.7, 0.15, 0.1, 0.05])

    schoolsup = np.random.choice(["yes", "no"], n, p=[0.15, 0.85])
    famsup = np.random.choice(["yes", "no"], n, p=[0.6, 0.4])
    paid = np.random.choice(["yes", "no"], n, p=[0.4, 0.6])
    activities = np.random.choice(["yes", "no"], n, p=[0.5, 0.5])
    nursery = np.random.choice(["yes", "no"], n, p=[0.8, 0.2])
    higher = np.random.choice(["yes", "no"], n, p=[0.9, 0.1])
    internet = np.random.choice(["yes", "no"], n, p=[0.8, 0.2])
    romantic = np.random.choice(["yes", "no"], n, p=[0.35, 0.65])

    famrel = np.random.randint(1, 6, n)
    freetime = np.random.randint(1, 6, n)
    goout = np.random.randint(1, 6, n)
    dalc = np.random.randint(1, 6, n)  # workday alcohol
    walc = np.random.randint(1, 6, n)  # weekend alcohol
    health = np.random.randint(1, 6, n)
    absences = np.random.poisson(4, n).clip(0, 30)

    # ---- Build a realistic underlying score using a weighted formula + noise ----
    base = 9.5
    score = (
        base
        + studytime * 1.1
        - failures * 2.8
        - absences * 0.15
        + (medu + fedu) * 0.3
        + (higher == "yes") * 1.0
        + (internet == "yes") * 0.3
        + (schoolsup == "yes") * 0.2
        + (famsup == "yes") * 0.15
        - (romantic == "yes") * 0.3
        - (goout - 3) * 0.3
        - (dalc + walc - 4) * 0.25
        + (health - 3) * 0.15
        + np.random.normal(0, 3.2, n)  # more noise -> more realistic spread
    )

    g1 = (score + np.random.normal(0, 1.5, n)).clip(0, 20).round().astype(int)
    g2 = (score + np.random.normal(0, 1.2, n)).clip(0, 20).round().astype(int)
    g3 = (score + np.random.normal(0, 1.0, n)).clip(0, 20).round().astype(int)

    df = pd.DataFrame({
        "school": school, "sex": sex, "age": age, "address": address,
        "famsize": famsize, "Pstatus": pstatus, "Medu": medu, "Fedu": fedu,
        "Mjob": mjob, "Fjob": fjob, "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": dalc, "Walc": walc,
        "health": health, "absences": absences,
        "G1": g1, "G2": g2, "G3": g3,
    })

    # pass/fail label (>=10 out of 20 is a pass, matching the real UCI convention)
    df["passed"] = (df["G3"] >= 10).astype(int)

    return df


if __name__ == "__main__":
    df = generate_dataset()
    os.makedirs("../data", exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head())
