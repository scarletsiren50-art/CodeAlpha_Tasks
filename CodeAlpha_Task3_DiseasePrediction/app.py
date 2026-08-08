from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "heart_disease_model.pkl"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ==========================================================
# HEADER
# ==========================================================

st.title("❤️ Heart Disease Risk Prediction")

st.write(
    """
    Enter the patient's health information below to obtain
    a machine-learning estimate of heart-disease risk.
    """
)

st.warning(
    "This application is an educational machine-learning "
    "project and does not provide a medical diagnosis. "
    "Do not use its prediction as a substitute for "
    "professional medical evaluation."
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("🤖 Model Information")

st.sidebar.write(
    """
    **Dataset:** UCI Cleveland Heart Disease

    **Final Model:** Support Vector Machine (SVM)

    **Patient Records:** 297

    **Input Features:** 13

    **Test Accuracy:** 88.33%

    **ROC-AUC:** 0.9487
    """
)

st.sidebar.subheader("Models Compared")

st.sidebar.write(
    """
    • Logistic Regression  
    • Random Forest  
    • Support Vector Machine  
    • XGBoost
    """
)


# ==========================================================
# INPUT FORM
# ==========================================================

st.header("Patient Information")

with st.form("heart_disease_form"):

    col1, col2, col3 = st.columns(3)

    # ------------------------------------------------------
    # COLUMN 1
    # ------------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=54,
            step=1,
        )

        sex_label = st.selectbox(
            "Sex",
            [
                "Female",
                "Male",
            ],
        )

        chest_pain_label = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic",
            ],
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=70,
            max_value=250,
            value=130,
            step=1,
        )

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=80,
            max_value=700,
            value=240,
            step=1,
        )

    # ------------------------------------------------------
    # COLUMN 2
    # ------------------------------------------------------

    with col2:

        fasting_bs_label = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [
                "No",
                "Yes",
            ],
        )

        resting_ecg_label = st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "ST-T Wave Abnormality",
                "Left Ventricular Hypertrophy",
            ],
        )

        max_heart_rate = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=60,
            max_value=220,
            value=150,
            step=1,
        )

        exercise_angina_label = st.selectbox(
            "Exercise-Induced Angina",
            [
                "No",
                "Yes",
            ],
        )

        st_depression = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

    # ------------------------------------------------------
    # COLUMN 3
    # ------------------------------------------------------

    with col3:

        st_slope_label = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            [
                "Upsloping",
                "Flat",
                "Downsloping",
            ],
        )

        major_vessels = st.selectbox(
            "Number of Major Vessels",
            [
                0,
                1,
                2,
                3,
            ],
        )

        thalassemia_label = st.selectbox(
            "Thalassemia",
            [
                "Normal",
                "Fixed Defect",
                "Reversible Defect",
            ],
        )


    submitted = st.form_submit_button(
        "🔍 Analyze Heart Disease Risk",
        type="primary",
        use_container_width=True,
    )


# ==========================================================
# CONVERT USER-FRIENDLY VALUES TO UCI CODES
# ==========================================================

if submitted:

    sex_mapping = {
        "Female": 0,
        "Male": 1,
    }

    chest_pain_mapping = {
        "Typical Angina": 1,
        "Atypical Angina": 2,
        "Non-anginal Pain": 3,
        "Asymptomatic": 4,
    }

    fasting_bs_mapping = {
        "No": 0,
        "Yes": 1,
    }

    resting_ecg_mapping = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2,
    }

    exercise_angina_mapping = {
        "No": 0,
        "Yes": 1,
    }

    st_slope_mapping = {
        "Upsloping": 1,
        "Flat": 2,
        "Downsloping": 3,
    }

    thalassemia_mapping = {
        "Normal": 3,
        "Fixed Defect": 6,
        "Reversible Defect": 7,
    }


    # ======================================================
    # CREATE INPUT DATAFRAME
    # ======================================================

    patient_data = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex_mapping[sex_label],
                "chest_pain_type":
                    chest_pain_mapping[chest_pain_label],
                "resting_bp": resting_bp,
                "cholesterol": cholesterol,
                "fasting_blood_sugar":
                    fasting_bs_mapping[fasting_bs_label],
                "resting_ecg":
                    resting_ecg_mapping[resting_ecg_label],
                "max_heart_rate": max_heart_rate,
                "exercise_angina":
                    exercise_angina_mapping[
                        exercise_angina_label
                    ],
                "st_depression": st_depression,
                "st_slope":
                    st_slope_mapping[st_slope_label],
                "major_vessels": major_vessels,
                "thalassemia":
                    thalassemia_mapping[
                        thalassemia_label
                    ],
            }
        ]
    )


    # ======================================================
    # PREDICTION
    # ======================================================

    with st.spinner(
        "Analyzing patient information..."
    ):

        prediction = model.predict(
            patient_data
        )[0]

        probabilities = model.predict_proba(
            patient_data
        )[0]


    no_disease_probability = (
        probabilities[0] * 100
    )

    disease_probability = (
        probabilities[1] * 100
    )


    # ======================================================
    # RESULTS
    # ======================================================

    st.divider()

    st.header("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Model Prediction: "
            "Heart Disease Pattern Detected"
        )

        st.write(
            "The entered values resemble patterns that "
            "the trained model associated with the "
            "heart-disease class."
        )

    else:

        st.success(
            "✅ Model Prediction: "
            "No Heart Disease Pattern Detected"
        )

        st.write(
            "The entered values resemble patterns that "
            "the trained model associated with the "
            "no-heart-disease class."
        )


    # ======================================================
    # PROBABILITIES
    # ======================================================

    st.subheader(
        "Model Probabilities"
    )


    probability_col1, probability_col2 = (
        st.columns(2)
    )


    with probability_col1:

        st.metric(
            "No Heart Disease",
            f"{no_disease_probability:.2f}%",
        )


    with probability_col2:

        st.metric(
            "Heart Disease",
            f"{disease_probability:.2f}%",
        )


    st.write("**No Heart Disease**")

    st.progress(
        int(
            round(
                no_disease_probability
            )
        )
    )


    st.write("**Heart Disease**")

    st.progress(
        int(
            round(
                disease_probability
            )
        )
    )


    # ======================================================
    # INPUT SUMMARY
    # ======================================================

    with st.expander(
        "📋 View Submitted Patient Data"
    ):

        display_data = pd.DataFrame(
            {
                "Feature": [
                    "Age",
                    "Sex",
                    "Chest Pain Type",
                    "Resting Blood Pressure",
                    "Cholesterol",
                    "Fasting Blood Sugar > 120",
                    "Resting ECG",
                    "Maximum Heart Rate",
                    "Exercise-Induced Angina",
                    "ST Depression",
                    "ST Slope",
                    "Major Vessels",
                    "Thalassemia",
                ],
                "Value": [
                    age,
                    sex_label,
                    chest_pain_label,
                    f"{resting_bp} mm Hg",
                    f"{cholesterol} mg/dL",
                    fasting_bs_label,
                    resting_ecg_label,
                    max_heart_rate,
                    exercise_angina_label,
                    st_depression,
                    st_slope_label,
                    major_vessels,
                    thalassemia_label,
                ],
            }
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
        )


    st.info(
        "The displayed probabilities are outputs from the "
        "trained machine-learning model. They are not "
        "clinical probabilities and should not be "
        "interpreted as a diagnosis or medical risk score."
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Heart Disease Prediction System | "
    "CodeAlpha Machine Learning Internship Project"
)