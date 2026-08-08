import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_model.pkl"


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ==========================================================
# HEADER
# ==========================================================

st.title("💳 Credit Risk Prediction System")

st.write(
    """
    This machine learning application predicts whether an applicant
    represents a **Good Credit Risk** or **Bad Credit Risk** based on
    financial and personal information.
    """
)

st.info(
    "The prediction is generated using a machine learning model "
    "trained on the German Credit dataset."
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("About")

st.sidebar.write(
    """
    **Machine Learning Internship Project**

    This application demonstrates a credit scoring system using
    classification algorithms.

    Models evaluated:

    • Logistic Regression  
    • Decision Tree  
    • Random Forest
    """
)

st.sidebar.warning(
    "Educational project only. "
    "This application should not be used for real financial decisions."
)


# ==========================================================
# INPUT SECTION
# ==========================================================

st.header("Applicant Information")

st.write(
    "Enter the applicant's financial information below."
)


col1, col2 = st.columns(2)


# ==========================================================
# COLUMN 1
# ==========================================================

with col1:

    checking_account = st.selectbox(
        "Checking Account Status",
        options=[
            "A11",
            "A12",
            "A13",
            "A14",
        ],
        format_func=lambda x: {
            "A11": "Less than 0 DM",
            "A12": "0 to 200 DM",
            "A13": "200 DM or more",
            "A14": "No checking account",
        }[x],
    )

    duration = st.slider(
        "Loan Duration (Months)",
        min_value=4,
        max_value=72,
        value=24,
    )

    credit_history = st.selectbox(
        "Credit History",
        options=[
            "A30",
            "A31",
            "A32",
            "A33",
            "A34",
        ],
        format_func=lambda x: {
            "A30": "No credits / all credits paid",
            "A31": "All credits at this bank paid",
            "A32": "Existing credits paid correctly",
            "A33": "Delay in paying past credits",
            "A34": "Critical account / other credits",
        }[x],
    )

    purpose = st.selectbox(
        "Loan Purpose",
        options=[
            "A40",
            "A41",
            "A42",
            "A43",
            "A44",
            "A45",
            "A46",
            "A48",
            "A49",
            "A410",
        ],
        format_func=lambda x: {
            "A40": "Car (New)",
            "A41": "Car (Used)",
            "A42": "Furniture / Equipment",
            "A43": "Radio / Television",
            "A44": "Domestic Appliances",
            "A45": "Repairs",
            "A46": "Education",
            "A48": "Retraining",
            "A49": "Business",
            "A410": "Other",
        }[x],
    )

    credit_amount = st.number_input(
        "Credit Amount (DM)",
        min_value=250,
        max_value=20000,
        value=3000,
        step=100,
    )

    savings_account = st.selectbox(
        "Savings Account",
        options=[
            "A61",
            "A62",
            "A63",
            "A64",
            "A65",
        ],
        format_func=lambda x: {
            "A61": "Less than 100 DM",
            "A62": "100 to 500 DM",
            "A63": "500 to 1000 DM",
            "A64": "1000 DM or more",
            "A65": "Unknown / No savings",
        }[x],
    )

    employment_since = st.selectbox(
        "Employment Duration",
        options=[
            "A71",
            "A72",
            "A73",
            "A74",
            "A75",
        ],
        format_func=lambda x: {
            "A71": "Unemployed",
            "A72": "Less than 1 year",
            "A73": "1 to 4 years",
            "A74": "4 to 7 years",
            "A75": "7 years or more",
        }[x],
    )

    installment_rate = st.slider(
        "Installment Rate (% of Income)",
        min_value=1,
        max_value=4,
        value=2,
    )

    personal_status_sex = st.selectbox(
        "Personal Status",
        options=[
            "A91",
            "A92",
            "A93",
            "A94",
        ],
        format_func=lambda x: {
            "A91": "Male - Divorced/Separated",
            "A92": "Female - Divorced/Separated/Married",
            "A93": "Male - Single",
            "A94": "Male - Married/Widowed",
        }[x],
    )

    other_debtors = st.selectbox(
        "Other Debtors / Guarantors",
        options=[
            "A101",
            "A102",
            "A103",
        ],
        format_func=lambda x: {
            "A101": "None",
            "A102": "Co-applicant",
            "A103": "Guarantor",
        }[x],
    )


# ==========================================================
# COLUMN 2
# ==========================================================

with col2:

    residence_since = st.slider(
        "Years at Current Residence",
        min_value=1,
        max_value=4,
        value=2,
    )

    property_status = st.selectbox(
        "Property",
        options=[
            "A121",
            "A122",
            "A123",
            "A124",
        ],
        format_func=lambda x: {
            "A121": "Real Estate",
            "A122": "Building Society / Life Insurance",
            "A123": "Car / Other Property",
            "A124": "No Property",
        }[x],
    )

    age = st.slider(
        "Age",
        min_value=18,
        max_value=80,
        value=30,
    )

    other_installment_plans = st.selectbox(
        "Other Installment Plans",
        options=[
            "A141",
            "A142",
            "A143",
        ],
        format_func=lambda x: {
            "A141": "Bank",
            "A142": "Stores",
            "A143": "None",
        }[x],
    )

    housing = st.selectbox(
        "Housing",
        options=[
            "A151",
            "A152",
            "A153",
        ],
        format_func=lambda x: {
            "A151": "Rent",
            "A152": "Own",
            "A153": "Free",
        }[x],
    )

    existing_credits = st.slider(
        "Number of Existing Credits",
        min_value=1,
        max_value=4,
        value=1,
    )

    job = st.selectbox(
        "Job Type",
        options=[
            "A171",
            "A172",
            "A173",
            "A174",
        ],
        format_func=lambda x: {
            "A171": "Unemployed / Unskilled",
            "A172": "Unskilled Resident",
            "A173": "Skilled Employee",
            "A174": "Management / Self-employed",
        }[x],
    )

    people_liable = st.slider(
        "Number of Dependents",
        min_value=1,
        max_value=2,
        value=1,
    )

    telephone = st.selectbox(
        "Registered Telephone",
        options=[
            "A191",
            "A192",
        ],
        format_func=lambda x: {
            "A191": "No",
            "A192": "Yes",
        }[x],
    )

    foreign_worker = st.selectbox(
        "Foreign Worker",
        options=[
            "A201",
            "A202",
        ],
        format_func=lambda x: {
            "A201": "Yes",
            "A202": "No",
        }[x],
    )


# ==========================================================
# CREATE INPUT DATAFRAME
# ==========================================================

input_data = pd.DataFrame(
    {
        "checking_account": [checking_account],
        "duration": [duration],
        "credit_history": [credit_history],
        "purpose": [purpose],
        "credit_amount": [credit_amount],
        "savings_account": [savings_account],
        "employment_since": [employment_since],
        "installment_rate": [installment_rate],
        "personal_status_sex": [personal_status_sex],
        "other_debtors": [other_debtors],
        "residence_since": [residence_since],
        "property": [property_status],
        "age": [age],
        "other_installment_plans": [other_installment_plans],
        "housing": [housing],
        "existing_credits": [existing_credits],
        "job": [job],
        "people_liable": [people_liable],
        "telephone": [telephone],
        "foreign_worker": [foreign_worker],
    }
)


# ==========================================================
# PREDICTION
# ==========================================================

st.divider()

if st.button(
    "🔍 Analyze Credit Risk",
    type="primary",
    use_container_width=True,
):

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    good_probability = probabilities[0] * 100
    bad_probability = probabilities[1] * 100


    st.header("Prediction Result")


    if prediction == 0:

        st.success(
            "✅ GOOD CREDIT RISK"
        )

        st.write(
            "The model predicts that this applicant "
            "has a relatively lower credit risk."
        )

    else:

        st.error(
            "⚠️ BAD CREDIT RISK"
        )

        st.write(
            "The model predicts that this applicant "
            "has a relatively higher credit risk."
        )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Good Credit Probability",
            f"{good_probability:.2f}%",
        )


    with result_col2:

        st.metric(
            "Bad Credit Probability",
            f"{bad_probability:.2f}%",
        )


    st.subheader("Risk Probability")

    st.progress(
        int(bad_probability)
    )

    st.caption(
        f"Estimated bad-credit probability: "
        f"{bad_probability:.2f}%"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Credit Scoring Model | Machine Learning Internship Project"
)