from pathlib import Path

import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)


# ==========================================================
# UCI CLEVELAND HEART DISEASE DATA
# ==========================================================

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)


COLUMN_NAMES = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_angina",
    "st_depression",
    "st_slope",
    "major_vessels",
    "thalassemia",
    "target",
]


# ==========================================================
# LOAD DATA
# ==========================================================

print("\n" + "=" * 65)
print("UCI HEART DISEASE DATASET PREPARATION")
print("=" * 65)

df = pd.read_csv(
    DATA_URL,
    names=COLUMN_NAMES,
    na_values="?",
)


print(
    f"\nOriginal dataset shape: {df.shape}"
)

print(
    f"Missing values: {df.isnull().sum().sum()}"
)


# ==========================================================
# CLEAN DATA
# ==========================================================

# Remove the small number of incomplete rows.
df = df.dropna().copy()


# UCI target:
# 0 = no disease
# 1-4 = presence/severity of disease
#
# Convert to binary classification:
# 0 = No Heart Disease
# 1 = Heart Disease

df["target"] = (
    df["target"] > 0
).astype(int)


# Convert columns loaded as object due to "?" values
df["major_vessels"] = pd.to_numeric(
    df["major_vessels"]
)

df["thalassemia"] = pd.to_numeric(
    df["thalassemia"]
)


# ==========================================================
# SAVE CLEAN DATASET
# ==========================================================

OUTPUT_PATH = (
    DATA_DIR / "heart_disease.csv"
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ==========================================================
# SUMMARY
# ==========================================================

print(
    f"\nClean dataset shape: {df.shape}"
)

print(
    f"Remaining missing values: "
    f"{df.isnull().sum().sum()}"
)


print("\nTarget distribution:")

print(
    df["target"].value_counts().sort_index()
)


print("\nTarget meaning:")
print("0 = No Heart Disease")
print("1 = Heart Disease")


print(
    "\nSaved dataset:"
)

print(
    OUTPUT_PATH
)

print("\n" + "=" * 65)
print("DATASET PREPARATION COMPLETE")
print("=" * 65)