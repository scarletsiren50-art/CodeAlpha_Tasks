from pathlib import Path
import pandas as pd


# -----------------------------
# PROJECT PATHS
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "german.data"


# -----------------------------
# COLUMN NAMES
# -----------------------------

columns = [
    "checking_account",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "credit_risk",
]


# -----------------------------
# LOAD DATA
# -----------------------------

print("\nLoading German Credit dataset...\n")

df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    names=columns
)


# -----------------------------
# BASIC INFORMATION
# -----------------------------

print("=" * 60)
print("GERMAN CREDIT DATASET")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
for number, column in enumerate(df.columns, start=1):
    print(f"{number}. {column}")


# -----------------------------
# MISSING VALUES
# -----------------------------

print("\nMissing values:")
print(df.isnull().sum())


# -----------------------------
# TARGET DISTRIBUTION
# -----------------------------

print("\nCredit risk distribution:")
print(df["credit_risk"].value_counts())

print("\nCredit risk percentages:")
print(
    df["credit_risk"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# -----------------------------
# NUMERICAL SUMMARY
# -----------------------------

print("\nNumerical summary:")
print(df.describe())


print("\nDataset loaded successfully!")