from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "heart_disease.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(
    DATA_PATH
)


print("\n" + "=" * 70)
print("HEART DISEASE DATASET EXPLORATION")
print("=" * 70)


print(
    f"\nDataset shape: {df.shape}"
)

print(
    f"Rows: {df.shape[0]}"
)

print(
    f"Features + target: {df.shape[1]}"
)


# ==========================================================
# BASIC INFORMATION
# ==========================================================

print("\nColumns:")

for column in df.columns:
    print(f"- {column}")


print(
    "\nMissing values:"
)

print(
    df.isnull().sum()
)


print(
    f"\nTotal missing values: "
    f"{df.isnull().sum().sum()}"
)


print(
    f"\nDuplicate rows: "
    f"{df.duplicated().sum()}"
)


# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

target_counts = (
    df["target"]
    .value_counts()
    .sort_index()
)


print(
    "\nHeart Disease Distribution:"
)

print(
    f"No Heart Disease (0): "
    f"{target_counts.get(0, 0)}"
)

print(
    f"Heart Disease (1): "
    f"{target_counts.get(1, 0)}"
)


# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

print(
    "\nSummary Statistics:"
)

print(
    df.describe().round(2)
)


# ==========================================================
# CREATE EXPLORATION FIGURE
# ==========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 10),
)


# Target distribution
target_counts.plot(
    kind="bar",
    ax=axes[0, 0],
)

axes[0, 0].set_title(
    "Heart Disease Distribution"
)

axes[0, 0].set_xlabel(
    "Target"
)

axes[0, 0].set_ylabel(
    "Number of Patients"
)

axes[0, 0].set_xticklabels(
    [
        "No Disease",
        "Disease",
    ],
    rotation=0,
)


# Age distribution
axes[0, 1].hist(
    df["age"],
    bins=15,
    edgecolor="black",
)

axes[0, 1].set_title(
    "Patient Age Distribution"
)

axes[0, 1].set_xlabel(
    "Age"
)

axes[0, 1].set_ylabel(
    "Frequency"
)


# Cholesterol distribution
axes[1, 0].hist(
    df["cholesterol"],
    bins=15,
    edgecolor="black",
)

axes[1, 0].set_title(
    "Cholesterol Distribution"
)

axes[1, 0].set_xlabel(
    "Cholesterol (mg/dL)"
)

axes[1, 0].set_ylabel(
    "Frequency"
)


# Maximum heart rate
axes[1, 1].hist(
    df["max_heart_rate"],
    bins=15,
    edgecolor="black",
)

axes[1, 1].set_title(
    "Maximum Heart Rate Distribution"
)

axes[1, 1].set_xlabel(
    "Maximum Heart Rate"
)

axes[1, 1].set_ylabel(
    "Frequency"
)


plt.suptitle(
    "UCI Heart Disease Dataset Exploration",
    fontsize=16,
)


plt.tight_layout()


OUTPUT_PATH = (
    OUTPUT_DIR
    / "dataset_exploration.png"
)


plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ==========================================================
# COMPLETE
# ==========================================================

print("\n" + "=" * 70)
print("DATASET EXPLORATION COMPLETE")
print("=" * 70)

print(
    "\nGenerated:"
)

print(
    "outputs/dataset_exploration.png"
)