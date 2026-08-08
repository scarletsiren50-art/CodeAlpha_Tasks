from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "german.data"
MODEL_PATH = BASE_DIR / "models" / "credit_model.pkl"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================================
# COLUMN NAMES
# ==========================================================

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


# ==========================================================
# LOAD DATA
# ==========================================================

print("\n" + "=" * 65)
print("CREDIT SCORING MODEL - MODEL COMPARISON")
print("=" * 65)

df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    names=columns,
)

print(f"\nDataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")


# ==========================================================
# TARGET PREPARATION
# ==========================================================

# Original dataset:
# 1 = Good credit
# 2 = Bad credit
#
# New representation:
# 0 = Good credit
# 1 = Bad credit

df["credit_risk"] = df["credit_risk"].map({
    1: 0,
    2: 1,
})

X = df.drop(columns=["credit_risk"])
y = df["credit_risk"]


# ==========================================================
# FEATURE GROUPS
# ==========================================================

categorical_features = [
    "checking_account",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

numerical_features = [
    "duration",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "people_liable",
]


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ==========================================================
# PREPROCESSOR FACTORY
# ==========================================================

def create_preprocessor():

    numerical_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                numerical_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )


# ==========================================================
# MODELS
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=10,
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
    ),
}


# ==========================================================
# TRAIN AND EVALUATE
# ==========================================================

results = []

trained_models = {}

roc_data = {}


for model_name, classifier in models.items():

    print("\n" + "=" * 65)
    print(f"TRAINING: {model_name}")
    print("=" * 65)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
    })

    trained_models[model_name] = pipeline

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    roc_data[model_name] = (
        fpr,
        tpr,
        roc_auc
    )

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Good Credit",
                "Bad Credit",
            ],
        )
    )


# ==========================================================
# MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==========================================================
# SELECT BEST MODEL
# ==========================================================

# ROC-AUC is used because it evaluates how well the model
# separates good and bad credit applicants across thresholds.

best_index = results_df["ROC-AUC"].idxmax()

best_model_name = results_df.loc[
    best_index,
    "Model"
]

best_model = trained_models[best_model_name]


print("\n" + "=" * 75)
print("BEST MODEL")
print("=" * 75)

print(f"\nSelected model: {best_model_name}")
print(
    f"ROC-AUC: "
    f"{results_df.loc[best_index, 'ROC-AUC']:.4f}"
)


# ==========================================================
# SAVE BEST MODEL
# ==========================================================

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nBest model saved successfully!")
print(f"Location: {MODEL_PATH}")


# ==========================================================
# SAVE RESULTS AS CSV
# ==========================================================

comparison_csv = OUTPUT_DIR / "model_results.csv"

results_df.to_csv(
    comparison_csv,
    index=False
)

print(f"\nModel results saved to:\n{comparison_csv}")


# ==========================================================
# MODEL COMPARISON CHART
# ==========================================================

chart_data = results_df.set_index("Model")[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    ]
]

ax = chart_data.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title(
    "Credit Scoring Model Performance Comparison",
    fontsize=15
)

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(
    rotation=0
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

comparison_path = (
    OUTPUT_DIR /
    "model_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=300
)

plt.close()

print(f"\nModel comparison chart saved:\n{comparison_path}")


# ==========================================================
# ROC CURVE
# ==========================================================

plt.figure(
    figsize=(9, 7)
)

for model_name, values in roc_data.items():

    fpr, tpr, auc_score = values

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc_score:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curves - Credit Scoring Models"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

roc_path = OUTPUT_DIR / "roc_curve.png"

plt.savefig(
    roc_path,
    dpi=300
)

plt.close()

print(f"\nROC curve saved:\n{roc_path}")


# ==========================================================
# CONFUSION MATRIX FOR BEST MODEL
# ==========================================================

best_predictions = best_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    best_predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Good Credit",
        "Bad Credit",
    ],
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

confusion_path = (
    OUTPUT_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300
)

plt.close()

print(
    f"\nConfusion matrix saved:\n"
    f"{confusion_path}"
)


# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 75)
print("TRAINING AND MODEL COMPARISON COMPLETE")
print("=" * 75)

print("\nGenerated files:")

print("1. models/credit_model.pkl")
print("2. outputs/model_results.csv")
print("3. outputs/model_comparison.png")
print("4. outputs/roc_curve.png")
print("5. outputs/confusion_matrix.png")

print("\nNext step: Build the Credit Risk Prediction application.")