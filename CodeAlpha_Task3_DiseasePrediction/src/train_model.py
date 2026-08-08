from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "heart_disease.csv"
)

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

print("\n" + "=" * 75)
print("HEART DISEASE PREDICTION - MODEL TRAINING")
print("=" * 75)

df = pd.read_csv(DATA_PATH)

X = df.drop(
    columns=["target"]
)

y = df["target"]


print(
    f"\nDataset shape: {df.shape}"
)

print(
    f"Features: {X.shape[1]}"
)

print(
    f"Samples: {len(X)}"
)


# ==========================================================
# FEATURE TYPES
# ==========================================================

categorical_features = [
    "sex",
    "chest_pain_type",
    "fasting_blood_sugar",
    "resting_ecg",
    "exercise_angina",
    "st_slope",
    "major_vessels",
    "thalassemia",
]


numeric_features = [
    "age",
    "resting_bp",
    "cholesterol",
    "max_heart_rate",
    "st_depression",
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


print("\nTrain/Test Split:")

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


print(
    "\nTraining target distribution:"
)

print(
    y_train.value_counts().sort_index()
)


print(
    "\nTesting target distribution:"
)

print(
    y_test.value_counts().sort_index()
)


# ==========================================================
# PREPROCESSING
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
    ]
)


# ==========================================================
# MODELS
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
    ),
}


# ==========================================================
# TRAIN AND EVALUATE
# ==========================================================

results = []

trained_models = {}

roc_information = {}


print("\n" + "=" * 75)
print("MODEL EVALUATION")
print("=" * 75)


for model_name, classifier in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


    pipeline.fit(
        X_train,
        y_train,
    )


    predictions = pipeline.predict(
        X_test
    )


    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]


    accuracy = accuracy_score(
        y_test,
        predictions,
    )


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )


    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )


    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )


    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
        }
    )


    trained_models[
        model_name
    ] = pipeline


    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )


    roc_information[
        model_name
    ] = (
        fpr,
        tpr,
        roc_auc,
    )


    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )


# ==========================================================
# MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False,
).reset_index(drop=True)


print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    "\n"
    + results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    OUTPUT_DIR
    / "model_comparison.csv",
    index=False,
)


# ==========================================================
# SELECT BEST MODEL
# ==========================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 75)
print("BEST MODEL")
print("=" * 75)

print(
    f"\nSelected Model: "
    f"{best_model_name}"
)

print(
    f"ROC-AUC: "
    f"{results_df.iloc[0]['ROC-AUC']:.4f}"
)


# ==========================================================
# BEST MODEL CONFUSION MATRIX
# ==========================================================

best_predictions = best_model.predict(
    X_test
)


cm = confusion_matrix(
    y_test,
    best_predictions,
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Disease",
        "Disease",
    ],
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)


display.plot(
    ax=ax,
    values_format="d",
)


plt.title(
    f"{best_model_name} - Confusion Matrix"
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "confusion_matrix.png",
    dpi=300,
)


plt.close()


# ==========================================================
# ROC CURVES
# ==========================================================

plt.figure(
    figsize=(9, 7)
)


for model_name, (
    fpr,
    tpr,
    roc_auc,
) in roc_information.items():

    plt.plot(
        fpr,
        tpr,
        label=(
            f"{model_name} "
            f"(AUC = {roc_auc:.3f})"
        ),
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Heart Disease Prediction - ROC Curves"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "roc_curve.png",
    dpi=300,
)


plt.close()


# ==========================================================
# MODEL COMPARISON GRAPH
# ==========================================================

plot_df = (
    results_df
    .set_index("Model")
    [
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
        ]
    ]
)


ax = plot_df.plot(
    kind="bar",
    figsize=(12, 7),
)


plt.title(
    "Heart Disease Prediction - Model Comparison"
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Model"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "model_comparison.png",
    dpi=300,
)


plt.close()


# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

joblib.dump(
    best_model,
    MODEL_DIR
    / "heart_disease_model.pkl",
)


# ==========================================================
# SAVE METRICS
# ==========================================================

best_metrics = results_df.iloc[
    0
]


metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
        ],
        "Value": [
            best_metrics["Accuracy"],
            best_metrics["Precision"],
            best_metrics["Recall"],
            best_metrics["F1-Score"],
            best_metrics["ROC-AUC"],
        ],
    }
)


metrics_df.to_csv(
    OUTPUT_DIR
    / "best_model_metrics.csv",
    index=False,
)


# ==========================================================
# COMPLETE
# ==========================================================

print("\n" + "=" * 75)
print("MODEL TRAINING COMPLETE")
print("=" * 75)


print("\nGenerated files:")

print(
    "1. outputs/model_comparison.csv"
)

print(
    "2. outputs/model_comparison.png"
)

print(
    "3. outputs/confusion_matrix.png"
)

print(
    "4. outputs/roc_curve.png"
)

print(
    "5. outputs/best_model_metrics.csv"
)

print(
    "6. models/heart_disease_model.pkl"
)