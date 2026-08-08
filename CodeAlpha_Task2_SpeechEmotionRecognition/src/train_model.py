from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_PATH = (
    BASE_DIR
    / "outputs"
    / "ravdess_features_v3.npz"
)

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

print("\n" + "=" * 72)
print("SPEECH EMOTION RECOGNITION - V3 MODEL COMPARISON")
print("=" * 72)

data = np.load(
    FEATURE_PATH,
    allow_pickle=True,
)

X = data["X"]
labels = data["y"]
actors = data["actors"]


print(f"\nSamples : {len(X)}")
print(f"Features: {X.shape[1]}")


# ==========================================================
# LABEL ENCODING
# ==========================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    labels
)

print("\nEmotion classes:")

for index, emotion in enumerate(
    label_encoder.classes_
):
    print(f"{index}: {emotion}")


# ==========================================================
# SPEAKER-INDEPENDENT SPLIT
# ==========================================================

train_mask = actors <= 16

validation_mask = (
    (actors >= 17)
    & (actors <= 20)
)

test_mask = actors >= 21


X_train = X[train_mask]
y_train = y[train_mask]

X_validation = X[validation_mask]
y_validation = y[validation_mask]

X_test = X[test_mask]
y_test = y[test_mask]


print("\nDataset split:")

print(
    f"Training   : {len(X_train)}"
)

print(
    f"Validation : {len(X_validation)}"
)

print(
    f"Testing    : {len(X_test)}"
)


# ==========================================================
# MODELS
# ==========================================================

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "SVM": Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    ),
}


# ==========================================================
# VALIDATION COMPARISON
# ==========================================================

results = []

trained_models = {}


print("\n" + "=" * 72)
print("VALIDATION MODEL COMPARISON")
print("=" * 72)


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
    )


    results.append(
        {
            "Model": model_name,
            "Validation Accuracy": accuracy,
            "Validation Macro F1": macro_f1,
        }
    )

    trained_models[
        model_name
    ] = model


    print(
        f"Accuracy : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro F1 : "
        f"{macro_f1:.4f}"
    )


# ==========================================================
# RESULTS TABLE
# ==========================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="Validation Macro F1",
    ascending=False,
)


print("\n" + "=" * 72)
print("MODEL COMPARISON")
print("=" * 72)

print(
    results_df.to_string(
        index=False
    )
)


comparison_path = (
    OUTPUT_DIR
    / "baseline_model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False,
)


# ==========================================================
# SELECT BEST MODEL
# ==========================================================

best_model_name = results_df.iloc[
    0
]["Model"]

print("\n" + "=" * 72)
print("BEST VALIDATION MODEL")
print("=" * 72)

print(
    f"\nSelected model: "
    f"{best_model_name}"
)


# ==========================================================
# RETRAIN USING TRAIN + VALIDATION
# ==========================================================

print(
    "\nRetraining selected model "
    "using training + validation data..."
)


X_final_train = np.concatenate(
    [
        X_train,
        X_validation,
    ]
)

y_final_train = np.concatenate(
    [
        y_train,
        y_validation,
    ]
)


best_model = models[
    best_model_name
]


best_model.fit(
    X_final_train,
    y_final_train,
)


# ==========================================================
# FINAL TEST
# ==========================================================

test_predictions = best_model.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)


test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
)


print("\n" + "=" * 72)
print("FINAL TEST RESULTS - V3")
print("=" * 72)


print(
    f"\nSelected Model : "
    f"{best_model_name}"
)

print(
    f"Test Accuracy  : "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Accuracy  : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Macro F1-Score : "
    f"{test_macro_f1:.4f}"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print(
    "\nClassification Report:\n"
)


report_text = classification_report(
    y_test,
    test_predictions,
    target_names=label_encoder.classes_,
    zero_division=0,
)


print(
    report_text
)


report = classification_report(
    y_test,
    test_predictions,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0,
)


report_df = pd.DataFrame(
    report
).transpose()


report_df.to_csv(
    OUTPUT_DIR
    / "classification_report_v3.csv"
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_,
)


fig, ax = plt.subplots(
    figsize=(11, 9)
)


display.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d",
)


plt.title(
    f"Speech Emotion Recognition - {best_model_name}"
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "confusion_matrix_v3.png",
    dpi=300,
)

plt.close()


# ==========================================================
# MODEL COMPARISON GRAPH
# ==========================================================

plot_df = results_df.set_index(
    "Model"
)


ax = plot_df[
    [
        "Validation Accuracy",
        "Validation Macro F1",
    ]
].plot(
    kind="bar",
    figsize=(9, 6),
)


plt.title(
    "Speech Emotion Recognition - Baseline Comparison"
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Model"
)

plt.ylim(
    0,
    1,
)

plt.xticks(
    rotation=0
)

plt.legend(
    loc="best"
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "baseline_model_comparison.png",
    dpi=300,
)

plt.close()


# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

joblib.dump(
    best_model,
    MODEL_DIR
    / "emotion_model_v3.pkl",
)


joblib.dump(
    label_encoder,
    MODEL_DIR
    / "label_encoder_v3.pkl",
)


# ==========================================================
# SAVE FINAL METRICS
# ==========================================================

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Test Accuracy",
            "Macro F1-Score",
        ],
        "Value": [
            test_accuracy,
            test_macro_f1,
        ],
    }
)


metrics_df.to_csv(
    OUTPUT_DIR
    / "model_metrics_v3.csv",
    index=False,
)


# ==========================================================
# COMPLETE
# ==========================================================

print("\n" + "=" * 72)
print("V3 MODEL COMPARISON COMPLETE")
print("=" * 72)


print("\nGenerated files:")

print(
    "1. outputs/baseline_model_comparison.csv"
)

print(
    "2. outputs/baseline_model_comparison.png"
)

print(
    "3. outputs/classification_report_v3.csv"
)

print(
    "4. outputs/confusion_matrix_v3.png"
)

print(
    "5. outputs/model_metrics_v3.csv"
)

print(
    "6. models/emotion_model_v3.pkl"
)

print(
    "7. models/label_encoder_v3.pkl"
)