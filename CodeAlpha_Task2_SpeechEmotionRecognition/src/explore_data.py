from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "ravdess"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================================
# SETTINGS
# ==========================================================

SAMPLE_RATE = 22050
DURATION = 3.0
N_MFCC = 40

FIXED_LENGTH = int(SAMPLE_RATE * DURATION)


EMOTIONS = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised",
}


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_mfcc(file_path):

    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION,
    )

    # Pad short recordings
    if len(audio) < FIXED_LENGTH:

        audio = np.pad(
            audio,
            (0, FIXED_LENGTH - len(audio)),
        )

    # Trim longer recordings
    else:

        audio = audio[:FIXED_LENGTH]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
    )

    # Mean + standard deviation across time
    mfcc_mean = np.mean(
        mfcc,
        axis=1,
    )

    mfcc_std = np.std(
        mfcc,
        axis=1,
    )

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
        ]
    )

    return features


# ==========================================================
# PROCESS DATASET
# ==========================================================

def main():

    print("\n" + "=" * 65)
    print("RAVDESS MFCC FEATURE EXTRACTION")
    print("=" * 65)

    audio_files = sorted(
        DATA_DIR.glob("Actor_*/*.wav")
    )

    if not audio_files:

        raise FileNotFoundError(
            f"No WAV files found in {DATA_DIR}"
        )

    print(
        f"\nAudio files found: "
        f"{len(audio_files)}"
    )

    feature_rows = []

    failed_files = []


    for index, audio_file in enumerate(
        audio_files,
        start=1,
    ):

        try:

            parts = audio_file.stem.split("-")

            emotion_code = parts[2]

            actor_id = int(parts[6])

            emotion = EMOTIONS[
                emotion_code
            ]

            features = extract_mfcc(
                audio_file
            )

            row = {
                "file": audio_file.name,
                "actor": actor_id,
                "emotion": emotion,
            }


            for feature_index, value in enumerate(
                features,
                start=1,
            ):

                row[
                    f"feature_{feature_index}"
                ] = value


            feature_rows.append(row)


        except Exception as error:

            failed_files.append(
                (
                    audio_file.name,
                    str(error),
                )
            )


        if (
            index % 100 == 0
            or index == len(audio_files)
        ):

            print(
                f"Processed "
                f"{index}/{len(audio_files)}"
            )


    # ======================================================
    # CREATE DATAFRAME
    # ======================================================

    df = pd.DataFrame(
        feature_rows
    )


    output_path = (
        OUTPUT_DIR /
        "ravdess_mfcc_features.csv"
    )


    df.to_csv(
        output_path,
        index=False,
    )


    # ======================================================
    # SUMMARY
    # ======================================================

    print("\n" + "=" * 65)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 65)

    print(
        f"\nSuccessfully processed: "
        f"{len(df)}"
    )

    print(
        f"Failed files: "
        f"{len(failed_files)}"
    )

    print(
        f"Feature columns: "
        f"{len([c for c in df.columns if c.startswith('feature_')])}"
    )

    print(
        f"\nDataset shape: "
        f"{df.shape}"
    )


    print(
        "\nEmotion distribution:"
    )

    print(
        df["emotion"]
        .value_counts()
        .sort_index()
    )


    print(
        "\nActor range:"
    )

    print(
        f"{df['actor'].min()} "
        f"to "
        f"{df['actor'].max()}"
    )


    print(
        "\nSaved feature dataset:"
    )

    print(
        output_path
    )


    if failed_files:

        print(
            "\nFiles that could "
            "not be processed:"
        )

        for filename, error in failed_files:

            print(
                f"{filename}: {error}"
            )


if __name__ == "__main__":
    main()