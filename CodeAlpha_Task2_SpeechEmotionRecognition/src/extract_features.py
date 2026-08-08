from pathlib import Path
from collections import Counter

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
FIXED_LENGTH = int(SAMPLE_RATE * DURATION)

N_MFCC = 40


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
# LOAD AND PREPARE AUDIO
# ==========================================================

def load_audio(file_path):

    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    # Trim silence
    audio, _ = librosa.effects.trim(
        audio,
        top_db=30,
    )

    # Pad short recordings
    if len(audio) < FIXED_LENGTH:

        audio = np.pad(
            audio,
            (
                0,
                FIXED_LENGTH - len(audio),
            ),
            mode="constant",
        )

    # Trim long recordings
    else:

        audio = audio[:FIXED_LENGTH]

    return audio


# ==========================================================
# HELPER
# ==========================================================

def summarize_feature(feature):

    """
    Convert a time-dependent audio feature into
    mean and standard-deviation statistics.
    """

    mean = np.mean(
        feature,
        axis=1,
    )

    std = np.std(
        feature,
        axis=1,
    )

    return np.concatenate(
        [mean, std]
    )


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features(audio):

    all_features = []


    # ------------------------------------------------------
    # 1. MFCC
    # ------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
    )

    all_features.extend(
        summarize_feature(mfcc)
    )


    # ------------------------------------------------------
    # 2. FIRST-ORDER MFCC DELTA
    # ------------------------------------------------------

    delta_mfcc = librosa.feature.delta(
        mfcc,
        order=1,
    )

    all_features.extend(
        summarize_feature(delta_mfcc)
    )


    # ------------------------------------------------------
    # 3. SECOND-ORDER MFCC DELTA
    # ------------------------------------------------------

    delta2_mfcc = librosa.feature.delta(
        mfcc,
        order=2,
    )

    all_features.extend(
        summarize_feature(delta2_mfcc)
    )


    # ------------------------------------------------------
    # 4. CHROMA
    # ------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=SAMPLE_RATE,
    )

    all_features.extend(
        summarize_feature(chroma)
    )


    # ------------------------------------------------------
    # 5. MEL SPECTROGRAM
    # ------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=64,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    all_features.extend(
        summarize_feature(mel_db)
    )


    # ------------------------------------------------------
    # 6. ZERO CROSSING RATE
    # ------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    all_features.extend(
        summarize_feature(zcr)
    )


    # ------------------------------------------------------
    # 7. RMS ENERGY
    # ------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio
    )

    all_features.extend(
        summarize_feature(rms)
    )


    # ------------------------------------------------------
    # 8. SPECTRAL CENTROID
    # ------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SAMPLE_RATE,
    )

    all_features.extend(
        summarize_feature(centroid)
    )


    # ------------------------------------------------------
    # 9. SPECTRAL BANDWIDTH
    # ------------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SAMPLE_RATE,
    )

    all_features.extend(
        summarize_feature(bandwidth)
    )


    # ------------------------------------------------------
    # 10. SPECTRAL ROLLOFF
    # ------------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SAMPLE_RATE,
    )

    all_features.extend(
        summarize_feature(rolloff)
    )


    return np.asarray(
        all_features,
        dtype=np.float32,
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n" + "=" * 72)
    print("SPEECH EMOTION RECOGNITION - FEATURE EXTRACTION V3")
    print("=" * 72)


    audio_files = sorted(
        DATA_DIR.glob("Actor_*/*.wav")
    )


    if not audio_files:

        raise FileNotFoundError(
            f"No WAV files found inside:\n{DATA_DIR}"
        )


    print(
        f"\nAudio files found: {len(audio_files)}"
    )


    X = []

    labels = []

    actors = []

    filenames = []

    failed_files = []


    # ======================================================
    # PROCESS AUDIO
    # ======================================================

    for index, audio_file in enumerate(
        audio_files,
        start=1,
    ):

        try:

            parts = audio_file.stem.split("-")

            emotion_code = parts[2]

            actor_id = int(
                parts[6]
            )

            emotion = EMOTIONS[
                emotion_code
            ]


            audio = load_audio(
                audio_file
            )


            feature_vector = extract_features(
                audio
            )


            X.append(
                feature_vector
            )

            labels.append(
                emotion
            )

            actors.append(
                actor_id
            )

            filenames.append(
                audio_file.name
            )


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
    # CONVERT TO ARRAYS
    # ======================================================

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    labels = np.asarray(
        labels
    )

    actors = np.asarray(
        actors,
        dtype=np.int32,
    )

    filenames = np.asarray(
        filenames
    )


    # ======================================================
    # DIAGNOSTICS
    # ======================================================

    print("\n" + "=" * 72)
    print("FEATURE DIAGNOSTICS")
    print("=" * 72)


    print(
        f"\nFeature matrix shape: "
        f"{X.shape}"
    )


    print(
        f"Number of features per recording: "
        f"{X.shape[1]}"
    )


    nan_count = np.isnan(
        X
    ).sum()


    inf_count = np.isinf(
        X
    ).sum()


    print(
        f"\nNaN values: {nan_count}"
    )

    print(
        f"Infinite values: {inf_count}"
    )


    feature_variance = np.var(
        X,
        axis=0,
    )


    print(
        f"\nMinimum feature variance: "
        f"{feature_variance.min():.8f}"
    )


    print(
        f"Maximum feature variance: "
        f"{feature_variance.max():.8f}"
    )


    zero_variance = np.sum(
        feature_variance == 0
    )


    print(
        f"Zero-variance features: "
        f"{zero_variance}"
    )


    # ======================================================
    # EMOTION DISTRIBUTION
    # ======================================================

    print(
        "\nEmotion distribution:"
    )


    emotion_counts = Counter(
        labels
    )


    for emotion in EMOTIONS.values():

        print(
            f"{emotion:<12}: "
            f"{emotion_counts.get(emotion, 0)}"
        )


    # ======================================================
    # ACTOR SPLIT DIAGNOSTICS
    # ======================================================

    train_mask = actors <= 16

    validation_mask = (
        (actors >= 17)
        &
        (actors <= 20)
    )

    test_mask = actors >= 21


    print("\nSpeaker-independent split:")

    print(
        f"Training actors   : 1-16 "
        f"({train_mask.sum()} samples)"
    )

    print(
        f"Validation actors : 17-20 "
        f"({validation_mask.sum()} samples)"
    )

    print(
        f"Testing actors    : 21-24 "
        f"({test_mask.sum()} samples)"
    )


    print(
        "\nTraining emotion distribution:"
    )

    print(
        Counter(
            labels[train_mask]
        )
    )


    print(
        "\nValidation emotion distribution:"
    )

    print(
        Counter(
            labels[validation_mask]
        )
    )


    print(
        "\nTesting emotion distribution:"
    )

    print(
        Counter(
            labels[test_mask]
        )
    )


    # ======================================================
    # SAVE NPZ
    # ======================================================

    npz_path = (
        OUTPUT_DIR /
        "ravdess_features_v3.npz"
    )


    np.savez_compressed(
        npz_path,
        X=X,
        y=labels,
        actors=actors,
        filenames=filenames,
    )


    # ======================================================
    # SAVE CSV FOR INSPECTION
    # ======================================================

    feature_names = [
        f"feature_{i + 1}"
        for i in range(
            X.shape[1]
        )
    ]


    df = pd.DataFrame(
        X,
        columns=feature_names,
    )


    df.insert(
        0,
        "emotion",
        labels,
    )


    df.insert(
        0,
        "actor",
        actors,
    )


    df.insert(
        0,
        "filename",
        filenames,
    )


    csv_path = (
        OUTPUT_DIR /
        "ravdess_features_v3.csv"
    )


    df.to_csv(
        csv_path,
        index=False,
    )


    # ======================================================
    # COMPLETE
    # ======================================================

    print("\n" + "=" * 72)
    print("FEATURE EXTRACTION V3 COMPLETE")
    print("=" * 72)


    print(
        f"\nSuccessfully processed: "
        f"{len(X)}"
    )


    print(
        f"Failed files: "
        f"{len(failed_files)}"
    )


    print(
        "\nSaved:"
    )

    print(
        "1. outputs/ravdess_features_v3.npz"
    )

    print(
        "2. outputs/ravdess_features_v3.csv"
    )


    if failed_files:

        print(
            "\nFailed files:"
        )

        for filename, error in failed_files:

            print(
                f"{filename}: {error}"
            )


if __name__ == "__main__":
    main()