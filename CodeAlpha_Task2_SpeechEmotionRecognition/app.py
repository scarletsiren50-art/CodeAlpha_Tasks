from pathlib import Path
import tempfile

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "emotion_model_v3.pkl"
)

LABEL_ENCODER_PATH = (
    BASE_DIR
    / "models"
    / "label_encoder_v3.pkl"
)


# ==========================================================
# AUDIO SETTINGS
# Must match Version 3 training
# ==========================================================

SAMPLE_RATE = 22050
DURATION = 3.0
FIXED_LENGTH = int(SAMPLE_RATE * DURATION)
N_MFCC = 40


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_models():

    model = joblib.load(
        MODEL_PATH
    )

    label_encoder = joblib.load(
        LABEL_ENCODER_PATH
    )

    return model, label_encoder


model, label_encoder = load_models()


# ==========================================================
# EMOTION DISPLAY
# ==========================================================

EMOTION_ICONS = {
    "Angry": "😠",
    "Calm": "😌",
    "Disgust": "🤢",
    "Fearful": "😨",
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprised": "😲",
}


# ==========================================================
# LOAD AUDIO
# ==========================================================

def load_audio(file_path):

    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    # Same silence trimming used during V3 training
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
# FEATURE SUMMARY
# ==========================================================

def summarize_feature(feature):

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
# Must match Version 3 EXACTLY
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
# HEADER
# ==========================================================

st.title(
    "🎙️ Speech Emotion Recognition"
)

st.write(
    """
    Upload a speech recording and the machine learning model
    will analyze acoustic characteristics of the voice to
    predict the expressed emotion.
    """
)

st.info(
    "The system analyzes MFCC, Delta MFCC, Chroma, "
    "Mel Spectrogram and spectral audio features."
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header(
    "🧠 About the Model"
)

st.sidebar.write(
    """
    **Dataset:** RAVDESS

    **Final Model:** Random Forest

    **Emotion Classes:** 8

    **Acoustic Features:** 402

    **Test Accuracy:** 45.00%

    **Macro F1-Score:** 0.4193
    """
)

st.sidebar.subheader(
    "Recognized Emotions"
)

st.sidebar.write(
    """
    😠 Angry

    😌 Calm

    🤢 Disgust

    😨 Fearful

    😊 Happy

    😐 Neutral

    😢 Sad

    😲 Surprised
    """
)

st.sidebar.warning(
    "Educational machine learning project. "
    "Predictions may not accurately represent a person's "
    "actual emotional state."
)


# ==========================================================
# UPLOAD SECTION
# ==========================================================

st.header(
    "Upload Speech Recording"
)

uploaded_file = st.file_uploader(
    "Choose a WAV audio file",
    type=["wav"],
)


# ==========================================================
# PROCESS UPLOAD
# ==========================================================

if uploaded_file is not None:

    st.success(
        f"Audio uploaded successfully: "
        f"{uploaded_file.name}"
    )

    st.audio(
        uploaded_file,
        format="audio/wav",
    )


    # ------------------------------------------------------
    # Save uploaded file temporarily
    # ------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav",
    ) as temporary_file:

        temporary_file.write(
            uploaded_file.getbuffer()
        )

        temporary_path = (
            temporary_file.name
        )


    try:

        # --------------------------------------------------
        # Load original audio for visualization
        # --------------------------------------------------

        original_audio, original_sr = librosa.load(
            temporary_path,
            sr=None,
            mono=True,
        )


        duration = librosa.get_duration(
            y=original_audio,
            sr=original_sr,
        )


        st.subheader(
            "Audio Information"
        )


        info_col1, info_col2 = st.columns(
            2
        )


        with info_col1:

            st.metric(
                "Duration",
                f"{duration:.2f} seconds",
            )


        with info_col2:

            st.metric(
                "Sample Rate",
                f"{original_sr} Hz",
            )


        # ==================================================
        # WAVEFORM
        # ==================================================

        st.subheader(
            "Speech Waveform"
        )


        time_axis = (
            np.arange(
                len(original_audio)
            )
            / original_sr
        )


        fig, ax = plt.subplots(
            figsize=(11, 4)
        )


        ax.plot(
            time_axis,
            original_audio,
        )


        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.set_title(
            "Uploaded Speech Waveform"
        )


        ax.grid(
            alpha=0.2
        )


        plt.tight_layout()


        st.pyplot(
            fig
        )


        plt.close(
            fig
        )


        # ==================================================
        # ANALYZE BUTTON
        # ==================================================

        st.divider()


        if st.button(
            "🧠 Analyze Emotion",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Analyzing speech characteristics..."
            ):

                # ------------------------------------------
                # Prepare audio exactly like training
                # ------------------------------------------

                processed_audio = load_audio(
                    temporary_path
                )


                # ------------------------------------------
                # Extract 402 features
                # ------------------------------------------

                features = extract_features(
                    processed_audio
                )


                if len(features) != 402:

                    st.error(
                        "Unexpected feature count. "
                        f"Expected 402 but received "
                        f"{len(features)}."
                    )

                    st.stop()


                features = features.reshape(
                    1,
                    -1,
                )


                # ------------------------------------------
                # Prediction
                # ------------------------------------------

                prediction_encoded = model.predict(
                    features
                )[0]


                predicted_emotion = (
                    label_encoder.inverse_transform(
                        [prediction_encoded]
                    )[0]
                )


                # ------------------------------------------
                # Probabilities
                # ------------------------------------------

                probabilities = model.predict_proba(
                    features
                )[0]


                confidence = (
                    np.max(probabilities)
                    * 100
                )


                emotion_icon = EMOTION_ICONS.get(
                    predicted_emotion,
                    "🎭",
                )


                # ==========================================
                # RESULT
                # ==========================================

                st.header(
                    "Prediction Result"
                )


                st.success(
                    f"{emotion_icon} "
                    f"Detected Emotion: "
                    f"{predicted_emotion.upper()}"
                )


                result_col1, result_col2 = (
                    st.columns(2)
                )


                with result_col1:

                    st.metric(
                        "Predicted Emotion",
                        f"{emotion_icon} "
                        f"{predicted_emotion}",
                    )


                with result_col2:

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%",
                    )


                # ==========================================
                # PROBABILITIES
                # ==========================================

                st.subheader(
                    "Emotion Probabilities"
                )


                emotion_probabilities = []


                for class_index, emotion in enumerate(
                    label_encoder.classes_
                ):

                    probability = (
                        probabilities[
                            class_index
                        ]
                        * 100
                    )


                    emotion_probabilities.append(
                        (
                            emotion,
                            probability,
                        )
                    )


                emotion_probabilities.sort(
                    key=lambda item: item[1],
                    reverse=True,
                )


                for emotion, probability in (
                    emotion_probabilities
                ):

                    icon = EMOTION_ICONS.get(
                        emotion,
                        "🎭",
                    )


                    st.write(
                        f"**{icon} {emotion}** "
                        f"— {probability:.2f}%"
                    )


                    st.progress(
                        int(
                            round(
                                probability
                            )
                        )
                    )


                # ==========================================
                # MFCC VISUALIZATION
                # ==========================================

                st.subheader(
                    "MFCC Visualization"
                )


                mfcc = librosa.feature.mfcc(
                    y=processed_audio,
                    sr=SAMPLE_RATE,
                    n_mfcc=N_MFCC,
                )


                fig_mfcc, ax_mfcc = plt.subplots(
                    figsize=(11, 5)
                )


                image = librosa.display.specshow(
                    mfcc,
                    x_axis="time",
                    sr=SAMPLE_RATE,
                    ax=ax_mfcc,
                )


                fig_mfcc.colorbar(
                    image,
                    ax=ax_mfcc,
                    format="%+2.0f",
                )


                ax_mfcc.set_title(
                    "MFCC Features of Uploaded Speech"
                )


                plt.tight_layout()


                st.pyplot(
                    fig_mfcc
                )


                plt.close(
                    fig_mfcc
                )


                # ==========================================
                # INTERPRETATION
                # ==========================================

                st.caption(
                    "Confidence represents the model's "
                    "estimated class probability. It should "
                    "not be interpreted as certainty about "
                    "a person's real emotional state."
                )


    except Exception as error:

        st.error(
            "The audio file could not be analyzed."
        )

        st.exception(
            error
        )


# ==========================================================
# EMPTY STATE
# ==========================================================

else:

    st.write(
        "👆 Upload a `.wav` speech recording "
        "to begin emotion analysis."
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Speech Emotion Recognition System | "
    "CodeAlpha Machine Learning Internship Project"
)