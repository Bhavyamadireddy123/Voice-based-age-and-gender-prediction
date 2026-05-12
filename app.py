import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tempfile
import os

from feature_extraction import extract_features


st.set_page_config(
    page_title="Voice Age Gender Predictor",
    page_icon="🎤"
)

st.title("🎤 Voice-Based Age & Gender Predictor")

try:

    gender_model = joblib.load(
        "gender_model.joblib"
    )

    age_model = joblib.load(
        "age_model.joblib"
    )

    le_gender = joblib.load(
        "le_gender.joblib"
    )

    le_age = joblib.load(
        "le_age.joblib"
    )

except:

    st.error(
        "Models not found. Run training.py first."
    )

    st.stop()


audio_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3"]
)

if audio_file is not None:

    st.audio(audio_file)

    if st.button("Predict"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_audio:

            temp_audio.write(audio_file.read())

            temp_path = temp_audio.name

        features = extract_features(temp_path)

        if features is not None:

            features = features.reshape(1, -1)

            gender_pred = gender_model.predict(
                features
            )[0]

            age_pred = age_model.predict(
                features
            )[0]

            gender = le_gender.inverse_transform(
                [gender_pred]
            )[0]

            age = le_age.inverse_transform(
                [age_pred]
            )[0]

            st.success(
                f"Predicted Gender: {gender}"
            )

            st.success(
                f"Predicted Age Group: {age}"
            )

            y, sr = librosa.load(temp_path)

            fig, ax = plt.subplots()

            librosa.display.waveshow(
                y,
                sr=sr,
                ax=ax
            )

            st.pyplot(fig)

        os.remove(temp_path)