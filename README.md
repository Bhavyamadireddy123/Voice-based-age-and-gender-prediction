# Voice-based-age-and-gender-prediction
A machine learning-based web application that predicts a person's age group and gender from voice recordings using audio feature extraction and Random Forest classification.

## Overview

This project is a machine learning-based web application that predicts the age group and gender of a person using voice recordings. The system accepts audio input through microphone recording or audio file upload and analyzes acoustic speech features to generate predictions.

The application uses audio signal processing techniques and machine learning algorithms to classify voices based on extracted acoustic characteristics. A Streamlit-based graphical interface is used to provide an interactive and user-friendly experience.

---

## Features
- Record voice directly using microphone
- Upload audio files in WAV or MP3 format
- Audio waveform visualization
- Gender prediction
- Age group prediction
- Interactive Streamlit web application
- Fast prediction results

---

## How to Use the Application

1. Open the Streamlit application in your web browser.
2. Choose either:
- Microphone Recording
- Audio File Upload
3. Record your voice or upload an audio file in WAV or MP3 format.
4. The uploaded audio will be displayed and visualized as a waveform.
5. Click on the **Analyze Voice** button.
6. The system extracts important acoustic features from the audio using Librosa.
7. The trained Random Forest machine learning models process the extracted features.
8. The application predicts:
- Gender
- Age Group
9. The prediction results along with confidence scores are displayed on the screen.
10. Users can test multiple audio samples to observe different prediction results.
  
---

## Technologies Used

Python, Streamlit, Librosa, Scikit-learn, NumPy, Pandas, Matplotlib, and Joblib.

---

## Methodology

The project uses machine learning techniques to predict the age group and gender of a person based on voice recordings. Audio input is collected through microphone recording or audio file upload and processed using the Librosa library.

Important acoustic features such as Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, and MFCCs are extracted from the speech signal. These features are converted into numerical vectors and used to train Random Forest classification models.

Separate machine learning models are trained for:
- Gender prediction
- Age group prediction

The trained models analyze the extracted voice features and generate prediction results, which are displayed through the Streamlit web interface along with waveform visualization.

---

## Results

<img width="1600" height="950" alt="image" src="https://github.com/user-attachments/assets/6a41150c-f2f1-48e9-a216-ae2f972d328c" />




<img width="1600" height="840" alt="image" src="https://github.com/user-attachments/assets/b7b37ef2-5ffa-4e51-a23c-59136474fb73" />



## Future Improvements

- Improve prediction accuracy using deep learning models such as CNN and LSTM
- Train the system using larger and more diverse voice datasets
- Add real-time voice streaming and live prediction support
- Implement advanced noise reduction techniques
- Support multiple languages and accents
- Deploy the application on cloud platforms
- Develop a mobile application version
- Add additional speech analysis features such as emotion detection and speaker recognition
