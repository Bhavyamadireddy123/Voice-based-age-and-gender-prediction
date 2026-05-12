import librosa
import numpy as np

def extract_features(file_path, sampling_rate=22050):

    try:
        audio, sr = librosa.load(file_path, sr=sampling_rate)

        features = []

        # Spectral Centroid
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sr)
        )
        features.append(spectral_centroid)

        # Spectral Bandwidth
        spectral_bandwidth = np.mean(
            librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        )
        features.append(spectral_bandwidth)

        # Spectral Rolloff
        spectral_rolloff = np.mean(
            librosa.feature.spectral_rolloff(y=audio, sr=sr)
        )
        features.append(spectral_rolloff)

        # MFCC Features
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=20
        )

        for mfcc in mfccs:
            features.append(np.mean(mfcc))

        return np.array(features)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None