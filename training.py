import pandas as pd
import numpy as np
import os
import joblib
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from feature_extraction import extract_features

import warnings
warnings.filterwarnings('ignore')


def train_models(limit_samples=3000):

    print("Downloading dataset...")

    path = kagglehub.dataset_download(
        "mozillaorg/common-voice"
    )

    print("Dataset path:", path)

    csv_path = os.path.join(path, "validated.tsv")
    audio_dir = os.path.join(path, "clips")

    print("Loading dataset...")

    df = pd.read_csv(csv_path, sep='\t')

    data = df[
        df['age'].notna() &
        df['gender'].notna()
    ].copy()

    if len(data) > limit_samples:
        data = data.sample(
            limit_samples,
            random_state=42
        )

    print(f"Using {len(data)} samples")

    X = []
    y_gender = []
    y_age = []

    count = 0

    for idx, row in data.iterrows():

        audio_path = os.path.join(
            audio_dir,
            row['path']
        )

        if os.path.exists(audio_path):

            features = extract_features(audio_path)

            if features is not None:

                X.append(features)

                y_gender.append(row['gender'])
                y_age.append(row['age'])

                count += 1

                if count % 100 == 0:
                    print(f"Processed {count} files")

    X = np.array(X)

    print("Training Gender Model...")

    le_gender = LabelEncoder()

    y_gender_encoded = le_gender.fit_transform(
        y_gender
    )

    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        X,
        y_gender_encoded,
        test_size=0.2,
        random_state=42
    )

    gender_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    gender_model.fit(
        X_train_g,
        y_train_g
    )

    gender_pred = gender_model.predict(
        X_test_g
    )

    print(
        "Gender Accuracy:",
        accuracy_score(y_test_g, gender_pred)
    )

    print("Training Age Model...")

    le_age = LabelEncoder()

    y_age_encoded = le_age.fit_transform(
        y_age
    )

    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X,
        y_age_encoded,
        test_size=0.2,
        random_state=42
    )

    age_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    age_model.fit(
        X_train_a,
        y_train_a
    )

    age_pred = age_model.predict(
        X_test_a
    )

    print(
        "Age Accuracy:",
        accuracy_score(y_test_a, age_pred)
    )

    print("Saving models...")

    joblib.dump(
        gender_model,
        "gender_model.joblib"
    )

    joblib.dump(
        age_model,
        "age_model.joblib"
    )

    joblib.dump(
        le_gender,
        "le_gender.joblib"
    )

    joblib.dump(
        le_age,
        "le_age.joblib"
    )

    print("Training Complete!")


if __name__ == "__main__":

    train_models(limit_samples=3000)