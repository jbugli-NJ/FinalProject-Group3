"""
Train a TF-IDF + LinearSVC classifier to predict policy areas from bill text.
Saves the model, vectorizer, and label-encoder for inference.

"""

# %% Imports

import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


# %% Define data loading function

def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# %% Define training method

def train(
    texts, labels,
    test_size: float = 0.2,
    random_state: int = 42
    ):

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    vect = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=3,
        stop_words='english'
    )
    X_train_tfidf = vect.fit_transform(X_train)

    clf = LinearSVC(random_state=random_state, max_iter=500, verbose=1)
    clf.fit(X_train_tfidf, y_train_enc)

    y_test_enc = le.transform(y_test)
    y_pred_enc = clf.predict(vect.transform(X_test))
    print(f'\nTest Accuracy: {accuracy_score(y_test_enc, y_pred_enc):.4f}\n')
    print('\nClassification Report:')
    print(classification_report(
        y_test_enc,
        y_pred_enc,
        target_names=le.classes_,
        zero_division=0
    ))

    return clf, vect, le

def save_artifacts(save_folder, model, vectorizer, label_encoder):

    os.makedirs(save_folder, exist_ok=True)

    joblib.dump(model, os.path.join(save_folder,'model.joblib'))
    joblib.dump(vectorizer, os.path.join(save_folder,'vectorizer.joblib'))
    joblib.dump(label_encoder, os.path.join(save_folder,'label_encoder.joblib'))


def main():

    parser = argparse.ArgumentParser(
        description='Train and save a policy_area classifier'
    )
    parser.add_argument(
        '--input', '-i', required=False,
        type=str,
        default='input_data.parquet',
        help='Path to .parquet data'
    )
    parser.add_argument(
        '--save_folder', '-s', required=False,
        default='model_artifacts',
        help='Folder to save .joblib files'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Fraction of data to hold out for testing'
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()

    df = load_data(args.input)
    texts = df['bill_text'].fillna('').values
    labels = df['policy_area'].fillna('').values

    model, vect, le = train(
        texts, labels,
        test_size=args.test_size,
        random_state=args.random_state
    )
    save_artifacts(
        args.save_folder,
        model, vect, le
    )

if __name__ == '__main__':
    main()
# %%
