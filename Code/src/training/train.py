"""
Train a TF-IDF + LinearSVC classifier to predict policy areas from bill text.
Saves the model, vectorizer, and label-encoder for inference.

"""

# %% Imports

import argparse
import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
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
    texts: pd.Series,
    labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 3
    ):
    """
    Splits data, trains a TF-IDF + LinearSVC pipeline using GridSearchCV,
    evaluates on the test set, extracts the best components, and returns
    the best classifier, best vectorizer, and label encoder.
    """

    # Split data and encode labels

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)


    # Define the vectorization/classification pipeline

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LinearSVC(random_state=random_state, dual='auto', verbose=0))
    ])


    # Create parameter grids

    tf_idf_dict = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_df': [0.85, 0.95],
        'tfidf__min_df': [2, 3],
    }

    param_grid = [
        {
            **tf_idf_dict,
            'clf': [LinearSVC(random_state=random_state, dual='auto', verbose=0)],
            'clf__C': [0.5, 1.0, 5.0],
            'clf__max_iter': [500]
        },
        {
            **tf_idf_dict,
            'clf': [LogisticRegression(random_state=random_state, solver='liblinear', max_iter=1000)],
            'clf__C': [0.1, 1.0, 10.0],
            # 'clf__penalty': ['l1', 'l2']
        },
        {
            **tf_idf_dict,
            'clf': [MultinomialNB()],
            'clf__alpha': [0.1, 0.5, 1.0]
        }
    ]
    print(f"\nParameter grid:\n{param_grid}")

    # Run grid search

    print(f"\nStarting grid search ({cv_folds} folds)...")
    start_time = time.time()
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_folds,
        scoring='recall_weighted',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train_enc)
    end_time = time.time()
    print(f"Grid search complete ({end_time - start_time:.2f} seconds)!")


    # Report results

    print(f"\nBest parameters:")
    print(grid_search.best_params_)
    print(f"\nBest Cross-Validation Accuracy: {grid_search.best_score_:.4f}")


    # Return best components

    best_pipeline = grid_search.best_estimator_
    best_vectorizer = best_pipeline.named_steps['tfidf']
    best_classifier = best_pipeline.named_steps['svc']


    # Evaluate on test set

    X_test_tfidf = best_vectorizer.transform(X_test)
    y_pred_enc = best_classifier.predict(X_test_tfidf)

    print(f'\nTest Accuracy: {accuracy_score(y_test_enc, y_pred_enc):.4f}\n')
    print('Classification Report (Test Set):')
    print(classification_report(
        y_test_enc,
        y_pred_enc,
        target_names=le.classes_,
        zero_division=0
    ))

    # Return the best classifier, vectorizer, and encoder separately

    return best_classifier, best_vectorizer, le


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
