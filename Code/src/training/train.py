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


# %% Define training method

def train(
    texts: pd.Series,
    labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 3
    ):
    """
    Executes training steps, including:
    1. Splitting the data
    2. Using a grid search pipeline to test models
    3. Evaluating results
    4. Returning the best model's artifacts for inference

    :param pd.Series texts: An input text series to process.

    :param pd.Series labels: A target policy area label series to process.

    :param float test_size: The size of the test split.

    :param int random_state: A random state seed.

    :param cv_folds: The number of folds for cross-validation.

    :return pd.DataFrame: A three-item tuple with:

        - The best classifier
        - The best vectorizer
        - The label encoder
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
    # NOTE: 'clf' is a placeholder for models

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LinearSVC(random_state=random_state, dual='auto', verbose=0))
    ])


    # Create parameter grids

    tf_idf_dict = {
        'tfidf__ngram_range': [(1, 2)],
        'tfidf__max_df': [0.85],
        'tfidf__min_df': [2],
    }

    param_grid = [
        {
            **tf_idf_dict,
            'clf': [LinearSVC(random_state=random_state, dual='auto', verbose=0)],
            'clf__C': [5.0],
            'clf__max_iter': [250]
        },
        {
            **tf_idf_dict,
            'clf': [LogisticRegression(random_state=random_state, solver='liblinear', verbose=1)],
            'clf__C': [1.0],
            'clf__max_iter': [250],
        },
        {
            **tf_idf_dict,
            'clf': [MultinomialNB()],
            'clf__alpha': [0.5],
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
        scoring='recall_macro',
        verbose=1
    )

    grid_search.fit(X_train, y_train_enc)
    end_time = time.time()
    print(f"Grid search complete ({end_time - start_time:.2f} seconds)!")

    scoring_metric = 'recall_macro'
    
    print(f"\n--- Sorted Grid Search Results ---")
    results_df = pd.DataFrame(grid_search.cv_results_)

    results_df = results_df.sort_values(by=['rank_test_score'])

    display_cols = [
        'params',
        'mean_test_score',
        'std_test_score',
        'rank_test_score',
        'mean_fit_time'
    ]

    display_cols = [col for col in display_cols if col in results_df.columns]

    pd.set_option('display.max_colwidth', None)
    print(results_df[display_cols].to_string(index=False))
    pd.reset_option('display.max_colwidth')


    # Report best results

    print(f"\nBest parameters:")
    print(grid_search.best_params_)

    # Note: grid_search.best_score_ uses the mean test score of the best estimator
    print(f"\nBest Cross-Validation {scoring_metric}: {grid_search.best_score_:.4f}")


    # Return best components

    best_pipeline = grid_search.best_estimator_
    best_vectorizer = best_pipeline.named_steps['tfidf']
    best_classifier = best_pipeline.named_steps['clf']


    # Evaluate on test set

    X_test_tfidf = best_vectorizer.transform(X_test)
    y_pred_enc = best_classifier.predict(X_test_tfidf)

    print(f'\nTest Accuracy: {accuracy_score(y_test_enc, y_pred_enc):.4f}\n')
    print('Classification Report (Test Set):')
    print(classification_report(
        y_test_enc,
        y_pred_enc,
        target_names=le.classes_,
        labels=range(len(le.classes_)),
        zero_division=0
    ))

    # Return the best classifier, vectorizer, and encoder separately

    return best_classifier, best_vectorizer, le


def save_artifacts(save_folder, model, vectorizer, label_encoder):
    """
    Save the best model, vectorizer, and label encoder in a folder for inference.
    All three are saved with the .joblib extension.

    :param str save_folder: The save folder path.

    :param model: The model to save.

    :param vectorizer: The vectorizer to save.

    :param label_encoder: The label encoder to save.
    """

    # Make sure the directory exists

    os.makedirs(save_folder, exist_ok=True)


    # Save artifacts

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

    df = pd.read_parquet(args.input)
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
