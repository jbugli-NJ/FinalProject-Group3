# %% Imports

import sys
import subprocess
import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder


# %% Provide paths to .joblib files

ARTIFACTS_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, 'vectorizer.joblib')
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'label_encoder.joblib')
TOP_N_WORDS = 15


# %% Define function to load artifacts

@st.cache_resource
def load_artifacts(model_path: str, vectorizer_path: str, label_encoder_path: str):
    """
    Loads the model, vectorizer, and label encoder.

    :param str model_path: The path to the stored model .joblib.

    :param str model_path: The path to the stored TF-IDF vectorizer .joblib.

    :param str model_path: The path to the stored label encoder .joblib.
    """

    # Check if the directory is found

    if os.path.exists(ARTIFACTS_DIR) == False:
        print('Specified artifact folder not found!')
        return None, None, None

    # Attempt to load models

    try:

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        label_encoder = joblib.load(label_encoder_path)

        return model, vectorizer, label_encoder

    except FileNotFoundError:

        st.error(f"Could not find file in: {ARTIFACTS_DIR}"
                 f"Ensure '{os.path.basename(model_path)}', '{os.path.basename(vectorizer_path)}', "
                 f"and '{os.path.basename(label_encoder_path)}' exist.")
        return None, None, None
    
    except Exception as e:
        st.error(f"Unexpected artifact loading error: {e}")
        return None, None, None


# %% Define inference function

def predict_policy_area(
    text: str,
    model,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder
    ):
    """
    Predicts the policy area for the given text.

    :param str text: The input text to evaluate.

    :param model: The model to use for predictions.

    :param TfidfVectorizer: A TF-IDF vectorizer.

    :param LabelEncoder: A label encoder.

    :return: A tuple with 2 items (both None on failure):
    
        - The decoded prediction label
        - A TF-IDF matrix
    """

    # Return nothing if no text is available

    if not text or not text.strip():
        st.warning('Not input text was provided!')
        return None, None

    try:

        # Transform text using the vectorizer, making sure features exist

        text_tfidf = vectorizer.transform([text])


        # Make sure features exist

        if text_tfidf.nnz == 0:
            st.warning((
                'Input text contains no known features '
                '(e.g., only stop words or unknown words)'
            ))
            return None, None


        # Predict and decode the label

        prediction_encoded = model.predict(text_tfidf)

        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

        return prediction_label, text_tfidf
    

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


# %% Define function to support explainability

def get_top_contributing_words(
    text,
    prediction_label,
    text_tfidf, model,
    vectorizer,
    label_encoder,
    top_n=10
    ):
    """
    Identifies the biggest contributors to the output class.

        - Contribution is estimated by coefficient * tfidf_score
    """
    if prediction_label is None or text_tfidf is None:
        return pd.DataFrame(columns=['Word', 'Contribution Score'])

    try:

        class_coefficients = model.coef_[0]

        # Multi-class:
        # predicted_class_index = label_encoder.transform([prediction_label])[0]
        # Multi-class: class_coefficients = model.coef_[predicted_class_index]

        # Get words from the vectorizer

        feature_names = vectorizer.get_feature_names_out()

        # Get the indices and TF-IDF scores for input text vector features
        # NOTE: text_tfidf is a matrix of shape (1, n_features)

        feature_indices = text_tfidf.indices
        tfidf_scores = text_tfidf.data


        # Calculate contribution score for each word present in the input text

        word_contributions = [
            {
                'Word': feature_names[idx],
                'Contribution Score': (class_coefficients[idx] * tfidf_score)
            }
            for idx, tfidf_score in zip(feature_indices, tfidf_scores)
        ]


        # Create a DataFrame, sorting by descending contribution score
        # TODO: Word cloud?

        contributions_df = pd.DataFrame(word_contributions)


        # Only consider positive contributions for "why" this class was chosen

        contributions_df = contributions_df[contributions_df['Contribution Score'] > 0]
        contributions_df = contributions_df.sort_values(by='Contribution Score', ascending=False)

        return contributions_df.head(top_n)

    except Exception as e:
        st.error(f"Explainability calculation error: {e}")
        return pd.DataFrame(columns=['Word', 'Contribution Score'])


# %% Define UI setup / input processing function

def initialize():

    # Define app layout

    st.set_page_config(page_title='Policy Area Classifier', layout='wide')
    st.title('Policy Area Classifier')
    st.markdown('Enter text (e.g., from a bill) to predict its policy area.')


    # Load artifacts
    # TODO: Argparse

    model, vectorizer, label_encoder = load_artifacts(
        MODEL_PATH,
        VECTORIZER_PATH,
        LABEL_ENCODER_PATH
    )


    # Set up interface elements

    if model and vectorizer and label_encoder:
        st.success('Model artifacts loaded successfully!')
        st.markdown((
            f"Model trained for **{len(label_encoder.classes_)}** policy areas: "
            f"'`{'`, `'.join(label_encoder.classes_)}`"
        ))


        # Input

        input_text = st.text_area(
            'Congressional Bill / Resolution Text',
            height=250,
            placeholder='Place legislative text here ...'
        )

        # Prediction button

        predict_button = st.button('Predict Policy Area', type='primary')

        if predict_button:
            if len(input_text.strip()) > 0:
                with st.spinner('Analyzing text ...'):

                    # Attempt prediction

                    predicted_label, text_tfidf_vector = predict_policy_area(input_text, model, vectorizer, label_encoder)

                    if predicted_label:
                        st.subheader('Prediction Result:')
                        st.markdown(f"The predicted policy area is: **{predicted_label}**")

                        # Explainability

                        st.subheader('Explanation:')
                        st.markdown(f"Top {TOP_N_WORDS} prediction contributors: ")

                        top_words_df = get_top_contributing_words(
                            input_text,
                            predicted_label,
                            text_tfidf_vector,
                            model,
                            vectorizer,
                            label_encoder,
                            top_n=TOP_N_WORDS
                        )

                        # TODO: Word cloud?

                        if not top_words_df.empty:

                            st.dataframe(
                                top_words_df.style.format({'Contribution Score': '{:.4f}'}),
                                use_container_width=True,
                                hide_index=True
                                )

                        else:
                            st.warning('No significant contributors identified!')

            else:
                st.warning("Please enter some text before predicting.")

    else:
        st.error('Model artifacts could not be loaded!')


# Define steps taken when run directly (used in CLI script)

if __name__ == '__main__':
    initialize()