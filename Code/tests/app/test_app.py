# %% Imports

import os
import pytest
from unittest.mock import MagicMock, patch # Use unittest.mock directly or via pytest-mock's mocker

from app.app import load_artifacts, predict_policy_area, get_top_contributing_words

from scipy.sparse import csr_matrix
import pandas as pd


# %% Mock data structures and functions needed

@pytest.fixture(autouse=True)
def mock_streamlit():
    with patch('app.app.st') as mock_st:
        mock_st.error = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.cache_resource = lambda func: func
        yield mock_st


@pytest.fixture
def mock_predict_components():
    """Provides mock model, vectorizer, and label encoder for prediction tests."""
    mock_model = MagicMock()
    mock_vectorizer = MagicMock()
    mock_label_encoder = MagicMock()

    mock_model.predict.return_value = [1]

    mock_tfidf_matrix = csr_matrix(([1.0], ([0], [5])), shape=(1, 10))
    mock_vectorizer.transform.return_value = mock_tfidf_matrix
    mock_label_encoder.inverse_transform.return_value = ["PolicyAreaA"]
    return mock_model, mock_vectorizer, mock_label_encoder

def test_predict_policy_area_success(mock_predict_components):
    """Test successful prediction."""

    mock_model, mock_vectorizer, mock_label_encoder = mock_predict_components
    text = "This is relevant text."

    label, tfidf = predict_policy_area(text, mock_model, mock_vectorizer, mock_label_encoder)

    # Assert
    assert label == "PolicyAreaA"
    assert tfidf is mock_vectorizer.transform.return_value # Check it returned the matrix
    mock_vectorizer.transform.assert_called_once_with([text])
    mock_model.predict.assert_called_once()
    mock_label_encoder.inverse_transform.assert_called_once_with([1])


def test_predict_policy_area_no_text(mock_predict_components, mock_streamlit):
    """Test prediction with empty or whitespace text."""

    mock_model, mock_vectorizer, mock_label_encoder = mock_predict_components
    empty_text = ""
    whitespace_text = "   \t\n "

    label1, tfidf1 = predict_policy_area(empty_text, mock_model, mock_vectorizer, mock_label_encoder)
    label2, tfidf2 = predict_policy_area(whitespace_text, mock_model, mock_vectorizer, mock_label_encoder)

    assert label1 is None and tfidf1 is None
    assert label2 is None and tfidf2 is None
    assert mock_streamlit.warning.call_count == 2 # Called for both empty and whitespace
    mock_vectorizer.transform.assert_not_called() # Should not attempt transform


def test_predict_policy_area_no_features(mock_predict_components, mock_streamlit):
    """Test prediction when TF-IDF vectorization yields no features (e.g., only stop words)."""

    mock_model, mock_vectorizer, mock_label_encoder = mock_predict_components

    zero_feature_matrix = csr_matrix((1, 10))
    assert zero_feature_matrix.nnz == 0
    mock_vectorizer.transform.return_value = zero_feature_matrix
    text = "only stop words"

    label, tfidf = predict_policy_area(text, mock_model, mock_vectorizer, mock_label_encoder)

    assert label is None and tfidf is None
    mock_vectorizer.transform.assert_called_once_with([text])
    mock_model.predict.assert_not_called()
    mock_streamlit.warning.assert_called_once()


@pytest.fixture
def mock_explain_components():
    """Provides mock components for explainability tests."""
    mock_model = MagicMock(spec=['coef_'])
    mock_vectorizer = MagicMock(spec=['get_feature_names_out'])
    mock_label_encoder = MagicMock(spec=['classes_'])

    mock_label_encoder.classes_ = ['PolicyAreaA', 'PolicyAreaB']

    mock_model.coef_ = [
        [0.1, -0.2, 0.5, 0.0, -0.1],
        [-0.1, 0.2, -0.5, 0.1, 0.1]
    ]
    mock_vectorizer.get_feature_names_out.return_value = ['word1', 'word2', 'word3', 'word4', 'word5']

    indices = [1, 2, 4]
    data = [0.8, 0.6, 0.7]
    mock_tfidf = csr_matrix((data, ([0]*len(indices), indices)), shape=(1, 5))

    return mock_model, mock_vectorizer, mock_label_encoder, mock_tfidf


def test_get_top_contributing_words_success(mock_explain_components):
    """Test successful calculation of contributing words."""

    mock_model, mock_vectorizer, mock_label_encoder, mock_tfidf = mock_explain_components
    prediction_label = 'PolicyAreaA'
    top_n = 3

    df = get_top_contributing_words(
        prediction_label, mock_tfidf, mock_model, mock_vectorizer, mock_label_encoder, top_n
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == top_n
    assert list(df.columns) == ['Word', 'Contribution Score']
    assert df['Word'].tolist() == ['word3', 'word5', 'word2']
    pd.testing.assert_series_equal(
        df['Contribution Score'],
        pd.Series([0.30, -0.07, -0.16], name='Contribution Score'),
        check_index=False,
        check_dtype=False,
        atol=1e-5
    )


def test_get_top_contributing_words_none_input():
    """Test when prediction_label or text_tfidf is None."""
    # Arrange
    mock_model, mock_vectorizer, mock_label_encoder, mock_tfidf = MagicMock(), MagicMock(), MagicMock(), MagicMock()

    # Act
    df1 = get_top_contributing_words(None, mock_tfidf, mock_model, mock_vectorizer, mock_label_encoder)
    df2 = get_top_contributing_words("Label", None, mock_model, mock_vectorizer, mock_label_encoder)

    # Assert
    assert isinstance(df1, pd.DataFrame) and df1.empty
    assert list(df1.columns) == ['Word', 'Contribution Score']
    assert isinstance(df2, pd.DataFrame) and df2.empty
    assert list(df2.columns) == ['Word', 'Contribution Score']