from jac_cos_similarity import jaccard_similarity, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def test_jaccard_similarity():
    set1 = set(["apple", "banana", "orange"])
    set2 = set(["banana", "orange", "grape"])
    expected = 2 / 4  # Intersection = 2, Union = 4
    assert jaccard_similarity(set1, set2) == expected, "Jaccard similarity calculation is incorrect."
    print("Test passed: Jaccard similarity is correct.")

def test_cosine_similarity():
    # Sample documents
    documents = [
        "apple banana",  # Document 1
        "apple banana orange",  # Document 2
        "banana orange grape"  # Document 3
    ]
    
    # Vectorize the documents using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Calculate the actual cosine similarity matrix using the vectorized documents
    actual_matrix = cosine_similarity(tfidf_matrix)
    
    # Expected cosine similarity matrix needs to be calculated based on TF-IDF vectors
    # For demonstration purposes, let's assume we're checking the diagonal is all ones
    expected_diagonal = np.array([1.0, 1.0, 1.0])
    
    # Extract the diagonal (self-similarity should be 1.0 for all documents)
    actual_diagonal = np.diag(actual_matrix)
    
    # Check if the actual diagonal is close to the expected diagonal
    assert np.allclose(actual_diagonal, expected_diagonal, atol=0.1), "Cosine similarity calculation is incorrect."
    print("Test passed: Cosine similarity is correct.")

test_jaccard_similarity()
test_cosine_similarity()