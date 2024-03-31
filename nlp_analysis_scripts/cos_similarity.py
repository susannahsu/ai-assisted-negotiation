import os
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words_list = list(set(stopwords.words('english')))

preprocessed_dir = '../preprocessed_responses/IoA_preprocessed_responses/hfa'

def load_preprocessed_texts(directory):
    texts = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def calculate_cosine_similarity(texts):
    # Adjust TF-IDF vectorization to include n-grams, but seems like unigrams best
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words_list)
    # Joining lemmatized tokens
    tfidf_matrix = vectorizer.fit_transform(texts)  
    # Calculate the cosine similarity between all pairs of documents
    # as represented by their TF-IDF vectors
    return cosine_similarity(tfidf_matrix)


preprocessed_texts = load_preprocessed_texts(preprocessed_dir)

# Cosine similarity can be computed directly on the list of preprocessed texts 
# since it operates on the TF-IDF matrix generated from the entire corpus.
cosine_sim_matrix = calculate_cosine_similarity(preprocessed_texts)

# Saving the similarity matrices
name = 'cosine_sim_matrix_IoA_hfa.npy'
np.save(name, cosine_sim_matrix)
