import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove markdown formatting characters
    text = re.sub(r'[\*\#]', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization with POS tagging
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    return lemmatized

def preprocess_text_for_bert(text):
    """
    Simplified preprocessing for use with BERT:
    - Lowercases the text (if using an uncased model).
    - Removes URLs, as they're usually not informative for NLP tasks.
    - Optionally, you can remove or replace numbers and other specific elements.
    """
    # Optionally lower case the text if using an uncased BERT model
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove markdown formatting characters
    text = re.sub(r'[\*\#]', '', text)
    
    # Replace numbers with a special token (or remove them)
    text = re.sub(r'\d+', '[NUMBER]', text)
    
    # Minimal cleanup, retain punctuation as BERT can utilize it.
    
    return text