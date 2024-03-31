import os
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Specify the directory containing preprocessed text files
preprocessed_dir = '../preprocessed_responses/IoA_preprocessed_BERT_responses/hfa'

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set the model to evaluation mode

def get_bert_embedding(text):
    """
    Generates a document embedding by averaging all token embeddings produced by BERT.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    embeddings = torch.mean(embeddings, dim=1)
    return embeddings

def load_texts(directory):
    """
    Loads and returns a list of texts from the specified directory.
    """
    texts = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def main():
    # Load preprocessed texts
    texts = load_texts(preprocessed_dir)
    
    # Generate embeddings for each text
    embeddings = torch.stack([get_bert_embedding(text) for text in texts]).squeeze()
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    name = 'cosine_sim_matrix_BERT_IoA_hfa.npy'
    # Optionally, save the similarity matrix for later use
    np.save(name, similarity_matrix)
    
    # Print the shape of the similarity matrix as a basic check
    print("Similarity matrix shape:", similarity_matrix.shape)

if __name__ == "__main__":
    main()
