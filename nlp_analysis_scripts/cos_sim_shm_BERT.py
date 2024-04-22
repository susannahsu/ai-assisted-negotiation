import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_text(directory):
    text_data = {}
    for i in range(1, 31):  # Assume up to 30 files
        filename = f'stakeholder_responses_{i}.txt'
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                text = file.read().strip()
                text_data[str(i)] = text
    return text_data

def get_bert_embeddings(text_data):
    embeddings = {}
    for key, text in text_data.items():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        # Use the mean of the last hidden state as the embedding
        embedding = outputs.last_hidden_state.mean(1).detach().numpy()
        embeddings[key] = embedding[0]
    return embeddings

def process_data_and_generate_heatmap():
    directory = os.getcwd()
    text_data = extract_text(directory)
    embeddings = get_bert_embeddings(text_data)
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')

    cosine_sim_matrix = pd.DataFrame(index=embeddings_df.index, columns=embeddings_df.index, dtype=float)

    for res1 in embeddings_df.index:
        for res2 in embeddings_df.index:
            if res1 != res2:
                try:
                    cosine_sim_matrix.at[res1, res2] = 1 - cosine(embeddings_df.loc[res1], embeddings_df.loc[res2])
                except Exception as e:
                    cosine_sim_matrix.at[res1, res2] = np.nan  # Set to NaN if an error occurs
            else:
                cosine_sim_matrix.at[res1, res2] = 1.0  # Diagonal entries as 1.0 for identical comparison

    # Filling NaN values with a default value or handling them according to the requirement
    cosine_sim_matrix.fillna(0, inplace=True)  # Optionally replace NaN with 0 or an appropriate value

    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_sim_matrix, annot=False, cmap='coolwarm', vmin=0.6, vmax=1)
    plt.title('BERT Cosine Similarity Heatmap of Stakeholder Responses')
    plt.xticks(ticks=np.arange(len(embeddings_df.index)), labels=embeddings_df.index, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(embeddings_df.index)), labels=embeddings_df.index, rotation=0)
    plt.tight_layout()
    plt.savefig('bert_cosine_similarity_heatmap.png')
    print("Heatmap saved at: bert_cosine_similarity_heatmap.png")

if __name__ == '__main__':
    process_data_and_generate_heatmap()
