import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from cos_similarity_BERT import get_bert_embedding, load_texts
from sklearn.metrics.pairwise import cosine_similarity

practitioner_responses_dir = '../preprocessed_responses/iceberg_css_preprocessed_BERT_practitioner_resp'
preprocessed_dir = '../preprocessed_responses/iceberg_css_preprocessed_BERT_responses/hfa'

def main():
    # Load all practitioner responses and generate embeddings
    practitioner_texts = load_texts(practitioner_responses_dir)
    practitioner_embeddings = torch.stack([get_bert_embedding(text).squeeze() for text in practitioner_texts])

    # Load all GPT responses and generate embeddings
    preprocessed_gpt_resps = load_texts(preprocessed_dir)
    gpt_resp_embeddings = torch.stack([get_bert_embedding(text).squeeze() for text in preprocessed_gpt_resps])

    # Calculate similarity scores
    def compare_practitioners_to_gpt(practitioner_embeddings, gpt_resp_embeddings):
        # Flatten embeddings to ensure they are 2D (examples, features)
        practitioner_embeddings = practitioner_embeddings.view(practitioner_embeddings.size(0), -1)
        gpt_resp_embeddings = gpt_resp_embeddings.view(gpt_resp_embeddings.size(0), -1)
        similarities = cosine_similarity(practitioner_embeddings, gpt_resp_embeddings)
        return np.mean(similarities, axis=0)  # Average across all practitioners

    similarity_scores = compare_practitioners_to_gpt(practitioner_embeddings, gpt_resp_embeddings)
    print("Average cosine similarity scores between the practitioners and each GPT response:", similarity_scores)
    np.save('ioa_hfa_benchmark_similarity_scores.npy', similarity_scores)
    average_score = np.mean(similarity_scores)

    # Visualizing the similarity scores
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, len(similarity_scores) + 1), similarity_scores, color='skyblue')
    # Label each bar with its value
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.03, f'{yval:.2f}', ha='center', va='bottom', color='black', fontsize=6)
    
    plt.axhline(y=average_score, color='red', linestyle='--')
    plt.text(len(similarity_scores), average_score - 0.02, f'Average: {average_score:.2f}', color='red', va='top', ha='right')
    
    # Set y-axis limit
    plt.ylim(0, 1.0)

    plt.xlabel('GPT Response Number')
    plt.ylabel('Average Cosine Similarity Score Across 3 Practitioners')
    plt.title('Average Cosine Similarity Scores Between Practitioners (3) and GPT Responses (Iceberg/CSS, HfA)')
    plt.grid(True)

    plt.savefig('benchmark_iceberg_css_hfa.png')
    plt.show()

if __name__ == "__main__":
    main()
