import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from cos_similarity_BERT import get_bert_embedding, load_texts
from sklearn.metrics.pairwise import cosine_similarity

practitioner_response = '../preprocessed_responses/ioa_preprocessed_BERT_practitioner_resp'
preprocessed_dir = '../preprocessed_responses/IoA_preprocessed_BERT_responses/hfa'


def main():
    preprocessed_practitioner = load_texts(practitioner_response)[0]
    preprocessed_gpt_resps = load_texts(preprocessed_dir)

    # Generate embeddings for each text
    practitioner_embedding = get_bert_embedding(preprocessed_practitioner).squeeze()
    gpt_resp_embeddings = torch.stack([get_bert_embedding(text) for text in preprocessed_gpt_resps]).squeeze()
    

    def compare_single_practitioner_to_gpt(practitioner_embedding, gpt_resp_embeddings):
        # Reshape the practitioner embedding to match dimensions if necessary
        practitioner_embedding = practitioner_embedding.unsqueeze(0)
        similarities = cosine_similarity(practitioner_embedding, gpt_resp_embeddings)
        return similarities[0]  # Flatten the array to get a simple list of similarities

    similarity_scores = compare_single_practitioner_to_gpt(practitioner_embedding, gpt_resp_embeddings)
    print("Cosine similarity scores between the practitioner and each GPT response:", similarity_scores)
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
    plt.ylabel('Cosine Similarity Score')
    plt.title('Cosine Similarity Scores Between Practitioner and GPT Responses (IoA, HfA)')
    plt.grid(True)

    plt.savefig('benchmark_ioa_hfa.png')
    plt.show()

if __name__ == "__main__":
    main()