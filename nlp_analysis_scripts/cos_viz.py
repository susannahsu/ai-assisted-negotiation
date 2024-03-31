import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the similarity matrices
# cosine_sim_matrix = np.load('bert_cosine_similarity_matrix.npy')
cosine_sim_matrix = np.load('cosine_sim_matrix_BERT_iceberg_css_hfa.npy')

plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim_matrix, annot=False, cmap='coolwarm', square=True)
plt.title('Cosine Similarity Heatmap for Iceberg & CSS Responses of HfA Case (BERT)')
plt.savefig('cos_sim_BERT_iceberg_css_hfa.png')
plt.show()
