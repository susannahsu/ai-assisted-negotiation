import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the similarity matrices
# cosine_sim_matrix = np.load('bert_cosine_similarity_matrix.npy')
cosine_sim_matrix = np.load('bert_cosine_similarity_matrix_iceberg_css_fwb.npy')
# jaccard_sim_matrix = np.load('jaccard_sim_matrix.npy')

plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim_matrix, annot=False, cmap='coolwarm', square=True)
plt.title('Cosine Similarity Heatmap for Iceberg CSS Responses of FWB Case (BERT)')
plt.savefig('cos_sim_BERT_iceberg_css_fwb.png')
plt.show()
