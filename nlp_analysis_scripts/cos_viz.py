import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the similarity matrices
# cosine_sim_matrix = np.load('bert_cosine_similarity_matrix.npy')
cosine_sim_matrix = np.load('cosine_sim_matrix_IoA_hfa.npy')

plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim_matrix, annot=False, cmap='coolwarm', square=True, vmin=0.6, vmax=1)
plt.title('Cosine Similarity Heatmap for IoA of HfA Case')
plt.savefig('cos_sim_IoA_hfa_new.png')
plt.show()
