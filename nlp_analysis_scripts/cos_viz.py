import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the similarity matrices
# cosine_sim_matrix = np.load('bert_cosine_similarity_matrix.npy')
cosine_sim_matrix1 = np.load('cosine_sim_matrix_BERT_iceberg_css_fwb.npy')
cosine_sim_matrix2 = np.load('cosine_sim_matrix_BERT_iceberg_css_hfa.npy')
cosine_sim_matrix3 = np.load('cosine_sim_matrix_BERT_IoA_fwb.npy')
cosine_sim_matrix3 = np.load('cosine_sim_matrix_BERT_IoA_hfa.npy')

plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(cosine_sim_matrix1, annot=False, cmap='coolwarm', square=True, vmin=0.94, vmax=1)
plt.title('Cosine Similarity Heatmap for Iceberg & CSS Responses of FwB Case (BERT)')
plt.savefig('cos_sim_BERT_iceberg_css_fwb.png')
plt.show()

plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(cosine_sim_matrix2, annot=False, cmap='coolwarm', square=True, vmin=0.94, vmax=1)
plt.title('Cosine Similarity Heatmap for Iceberg & CSS Responses of HfA Case (BERT)')
plt.savefig('cos_sim_BERT_iceberg_css_hfa.png')
plt.show()

plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(cosine_sim_matrix3, annot=False, cmap='coolwarm', square=True, vmin=0.94, vmax=1)
plt.title('Cosine Similarity Heatmap for IoA Responses of FwB Case (BERT)')
plt.savefig('cos_sim_BERT_IoA_css_fwb.png')
plt.show()

plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(cosine_sim_matrix3, annot=False, cmap='coolwarm', square=True, vmin=0.94, vmax=1)
plt.title('Cosine Similarity Heatmap for IoA Responses of HfA Case (BERT)')
plt.savefig('cos_sim_BERT_IoA_hfa.png')
plt.show()