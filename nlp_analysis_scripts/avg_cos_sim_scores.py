import numpy as np

iceberg_css_fwb = np.load('cosine_sim_matrix_BERT_iceberg_css_fwb.npy')
iceberg_css_hfa = np.load('cosine_sim_matrix_BERT_iceberg_css_hfa.npy')
ioa_fwb = np.load('cosine_sim_matrix_BERT_IoA_fwb.npy')
ioa_hfa = np.load('cosine_sim_matrix_BERT_IoA_hfa.npy')

print("Iceberg_css_fwb\n")
print(iceberg_css_fwb)
print("Iceberg_css_hfa\n")
print(iceberg_css_hfa)
print("ioa_fwb\n")
print(ioa_fwb)
print("ioa_hfa\n")
print(ioa_hfa)

assert iceberg_css_fwb.shape[0] == iceberg_css_fwb.shape[1], "Matrix must be square."
assert iceberg_css_hfa.shape[0] == iceberg_css_hfa.shape[1], "Matrix must be square."
assert ioa_fwb.shape[0] == ioa_fwb.shape[1], "Matrix must be square."
assert ioa_hfa.shape[0] == ioa_hfa.shape[1], "Matrix must be square."

# calculate the average cosine similarity, excluding the diagonal
i_upper_iceberg_css_fwb = np.triu_indices_from(iceberg_css_fwb, k=1)
i_upper_iceberg_css_hfa = np.triu_indices_from(iceberg_css_hfa, k=1)
i_upper_ioa_fwb = np.triu_indices_from(ioa_fwb, k=1)
i_upper_ioa_hfa = np.triu_indices_from(ioa_hfa, k=1)

avg_sim_iceberg_css_fwb = np.mean(iceberg_css_fwb[i_upper_iceberg_css_fwb])
avg_sim_iceberg_css_hfa = np.mean(iceberg_css_hfa[i_upper_iceberg_css_hfa])
avg_sim_ioa_fwb = np.mean(ioa_fwb[i_upper_ioa_fwb])
avg_sim_ioa_hfa = np.mean(ioa_hfa[i_upper_ioa_hfa])

def print_upper_stats(matrix, indices, name):
    upper_triangle_values = matrix[indices]
    min_val = np.min(upper_triangle_values)
    max_val = np.max(upper_triangle_values)
    median_val = np.median(upper_triangle_values)
    print(f"{name} - Upper Triangle Min: {min_val}, Max: {max_val}, Median: {median_val}")

print_upper_stats(iceberg_css_fwb, i_upper_iceberg_css_fwb, "Iceberg_css_fwb")
print_upper_stats(iceberg_css_hfa, i_upper_iceberg_css_hfa, "Iceberg_css_hfa")
print_upper_stats(ioa_fwb, i_upper_ioa_fwb, "ioa_fwb")
print_upper_stats(ioa_hfa, i_upper_ioa_hfa, "ioa_hfa")

print("Iceberg CSS FwB - Average cosine similarity (excluding self-similarity):", avg_sim_iceberg_css_fwb)
print("Iceberg CSS HfA - Average cosine similarity (excluding self-similarity):", avg_sim_iceberg_css_hfa)
print("IoA FwB - Average cosine similarity (excluding self-similarity):", avg_sim_ioa_fwb)
print("IoA HfA - Average cosine similarity (excluding self-similarity):", avg_sim_ioa_hfa)