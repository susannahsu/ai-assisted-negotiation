import numpy as np

# Load existing matrices
iceberg_css_fwb = np.load('cosine_sim_matrix_BERT_iceberg_css_fwb.npy')
iceberg_css_hfa = np.load('cosine_sim_matrix_BERT_iceberg_css_hfa.npy')
ioa_fwb = np.load('cosine_sim_matrix_BERT_IoA_fwb.npy')
ioa_hfa = np.load('cosine_sim_matrix_BERT_IoA_hfa.npy')
shm_hfa = np.load('cosine_sim_matrix_BERT_shm_hfa.npy')
shm_fwb = np.load('cosine_sim_matrix_BERT_shm_fwb.npy')

# Print loaded matrices
print("Iceberg_css_fwb\n")
print(iceberg_css_fwb)
print("Iceberg_css_hfa\n")
print(iceberg_css_hfa)
print("ioa_fwb\n")
print(ioa_fwb)
print("ioa_hfa\n")
print(ioa_hfa)
print("shm_hfa\n")
print(shm_hfa)
print("shm_fwb\n")
print(shm_fwb)

# Assert matrices are square
assert iceberg_css_fwb.shape[0] == iceberg_css_fwb.shape[1], "Matrix must be square."
assert iceberg_css_hfa.shape[0] == iceberg_css_hfa.shape[1], "Matrix must be square."
assert ioa_fwb.shape[0] == ioa_fwb.shape[1], "Matrix must be square."
assert ioa_hfa.shape[0] == ioa_hfa.shape[1], "Matrix must be square."
assert shm_hfa.shape[0] == shm_hfa.shape[1], "Matrix must be square."
assert shm_fwb.shape[0] == shm_fwb.shape[1], "Matrix must be square."

# Calculate average cosine similarity excluding the diagonal for each matrix
def calculate_average_upper_triangular(matrix):
    indices = np.triu_indices_from(matrix, k=1)
    return np.mean(matrix[indices])

avg_sim_iceberg_css_fwb = calculate_average_upper_triangular(iceberg_css_fwb)
avg_sim_iceberg_css_hfa = calculate_average_upper_triangular(iceberg_css_hfa)
avg_sim_ioa_fwb = calculate_average_upper_triangular(ioa_fwb)
avg_sim_ioa_hfa = calculate_average_upper_triangular(ioa_hfa)
avg_sim_shm_hfa = calculate_average_upper_triangular(shm_hfa)
avg_sim_shm_fwb = calculate_average_upper_triangular(shm_fwb)

# Function to print upper triangle statistics for matrices
def print_upper_stats(matrix, name):
    indices = np.triu_indices_from(matrix, k=1)
    min_val = np.min(matrix[indices])
    max_val = np.max(matrix[indices])
    median_val = np.median(matrix[indices])
    print(f"{name} - Min: {min_val}, Max: {max_val}, Median: {median_val}")

# Print statistics for each matrix
print_upper_stats(iceberg_css_fwb, "Iceberg_css_fwb")
print_upper_stats(iceberg_css_hfa, "Iceberg_css_hfa")
print_upper_stats(ioa_fwb, "IoA FwB")
print_upper_stats(ioa_hfa, "IoA HfA")
print_upper_stats(shm_hfa, "ShM HfA")
print_upper_stats(shm_fwb, "ShM FwB")

# Print average similarities
print("Average cosine similarity (excluding self-similarity):")
print(f"Iceberg CSS FwB: {avg_sim_iceberg_css_fwb}")
print(f"Iceberg CSS HfA: {avg_sim_iceberg_css_hfa}")
print(f"IoA FwB: {avg_sim_ioa_fwb}")
print(f"IoA HfA: {avg_sim_ioa_hfa}")
print(f"ShM HfA: {avg_sim_shm_hfa}")
print(f"ShM FwB: {avg_sim_shm_fwb}")
