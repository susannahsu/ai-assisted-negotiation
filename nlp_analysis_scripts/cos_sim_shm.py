import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# Function to extract coordinates from the script
def extract_stakeholder_coordinates(directory, coordinate_pattern):
    stakeholders_data = {}
    for i in range(1, 31):  # Assuming there are up to 30 files
        filename = f'stakeholder_responses_{i}.txt'
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):  # Check if the file exists
            with open(file_path, 'r') as file:
                content = file.read()
                raw_coords = re.findall(coordinate_pattern, content)
                processed_coords = {}
                for name, coord_str in raw_coords:
                    try:
                        coords = tuple(map(float, coord_str.split(', ')))
                        if len(coords) != 2 or coords == (0.0, 0.0):
                            continue  # Skip coordinates that are (0,0)
                        processed_coords[name] = coords
                    except ValueError as e:
                        print(f"Error converting coordinates for {name}: {e}")
                stakeholders_data[str(i)] = processed_coords  # Use only the number as the key
    return stakeholders_data

# Function to calculate centroid
def calculate_centroid(coordinates):
    coords = np.array(coordinates)
    return np.mean(coords, axis=0)

def process_data_and_generate_heatmap():
    directory = os.getcwd()
    coordinate_pattern = re.compile(r"'([^']+)':\s*\(([^)]+)\)")
    stakeholders_data = extract_stakeholder_coordinates(directory, coordinate_pattern)
    centroids = {response: calculate_centroid(list(stakeholders.values()))
                 for response, stakeholders in stakeholders_data.items() if stakeholders}

    centroids_df = pd.DataFrame.from_dict(centroids, orient='index', columns=['X', 'Y']).dropna()

    cosine_sim_matrix = pd.DataFrame(np.nan, index=centroids_df.index, columns=centroids_df.index)

    for res1 in centroids_df.index:
        for res2 in centroids_df.index:
            if res1 != res2:
                vector_a = centroids_df.loc[res1]
                vector_b = centroids_df.loc[res2]
                cosine_sim_matrix.at[res1, res2] = 1 - cosine(vector_a, vector_b)
            else:
                cosine_sim_matrix.at[res1, res2] = 1.0  # Diagonal entries as 1.0 for identical comparison

    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_sim_matrix, annot=False, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Cosine Similarity Heatmap of Stakeholder Centroids (HFA)')
    plt.xticks(ticks=np.arange(len(centroids_df.index)), labels=centroids_df.index, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(centroids_df.index)), labels=centroids_df.index, rotation=0)
    plt.tight_layout()
    plt.savefig('cosine_similarity_heatmap.png')
    print("Heatmap saved at: cosine_similarity_heatmap.png")

if __name__ == '__main__':
    process_data_and_generate_heatmap()
