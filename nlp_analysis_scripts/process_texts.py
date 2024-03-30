import os
import hashlib
from preprocessing import preprocess_text, preprocess_text_for_bert

original_dir = '../gpt_responses/iceberg_css_responses/hfa'
preprocessed_dir = '../preprocessed_responses/iceberg_css_preprocessed_responses/hfa'

if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

def preprocess_and_save_file(file_path):
    # Compute a unique filename for the preprocessed file
    base_name = os.path.basename(file_path)
    preprocessed_file_path = os.path.join(preprocessed_dir, base_name)
    
    # Check if the preprocessed file already exists
    if not os.path.exists(preprocessed_file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            original_text = file.read()
            preprocessed_text = " ".join(preprocess_text(original_text))
            with open(preprocessed_file_path, 'w', encoding='utf-8') as preprocessed_file:
                preprocessed_file.write(preprocessed_text)
    return preprocessed_file_path

def preprocess_and_save_file_for_bert(file_path):
    # Compute a unique filename for the preprocessed file
    base_name = os.path.basename(file_path)
    preprocessed_file_path = os.path.join(preprocessed_dir, base_name)
    
    # Check if the preprocessed file already exists
    if not os.path.exists(preprocessed_file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            original_text = file.read()
            # Preprocess text specifically for BERT
            preprocessed_text = preprocess_text_for_bert(original_text)
            with open(preprocessed_file_path, 'w', encoding='utf-8') as preprocessed_file:
                preprocessed_file.write(preprocessed_text)
    return preprocessed_file_path

# Preprocess all files and collect their paths
# preprocessed_paths = [preprocess_and_save_file(os.path.join(original_dir, file))
#                       for file in os.listdir(original_dir) if file.endswith('.txt')]

# Preprocess all files and collect their paths for BERT version
preprocessed_paths = [preprocess_and_save_file(os.path.join(original_dir, file))
                      for file in os.listdir(original_dir) if file.endswith('.txt')]

print(f"Preprocessed files saved to: {preprocessed_dir}")