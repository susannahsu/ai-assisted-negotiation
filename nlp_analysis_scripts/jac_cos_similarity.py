import hashlib
import os

from preprocesssing import preprocess_text

preprocessed_dir = '../preprocessed_reponses/IoA_preprocessed_responses/fwb'

if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

