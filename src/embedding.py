'''

import os
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

from sentence_transformers import SentenceTransformer
import numpy as np 

# loading 3 different embedding models 
model_all_minilm = SentenceTransformer('all-MiniLM-L6-v2')
model_all_mpnet = SentenceTransformer('all-mpnet-base-v2')
model_instructorxl = SentenceTransformer('hkunlp/instructor-xl')

def get_embedding(text, model_name):
    """
    Gets the embedding for a given text using one of 3 embedding models.

    Parameters: 
        text (str): The input text to embed.
        model_name (str): name of the model to use: 
            - 'all-MiniLM-L6-v2'
            - 'all-mpnet-base-v2'
            - 'hkunlp/instructor-xl'
    Returns: 
        List[float]: The embedding vector for the input text.
    """
    if model_name == 'all-MiniLM-L6-v2':
        embedding = model_all_minilm.encode(text)
    elif model_name == 'all-mpnet-base-v2':
        embedding = model_all_mpnet.encode(text)
    elif model_name == 'hkunlp/instructor-xl':
        embedding = model_instructorxl.encode(text)
    else: 
        raise ValueError(f"Invalid model name: {model_name}")
    
    return embedding.tolist()

# Example usage:
if __name__ == "__main__":
    sample_text = "This is a sample text to compute an embedding."
    emb1 = get_embedding(sample_text, model_name='all-MiniLM-L6-v2')
    emb2 = get_embedding(sample_text, model_name='all-mpnet-base-v2')
    emb3 = get_embedding(sample_text, model_name='hkunlp/instructor-xl')
    print("Embedding from all-MiniLM-L6-v2:", emb1[:5])
    print("Embedding from all-mpnet-base-v2:", emb2[:5])
    print("Embedding from InstructorXL (placeholder):", emb3[:5])
'''