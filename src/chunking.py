"""
#######
"""

import nltk 
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

def chunk_by_tokens(text, chunk_size, overlap):
    """
    Splits text into chunks based on a token count with a specified overlap. 
    
    Parameters: 
        text (str): The cleaned input text to be chunked.
        chunk_size (int): # of tokens per chunk (e.g., 200, 500, 1000). 
        overlap (int): # of tokens that should overlap between consecutive chunks (e.g., 0, 50, 100).
    
    Returns:
        List[str]: A list of text chunks.   
    """
    # tokenizing the text using NLTK's work_tokenize 
    tokens = word_tokenize(text)
    chunks = []

    # step size = number of new tokens in each chunk 
    step = chunk_size - overlap
    if step <= 0: 
        raise ValueError("Overlap must be smaller than chunk size.")
    
    # creating chunks
    for i in range(0, len(tokens), step):
        # extracting a slice of tokens for the current chunk 
        chunk_tokens = tokens[i:i + chunk_size]
        # reconstructing the text chunk from tokens 
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks 

if __name__ == "__main__":
    # Example text for demonstration purposes
    sample_text = (
        "This is an example text to demonstrate token-based chunking. "
        "It will be split into chunks of a specified size with a given overlap. "
        "The quick brown fox jumps over the lazy dog. " * 50  # repeat to increase length
    )
    
    # Define chunk parameters
    chunk_size = 200   # number of tokens per chunk
    overlap = 50       # number of tokens overlapping between chunks
    
    # Get chunks
    chunks = chunk_by_tokens(sample_text, chunk_size, overlap)
    
    print(f"Total chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---\n{chunk[:200]} ...")  # print first 200 characters for preview