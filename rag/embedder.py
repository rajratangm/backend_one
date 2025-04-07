from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(texts: List[str]):
    # Filter out very short or empty chunks (they don’t add meaning)
    cleaned_texts = [t.strip() for t in texts if len(t.strip()) > 5]
    
    if not cleaned_texts:
        return np.array([])

    # Generate dense vector embeddings
    embeddings = model.encode(cleaned_texts, convert_to_tensor=False, show_progress_bar=True)

    return embeddings, cleaned_texts  # return both to track index
