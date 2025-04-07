# import chromadb
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
# from typing import List

# # Initialize ChromaDB client
# client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="vectorstore/"))

# # Create or get collection
# collection = client.get_or_create_collection(name="legal_docs")

# model = SentenceTransformer("all-MiniLM-L6-v2")

# def embed_texts(texts: List[str]) -> List[List[float]]:
#     # Filter out empty or too short ones
#     texts = [t for t in texts if len(t.strip()) > 20]
#     return model.encode(texts, convert_to_tensor=False)

# def store_embeddings(texts: List[str]):
#     embeddings = embed_texts(texts)
    
#     # Add to ChromaDB with IDs
#     ids = [f"chunk-{i}" for i in range(len(texts))]
#     collection.add(documents=texts, embeddings=embeddings, ids=ids)

#     print(f"Stored {len(texts)} chunks in ChromaDB.")

# def query_db(query: str, k=3):
#     query_embedding = model.encode([query], convert_to_tensor=False)[0]
    
#     results = collection.query(query_embeddings=[query_embedding], n_results=k)
    
#     return results["documents"][0]  # list of top-k text chunks

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List
import uuid  # For generating unique IDs

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="vectorstore/")


# Create or get the collection
collection = client.get_or_create_collection(name="legal_docs")

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Filter out empty or too short texts
    texts = [t for t in texts if len(t.strip()) > 20]
    if not texts:
        return []
    return model.encode(texts, convert_to_tensor=False)

def store_embeddings(texts: List[str]):
    texts = [t for t in texts if len(t.strip()) > 20]
    if not texts:
        print("No valid text chunks to store.")
        return

    embeddings = embed_texts(texts)

    # Generate unique IDs for each document
    ids = [str(uuid.uuid4()) for _ in texts]

    # Store in Chroma
    collection.add(documents=texts, embeddings=embeddings, ids=ids)

    print(f"✅ Stored {len(texts)} chunks in ChromaDB.")

def query_db(query: str, k=3):
    if not query or len(query.strip()) < 3:
        print("Query too short or empty.")
        return []

    query_embedding = model.encode([query], convert_to_tensor=False)[0]

    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    return results.get("documents", [[]])[0]  # Return list of top-k chunks
