# # rag/pipeline.py

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import TextLoader

# import os
# from dotenv import load_dotenv
# load_dotenv()

# # Configuration
# PERSIST_DIR = "vectorstore/"
# EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# CHROMA_COLLECTION = "legal_docs"


# def process_document(file_path: str):
#     # 1. Load raw text
#     loader = TextLoader(file_path)
#     documents = loader.load()

#     # 2. Clean & chunk (LangChain handles it too)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
#     chunks = splitter.split_documents(documents)

#     # 3. Embeddings
#     embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

#     # 4. Store in Chroma
#     vectordb = Chroma.from_documents(
#         documents=chunks,
#         embedding=embedding_model,
#         persist_directory=PERSIST_DIR,
#         collection_name=CHROMA_COLLECTION
#     )
#     vectordb.persist()
#     print(f"✅ Stored {len(chunks)} chunks in ChromaDB")


# def ask_question(query: str):
#     # Load the vector store
#     embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

#     vectordb = Chroma(
#         persist_directory=PERSIST_DIR,
#         embedding_function=embedding_model,
#         collection_name=CHROMA_COLLECTION
#     )

#     # LLM + Retrieval
#     retriever = vectordb.as_retriever(search_kwargs={"k": 3})
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=ChatGroq(temperature=0, model_name="mixtral-8x7b-32768"),
#         chain_type="stuff",
#         retriever=retriever
#     )

#     # Ask question
#     result = qa_chain.run(query)
#     print("\n🔎 Answer:")
#     print(result)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--file", type=str, help="Path to the document to process")
#     parser.add_argument("--ask", type=str, help="Query to run after loading index")
#     args = parser.parse_args()

#     if args.file:
#         process_document(args.file)
#     if args.ask:
#         ask_question(args.ask)
#     if not args.file and not args.ask:  
#         print("Please provide a file to process or a question to ask.")
#     if not os.path.exists(PERSIST_DIR): 
#         os.makedirs(PERSIST_DIR)
#         print(f"Created directory: {PERSIST_DIR}")

# rag/pipeline.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

from chromadb.config import Settings  # NEW

import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
PERSIST_DIR = "vectorstore/"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "legal_docs"

# New Chroma settings object
CHROMA_SETTINGS = Settings(
    persist_directory=PERSIST_DIR,
    anonymized_telemetry=False
)

def process_document(file_path: str):
    # 1. Load raw text
    loader = TextLoader(file_path)
    documents = loader.load()

    # 2. Clean & chunk (LangChain handles it too)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = splitter.split_documents(documents)

    # 3. Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # 4. Store in Chroma (NEW: client_settings)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR,
        collection_name=CHROMA_COLLECTION,
        client_settings=CHROMA_SETTINGS  # <-- NEW
    )
    vectordb.persist()
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")


def ask_question(query: str):
    # Load the vector store
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_name=CHROMA_COLLECTION,
        client_settings=CHROMA_SETTINGS  # <-- NEW
    )

    # LLM + Retrieval
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(temperature=0, model_name="mixtral-8x7b-32768"),
        chain_type="stuff",
        retriever=retriever
    )

    # Ask question
    result = qa_chain.run(query)
    print("\n🔎 Answer:")
    print(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to the document to process")
    parser.add_argument("--ask", type=str, help="Query to run after loading index")
    args = parser.parse_args()

    if args.file:
        process_document(args.file)
    if args.ask:
        ask_question(args.ask)
    if not args.file and not args.ask:
        print("Please provide a file to process or a question to ask.")
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)
        print(f"Created directory: {PERSIST_DIR}")
