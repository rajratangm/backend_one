from rag.retriever import query_db  # This uses ChromaDB
from rag.llm import generate_answer  # This will call your language model

def generate_response(user_query: str, k: int = 3) -> str:
    # Step 1: Retrieve relevant chunks
    retrieved_chunks = query_db(user_query, k=k)

    # Step 2: Prepare context
    context = "\n\n".join(retrieved_chunks)

    # Step 3: Generate answer using LLM
    response = generate_answer(user_query, context)

    return response
