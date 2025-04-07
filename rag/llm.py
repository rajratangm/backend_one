from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

import os
from dotenv import load_dotenv

load_dotenv()

# Groq uses an OpenAI-compatible schema via LangChain wrapper
groq_chat = ChatGroq(
    model_name="llama3-8b-8192",  # or "llama2-70b-4096", "gemma-7b-it"
    temperature=0.2
)
# groq_chat = ChatGroq(model="llama3-8b-8192", api_key="your_key")

def generate_answer(query: str, context: str) -> str:
    prompt = f"""You are a helpful legal assistant. Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

    response = groq_chat.invoke([HumanMessage(content=prompt)])
    return response.content
