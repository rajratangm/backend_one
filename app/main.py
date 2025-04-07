# # app/main.py

# import streamlit as st
# from rag.generator import generate_response  # <-- Fix import path

# st.set_page_config(page_title="Legal Advice Assistant", layout="centered")
# st.title("🧑‍⚖️ Legal Advice Assistant")
# st.markdown("Ask me any legal question and I’ll try to help!")

# query = st.text_input("Enter your legal question:")

# if query:
#     with st.spinner("Thinking..."):
#         answer = generate_response(query)  # Use actual RAG pipeline
#     st.success(f"You asked: `{query}`")
#     st.info(f"Answer: {answer}")
# else:
#     st.warning("Please enter a question to get started.")


# app/main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.generator import generate_response  # Adjusted import path to match project structure

app = FastAPI()

# Allow React frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response schema
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer = generate_response(request.query)
    return {"answer": answer}

# if __name__ == "__main__":
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860)
#     import uvicorn
#     import os

#     # port = int(os.environ.get("PORT", 8000))  # Render sets PORT env variable
#     uvicorn.run("main:app", host="0.0.0.0", 
#                 # port=port, 
#                 reload=False)

