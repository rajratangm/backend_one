from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def clean_legal_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove page numbers or headers (example patterns)
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\\f', '', text)  # form feed characters

    # Normalize unicode (e.g., smart quotes)
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")

    # Remove any unwanted special characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.lower()  # Convert to lowercase

    return text.strip()

def smart_chunk(text):
    text = clean_legal_text(text)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)