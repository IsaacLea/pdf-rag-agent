import os
import re
import pdfplumber
import openai
import pinecone
from pinecone import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

API_KEY_PINECONE = os.getenv('API_KEY_PINECONE')
API_KEY_OPENAI = os.getenv('API_KEY_OPENAI')

# Initialize OpenAI
openai.api_key = API_KEY_OPENAI
MODEL = "text-embedding-ada-002"

pinecone = Pinecone(
        api_key=API_KEY_PINECONE
    )

# Initialize Pinecone
# pinecone.init(api_key="pcsk_2k7YqV_pK3rNqKsy1MFVnC2C2u2c6QUupm413vdvx9afsD5crQfQJsNrzpPDRYN8PSdyt", environment='pdf-rag')

# Create a dense index with integrated embedding
index_name = "pdf-rag"
if not pinecone.has_index(index_name):
    pinecone.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )

# Define a function to preprocess text
def preprocess_text(text):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text)
    return text

def process_pdf(file_path):
    # create a loader
    loader = PyPDFLoader(file_path)
    # load your data
    data = loader.load()
    # Split your data up into smaller documents with Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(data)
    # Convert Document objects into strings
    texts = [str(doc) for doc in documents]
    return texts

# Define a function to create embeddings
def create_embeddings(texts):
    embeddings_list = []
    for text in texts:
        res = openai.Embedding.create(input=[text], engine=MODEL)
        embeddings_list.append(res['data'][0]['embedding'])
    return embeddings_list

# Define a function to upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(index, embeddings, ids):
    index.upsert(vectors=[(id, embedding) for id, embedding in zip(ids, embeddings)])

# Process a PDF and create embeddings
file_path = "TestPDF1.pdf"  # Replace with your actual file path
texts = process_pdf(file_path)
embeddings = create_embeddings(texts)

# Upsert the embeddings to Pinecone
upsert_embeddings_to_pinecone(index, embeddings, [file_path])