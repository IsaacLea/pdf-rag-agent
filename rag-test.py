import os
import re
import pdfplumber
import openai
import pinecone
from pinecone import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


MODEL = "text-embedding-3-small"
API_KEY_PINECONE = os.getenv('API_KEY_PINECONE')
API_KEY_OPENAI = os.getenv('API_KEY_OPENAI')

pc = Pinecone(
        # api_key=os.environ.get("PINECONE_API_KEY")
        api_key=API_KEY_PINECONE
    )

def create_embedding(query):
    from openai import OpenAI

    # Get OpenAI api key from platform.openai.com
    # openai_api_key = os.getenv('OPENAI_API_KEY') or 'sk-...'

    # Instantiate the OpenAI client
    client = OpenAI(api_key=API_KEY_OPENAI)

    # res = client.embeddings.create(
    #     input=query,
    #     model="text-embedding-3-small"
    # )

    # Create an embedding
    res = client.embeddings.create(
      model=MODEL,
      input=[query],
    )
    return res.data[0].embedding

query = (
    "Where did apples originate from?"
)

xq = create_embedding(query)

print(xq)

# Retrieve from Pinecone
# Get relevant contexts (including the questions)
index_name = "pdf-rag1"
dense_index = pc.Index(index_name)
query_results = dense_index.query(
    namespace="example-namespace",
    vector=xq, top_k=2, include_metadata=True)
print(query_results)

