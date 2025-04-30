import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

API_KEY_PINECONE = os.getenv('API_KEY_PINECONE')
# API_KEY_OPENAI = os.getenv('API_KEY_OPENAI')
MODEL = "multilingual-e5-large"

load_dotenv()
# openai = OpenAI(api_key=API_KEY_OPENAI)
pc = Pinecone(api_key=API_KEY_PINECONE)

index_name = "tiger-900-sm"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":MODEL,
            "field_map":{"text": "chunk_text"}
        }
    )

dense_index = pc.Index(index_name)
# embeddings = OpenAIEmbeddings(model=MODEL, api_key=API_KEY_OPENAI)
embeddings = PineconeEmbeddings(pinecone_api_key=API_KEY_PINECONE, model=MODEL)

vector_store = PineconeVectorStore(index=dense_index, embedding=embeddings)

# Load and parse out the PDF pages
pdf_loader = PyPDFLoader(file_path="Tiger900SM.pdf", mode="page")
documents = pdf_loader.load()

documents = [Document(page_content=doc.page_content, metadata={"page": doc.metadata["page"] + 1}) for doc in documents]

documentsSubset = documents[:2]
# print(documents[0].metadata)

# Chunk the parsed PDF content
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunked_documents = text_splitter.split_documents(documentsSubset)

# Add chunked documents to the vector store
chunked_ids = [f"chunk_{i}" for i in range(len(documentsSubset))]

# print(f"Adding {len(chunked_documents)} documents to the vector store...")
# Upload/Insert the chunked documents into the vector store in Pinecone
vector_store.add_documents(documents=documentsSubset, ids=chunked_ids)