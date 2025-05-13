import os
import re
import pdfplumber
import openai
import pinecone
from pinecone import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec

API_KEY_PINECONE = os.getenv('API_KEY_PINECONE')
API_KEY_OPENAI = os.getenv('API_KEY_OPENAI')

pc = Pinecone(
        api_key=API_KEY_PINECONE
    )

# Create a dense index with integrated embedding
index_name = "pdf-rag2"
# if not pc.has_index(index_name):
#     pc.create_index_for_model(
#         name=index_name,
#         cloud="aws",
#         region="us-east-1",
#         embed={
#             "model":"llama-text-embed-v2",
#             "field_map":{"text": "chunk_text"}
#         }
#     )

# spec = ServerlessSpec(cloud="aws", region="us-east-1")

# if not pc.has_index(index_name):
#     pc.create_index(
#         index_name,
#         dimension=1536,  # dimensionality of text-embed-3-small
#         metric='dotproduct',
#         spec=spec
#     )

# Define a sample dataset where each item has a unique ID and piece of text
# data = [
#     {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
#     {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
#     {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
#     {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
#     {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
#     {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
# ]

# Convert the text into numerical vectors that Pinecone can index
# embeddings = pc.inference.embed(
#     model="llama-text-embed-v2",
#     inputs=[d['text'] for d in data],
#     parameters={"input_type": "passage", "truncate": "END"}
# )

# print(embeddings)

# Target the index
dense_index = pc.Index(index_name)

# Upsert the records into a namespace
dense_index.upsert_records(
    "example-namespace",
    [
        {
            "_id": "rec1",
            "chunk_text": "Apples are a great source of dietary fiber, which supports digestion and helps maintain a healthy gut.",
            "category": "digestive system", 
        },
        {
            "_id": "rec2",
            "chunk_text": "Apples originated in Central Asia and have been cultivated for thousands of years, with over 7,500 varieties available today.",
            "category": "cultivation",
        },
        {
            "_id": "rec3",
            "chunk_text": "Rich in vitamin C and other antioxidants, apples contribute to immune health and may reduce the risk of chronic diseases.",
            "category": "immune system",
        },
        {
            "_id": "rec4",
            "chunk_text": "The high fiber content in apples can also help regulate blood sugar levels, making them a favorable snack for people with diabetes.",
            "category": "endocrine system",
        },
    ]
) 