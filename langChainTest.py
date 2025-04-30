
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeEmbeddings

load_dotenv()

API_KEY_PINECONE = os.getenv('API_KEY_PINECONE')
API_KEY_OPENAI = os.getenv('API_KEY_OPENAI')

MODEL = "multilingual-e5-large"
index_name = "tiger-900-sm"
# query = "Who is the manual designed for?"
query = "What page says who the manual is designed for?"

openai = OpenAI(api_key=API_KEY_OPENAI)
pc = Pinecone(api_key=API_KEY_PINECONE)

dense_index = pc.Index(index_name)
# embeddings = OpenAIEmbeddings(model=MODEL, api_key=API_KEY_OPENAI)
embeddings = PineconeEmbeddings(pinecone_api_key=API_KEY_PINECONE, model=MODEL)
# print(embeddings)
vector_store = PineconeVectorStore(index=dense_index, embedding=embeddings)

results = vector_store.similarity_search(
    query,
    k=2,
    # filter={"source": "tweet"},
)

# print(results[0].page_content)

print(results[0].metadata["page"])

# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")

def generate_answer_with_context(query, matchingDocuments):

    context = ""

    for document in matchingDocuments:
        context = context + "text: " + document.page_content + "Page number: " + str(document.metadata["page"]) + "\n"
        context = context + f"Page number: {document.metadata["page"]} \n"


    prompt = f"Context: {context}\n\nUser Query: {query}\nAnswer:"
    response = openai.responses.create(
        model="gpt-3.5-turbo",
        input=prompt,
    )

    return response


response = generate_answer_with_context(query, results)

print(response.output_text)