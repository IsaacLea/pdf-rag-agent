from pinecone import Pinecone
from openai import OpenAI
import os

API_KEY_PINECONE = os.getenv('API_KEY_PINECONE')
API_KEY_OPENAI = os.getenv('API_KEY_OPENAI')
MODEL = "text-embedding-3-small"

openai = OpenAI(api_key=API_KEY_OPENAI)
pc = Pinecone(
        # api_key=os.environ.get("PINECONE_API_KEY")
        api_key=API_KEY_PINECONE
    )

# Define the query
query = "Where did apples originate from?"

index_name = "pdf-rag1"
dense_index = pc.Index(index_name)

# Convert the query into a numerical vector that Pinecone can search with
query_embedding = pc.inference.embed(
    model="llama-text-embed-v2",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)



def generate_answer_with_context(query, contextMatches):

    context = ""

    for match in contextMatches:
        context = context + "text: " + match.metadata["chunk_text"] + "Category: " + match.metadata["category"] + "\n"
        context = context + match.metadata["chunk_text"] + "\n"
        # print(match.metadata["chunk_text"])
       

        contextMatches = match.metadata["chunk_text"]


    print(context);

    # context = "\n".join(contextMatches)
    prompt = f"Context: {context}\n\nUser Query: {query}\nAnswer:"
    response = openai.responses.create(
        model="gpt-3.5-turbo",
        input=prompt,
    )

    # return response.choices[0].text.strip()
    return response

# Search the index for the three most similar vectors
results = dense_index.query(
        namespace="example-namespace",
        vector=query_embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )

# print(results.matches)

res = generate_answer_with_context(query, results.matches)

print(res.output_text)
# Search the dense index
# results = dense_index.search(
#     namespace="example-namespace",
#     query={
#         "top_k": 10,
#         "inputs": {
#             'text': query
#         }
#     }
# )

# # Print the results
# for hit in results['result']['hits']:
#         print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}")