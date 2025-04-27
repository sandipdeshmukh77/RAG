from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import OpenAI


LLM = "openai"  # or "google"
load_dotenv()
client = None
if LLM == "openai":
    # Initialize OpenAI client with the specified API key and base URL
    client = OpenAI()
else:
    client = OpenAI(
        api_key=os.getenv('GOOGLE_API_KEY'),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def initialize_embeddings():
    if LLM == "openai":
        # Initialize OpenAI embeddings with the specified model
        embeder = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        # Initialize Google Generative AI embeddings with the specified model
        embeder = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
        )
    return embeder

def create_and_index_vector_store(documents, embedding, collection_name, url):
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name=collection_name,
        url=url,
    )
    vector_store.add_documents(documents=documents)
    print("indexing done")
    # return vector_store



def call_llm(query,json_format=False):
    if json_format:
        response = client.chat.completions.create(
            model="gpt-4o" if LLM == "openai" else "gemini-2.0-flash",
            messages=query,
            response_format={"type": "json_object"}
        )
    else:
         # If not JSON format, just use the default response
        response = client.chat.completions.create(
                model="gpt-4o" if LLM == "openai" else "gemini-2.0-flash",
                messages=query
            )
    return response.choices[0].message.content