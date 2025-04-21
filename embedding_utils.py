from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore


def initialize_embeddings():
    load_dotenv()
    embeder = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY")
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
    return vector_store