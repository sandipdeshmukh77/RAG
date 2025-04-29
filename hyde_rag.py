import asyncio
from openai import OpenAI
from dotenv import load_dotenv
import json
from llm_utils import initialize_embeddings, call_llm, create_and_index_vector_store
from langchain_qdrant import QdrantVectorStore
from typing import List, Dict


async def generate_hypothetical_document(query: str) -> str:
    try:
        prompt = f"Given the following question, generate a hypothetical document that could contain the answer: {query}\n\nHypothetical Document:"
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(messages)
        return response
    except Exception as e:
        print(f"LLM error during hypothetical document generation: {e}")
        return ""


async def get_embedding_and_search(hypothetical_document: str):

    try:
        embedder = initialize_embeddings()
        retriever = QdrantVectorStore.from_existing_collection(
            embedding=embedder,
            collection_name="test_rag",
            url="http://localhost:6333",
        )

        # Perform semantic search on the hypothetical document embedding
        relevant_chunks = retriever.similarity_search(query=hypothetical_document)

        print(f"Exiting get_embedding_and_search")
        return relevant_chunks
    except Exception as e:
        print(f"Search error for query '{query}': {e}")
        return []



async def run_llm(query: str, context: List[str]) -> str:
    print(f"Entering run_llm with query: {query}")
    try:
        context_text = "\n".join([str(item) for item in context])
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer the question accurately.",
            },
            {"role": "user", "content": f"Question: {query}\nContext: {context_text}"},
        ]

        response = call_llm(messages)
        print(f"Exiting run_llm")
        return response
    except Exception as e:
        print(f"LLM error: {e}")
        return "Sorry, I encountered an error while processing your query."


async def main_logic(original_query: str):
    try:
        # Generate hypothetical document
        hypothetical_document = await generate_hypothetical_document(original_query)
        print(f"Hypothetical Document: {hypothetical_document}")
        # Get relevant chunks based on the hypothetical document
        relevant_chunks = await get_embedding_and_search(hypothetical_document)
        context_chunks = [chunk.page_content for chunk in relevant_chunks]

        # Generate final response using the retrieved context
        final_response = await run_llm(original_query, context_chunks)

        print(f"Exiting main_logic")
        return final_response

    except Exception as e:
        print(f"Error in main logic: {e}")
        return "An error occurred while processing your request."


if __name__ == "__main__":
    query = input("Enter your query: ")
    response = asyncio.run(main_logic(query))
    print("\nFinal Response:")
    print(response)
