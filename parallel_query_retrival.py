import asyncio
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from embedding_utils import initialize_embeddings
from langchain_qdrant import QdrantVectorStore
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

async def generate_similar_queries_with_llm(query: str, num_queries: int = 3) -> List[str]:
    try:
        prompt = f"Generate {num_queries} similar queries to: '{query}'. Return only a JSON object with a 'queries' key containing an array of strings."
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=messages,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        
        try:
            data = json.loads(content)
            return data.get('queries', [])
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return []

async def get_embedding_and_search(query: str):
    try:
        embedder = initialize_embeddings()
        retriever = QdrantVectorStore.from_existing_collection(
            embedding=embedder,
            collection_name="test_rag",
            url="http://localhost:6333",
        )
        relevant_chunks = retriever.similarity_search(query=query)
        print(f"Search completed for: {query}")
        return relevant_chunks
    except Exception as e:
        print(f"Search error for query '{query}': {e}")
        return []

def identify_unique_chunks(results: List) -> List:
    seen = set()
    unique_chunks = []
    
    for chunk_list in results:
        for chunk in chunk_list:
            # Use the chunk's page_content as the unique identifier
            if chunk.page_content not in seen:
                seen.add(chunk.page_content)
                unique_chunks.append(chunk)
    
    return unique_chunks

async def run_llm(query: str, context: List) -> str:
    try:
        context_text = "\n".join([chunk.page_content for chunk in context])
        messages = [
            {"role": "system", "content": "Use the provided context to answer the question accurately."},
            {"role": "user", "content": f"Question: {query}\nContext: {context_text}"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Fixed model name
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM error: {e}")
        return "Sorry, I encountered an error while processing your query."

async def main_logic(original_query: str):
    try:
        similar_queries = await generate_similar_queries_with_llm(original_query)
        all_queries = [original_query] + similar_queries
        print(f"Processing queries: {all_queries}")

        # Parallel embedding and semantic search
        tasks = [get_embedding_and_search(query) for query in all_queries]
        results = await asyncio.gather(*tasks)

        # Identify unique response chunks
        unique_chunks = identify_unique_chunks(results)
        print(f"Found {len(unique_chunks)} unique chunks")

        # Generate final response
        llm_response = await run_llm(original_query, unique_chunks)
        return llm_response

    except Exception as e:
        print(f"Error in main logic: {e}")
        return "An error occurred while processing your request."

if __name__ == "__main__":
    query = input("Enter your query: ")
    response = asyncio.run(main_logic(query))
    print("\nFinal Response:")
    print(response)
