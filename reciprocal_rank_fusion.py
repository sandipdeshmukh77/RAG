import asyncio
from openai import OpenAI
from dotenv import load_dotenv
import json
from llm_utils import initialize_embeddings,call_llm
from langchain_qdrant import QdrantVectorStore
from typing import List, Dict


async def generate_similar_queries_with_llm(query: str, num_queries: int = 3) -> List[str]:
    try:
        prompt = f"Generate {num_queries} similar queries to: '{query}'. Return only a JSON object with a 'queries' key containing an array of strings."
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(messages,json_format=True)
        
        try:
            data = json.loads(response)
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
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def identify_unique_chunks_ranking(results: List) -> List:
    # Convert each result list to a list of page_content (as document IDs)
    rankings = [[chunk.page_content for chunk in result] for result in results]
    
    # Apply reciprocal rank fusion
    fused_rankings = reciprocal_rank_fusion(rankings)

    
    # Create a mapping of page_content to chunks for easy lookup
    chunk_map = {}
    for result in results:
        for chunk in result:
            chunk_map[chunk.page_content] = chunk
    
    # Get the top chunks based on RRF scores
    unique_chunks = []
    for doc_id, _ in fused_rankings:
        if doc_id in chunk_map:
            unique_chunks.append(chunk_map[doc_id])
    
    # Return top 3 chunks
    return unique_chunks[:3]


async def run_llm(query: str, context: List) -> str:
    try:
        context_text = "\n".join([chunk.page_content for chunk in context])
        messages = [
            {"role": "system", "content": "Use the provided context to answer the question accurately."},
            {"role": "user", "content": f"Question: {query}\nContext: {context_text}"}
        ]
        
        response = call_llm(messages)
        return response
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
        unique_chunks_ranking = identify_unique_chunks_ranking(results)
      

        # Generate final response
        llm_response = await run_llm(original_query, unique_chunks_ranking)
        return llm_response

    except Exception as e:
        print(f"Error in main logic: {e}")
        return "An error occurred while processing your request."

if __name__ == "__main__":
    query = input("Enter your query: ")
    response = asyncio.run(main_logic(query))
    print("\nFinal Response:")
    print(response)
