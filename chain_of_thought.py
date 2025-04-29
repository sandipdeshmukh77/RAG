import asyncio
from openai import OpenAI
from dotenv import load_dotenv
import json
from llm_utils import initialize_embeddings, call_llm, create_and_index_vector_store
from langchain_qdrant import QdrantVectorStore
from typing import List, Dict


async def generate_sub_queries_with_llm(query: str, num_queries: int = 3) -> List[str]:
    try:
        prompt = f"Generate {num_queries} sub queries to: '{query}'. Return only a JSON object with a 'queries' key containing an array of strings.example of subquery -- original query: what is fs module in node js? subquery:[what is fs module,what is node js]? "
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(messages, json_format=True)

        try:
            data = json.loads(response)
            queries = data.get('queries', [])
            print(f"Exiting generate_sub_queries_with_llm with queries: {queries}")
            return queries
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return []


async def get_embedding_and_search(query: str):
    print(f"Entering get_embedding_and_search with query: {query}")
    try:
        embedder = initialize_embeddings()
        retriever = QdrantVectorStore.from_existing_collection(
            embedding=embedder,
            collection_name="test_rag",
            url="http://localhost:6333",
        )
        relevant_chunks = retriever.similarity_search(query=query)
        print(f"Exiting get_embedding_and_search")
        return relevant_chunks
    except Exception as e:
        print(f"Search error for query '{query}': {e}")
        return []


def identify_unique_chunks(results: List) -> List:
    print(f"Entering identify_unique_chunks")
    seen = set()
    unique_chunks = []

    for chunk_list in results:
        for chunk in chunk_list:
            # Use the chunk's page_content as the unique identifier
            if chunk.page_content not in seen:
                seen.add(chunk.page_content)
                unique_chunks.append(chunk)

    print(f"Exiting identify_unique_chunks")
    return unique_chunks


async def run_llm(query: str, context: List[str], previous_response: str = None) -> str:
    print(f"Entering run_llm with query: {query}")
    try:
        context_text = "\n".join([str(item) for item in context]) #Converting to string to avoid type errors
        if previous_response:
            context_text = f"{context_text}\nPrevious Response: {previous_response}"

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
    print(f"Entering main_logic with original query: {original_query}")
    try:
        similar_queries = await generate_sub_queries_with_llm(original_query)
        all_queries = [original_query] + similar_queries
        print(f"Processing queries: {all_queries}")

        overall_context = []
        previous_response = None
        for query in all_queries:
            # embedding and semantic search
            relevant_chunks = await get_embedding_and_search(query)
            context_chunks = [chunk.page_content for chunk in relevant_chunks]

            # Generate response with accumulated context
            llm_response = await run_llm(query, context_chunks, previous_response)
            overall_context.append(llm_response)
            previous_response = llm_response  # Store current response for next iteration

        # Final answer using all accumulated responses as context
        final_context = "\n".join(overall_context)
        final_response = await run_llm(original_query, [final_context])

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
