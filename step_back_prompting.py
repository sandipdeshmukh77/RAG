
# Import required libraries
import json
from langchain_community.document_loaders import PyPDFLoader  # For loading PDF files
from pathlib import Path  # For handling file paths in a platform-independent way
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain_openai import OpenAIEmbeddings  # For creating text embeddings
from langchain_qdrant import QdrantVectorStore  # For vector storage and retrieval
from dotenv import load_dotenv  # For loading environment variables
from openai import OpenAI  # For interacting with OpenAI API
import os  
from llm_utils import initialize_embeddings, call_llm # Custom utility for embedding initialization


# Initialize OpenAI embeddings with the specified model
embeder = initialize_embeddings()
retriver = QdrantVectorStore.from_existing_collection(
        embedding=embeder,
        collection_name="test_rag",
        url="http://localhost:6333",
    )



def generate_step_back_query(question: str) -> str:
    """
    Generate a step back query based on the provided question.
    """
    # Define the system prompt for generating the step back query

    # Initialize messages with the system prompt
    step_back_query_system_prompt = """
        You are a helpful assistant your task is generating *step back query* for the given question.

        A step back query is a broader or more general version of the original question. It should still be related to the original topic but allow for retrieving a wider range of relevant information.  
        The step back query must be phrased as a question.

        Guidelines:
        - Keep the core subject of the question intact.
        - Broaden the scope slightly by asking about uses, applications, history, benefits, challenges, or overview.
        - Rephrase the question if necessary to sound more general.

        Examples:

        Original question: What is Node.js?  
        Step back question: What is Node.js used for?

        Original question: What is Node.js and its uses?  
        Step back question: What are the applications of Node.js?

        Original question: How does React work?  
        Step back question: What are the key principles behind React?

        Original question: What is the best database for e-commerce?  
        Step back question: What are different database options for e-commerce applications?

        Original question: How can I optimize a Python script?  
        Step back question: What are best practices for optimizing Python programs?

        Original question: What are the benefits of using Kubernetes?  
        Step back question: Why is Kubernetes widely adopted for container orchestration?

        Original question: What is the future of AI in healthcare?  
        Step back question: How is AI transforming the healthcare industry?

        Original question: What are common issues in microservices architecture?  
        Step back question: What challenges do organizations face when using microservices architecture?

        Make sure the step back query stays natural, helpful, and inquisitive.
        **Output Format:**
        - Your response must be in **JSON** format.
        - JSON structure:
        {
        "original_question": "<original question>",
        "step_back_question": "<generated step back question>"
        }
        """
    messages = [
        {"role": "system", "content": step_back_query_system_prompt},
        {"role": "user", "content": question}
    ]

    # Call the LLM to generate the step back query
    response = call_llm(messages,json_format=True)

    response_json = json.loads(response)  
    # Return the generated step back query
    step_back_query = response_json.get("step_back_question")
    return step_back_query

# Initialize an empty list to store conversation history
conversation_history = []

while True:
    # Get user input for the question
    query = input("Enter your question (or 'exit' to end): ")
    
    # Check if user wants to exit
    if query.lower() == 'exit':
        break

    step_back_query = generate_step_back_query(query)

    # Search for relevant text chunks based on the step back query
    relevant_chunks = retriver.similarity_search(query=step_back_query)

    # Define the system prompt for the chatbot
    system_prompt = f"""
    You are a helpful assistant. You help the user to find the answer to their question based on the provided context.
    context: {relevant_chunks}
    You will be provided with a context and a question. You need to answer the question based on the context.
    If the context does not provide enough information to answer the question, you should say "I don't know".
    Note:
    Answer should be in detaild and should not be too short.
    Answer should be in a conversational tone.
    """

    # Initialize the messages list with the system prompt
    messages = [
        {"role": "system", "content": system_prompt},
    ]


    # Add conversation history to messages
    for msg in conversation_history:
        messages.append(msg)

    # Add current user query to messages
    messages.append({"role": "user", "content": query})

    # Generate response using OpenAI's chat completion
    response = call_llm(messages)

    # Store the interaction in conversation history
    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": response})

    # Keep conversation history limited to last 4 interactions (8 messages)
    if len(conversation_history) > 8:
        conversation_history = conversation_history[-8:]

    # Print the generated response
    print("\nAssistant:", response, "\n")