# Import required libraries
from langchain_community.document_loaders import PyPDFLoader  # For loading PDF files
from pathlib import Path  # For handling file paths in a platform-independent way
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain_openai import OpenAIEmbeddings  # For creating text embeddings
from langchain_qdrant import QdrantVectorStore  # For vector storage and retrieval
from dotenv import load_dotenv  # For loading environment variables
from openai import OpenAI  # For interacting with OpenAI API
import os  
from llm_utils import initialize_embeddings, create_and_index_vector_store ,call_llm # Custom utility for embedding initialization

# Set up the path to the PDF file in the same directory as the script
file_path = Path(__file__).parent /"nodejs.pdf"

# Initialize PDF loader with the file path
loader = PyPDFLoader(file_path)

# Load and parse the PDF document
docs = loader.load()

# Configure text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# Split the documents into smaller chunks
spit_docs = text_splitter.split_documents(documents = docs)

# Initialize OpenAI embeddings with the specified model
embeder = initialize_embeddings()

# The following code block is commented out as it's used for initial setup
# It creates a new vector store and adds documents to it
#-------------------start----------------------
# create_and_index_vector_store(
#     documents=spit_docs,
#     embedding=embeder,
#     collection_name="test_rag_with_gemini",
#     url="http://localhost:6333"
# )
#------------------------end-------------------
# print("indexing done")
# Connect to existing Qdrant collection
retriver = QdrantVectorStore.from_existing_collection(
        embedding=embeder,
        collection_name="test_rag",
        url="http://localhost:6333",
    )

# Initialize an empty list to store conversation history
conversation_history = []

while True:
    # Get user input for the question
    query = input("Enter your question (or 'exit' to end): ")
    
    # Check if user wants to exit
    if query.lower() == 'exit':
        break

    # Search for relevant text chunks based on the query
    relevant_chunks = retriver.similarity_search(query=query)

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