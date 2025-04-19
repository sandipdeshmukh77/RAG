
# RAG Chatbot Project
To clone the repository, use the following command:
```bash
git clone https://github.com/sandipdeshmukh77/simple-rag-chatbot
```
This project implements a Retrieval-Augmented Generation (RAG) chatbot.
## Files Description
- **.env**: Contains environment variables, such as the OpenAI API key.
- **.gitignore**: Specifies intentionally untracked files that Git should ignore.
- **docker-compose.db.yml**: Configuration file for running Qdrant vector database using Docker Compose.
- **nodejs.pdf**: A PDF document that is used as the knowledge base for the chatbot.
- **rag-chatbot.py**: The main Python script that implements the RAG chatbot.
- **requirement.txt**: Lists the Python packages required to run the project.
## Overview
The `rag-chatbot.py` script performs the following steps:
1.  Loads the PDF document (`nodejs.pdf`).
2.  Splits the document into smaller chunks for better processing.
3.  Creates embeddings for each chunk using the OpenAI API.
4.  Stores the embeddings in a Qdrant vector database.
5.  Retrieves relevant chunks from the database based on a user's query.
6.  Generates a response using OpenAI's chat completion, incorporating the retrieved information.
## Requirements
- Python 3.6+
- pip
## Dependencies
The project dependencies are listed in `requirement.txt`. To install them, run:
```bash
pip install -r requirements.txt
```
## Environment Variables
Create a `.env` file in the project root directory and add your OpenAI API key:
```
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```
## Running the Chatbot
1.  Ensure Qdrant is running, by using the docker-compose.db.yml file `docker-compose -f docker-compose.db.yml up -d`
2.  Run the `rag-chatbot.py` script:
```bash
python rag-chatbot.py
```
