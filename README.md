# RAG-Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot implemented in Python.

## Prerequisites

- Python 3.7+
- pip
- Virtualenv (optional but recommended)
- Docker (optional, for running with Docker Compose)

## Setup

1.  Create a virtual environment (recommended):

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

2.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Configure the environment variables:

    Create a `.env` file in the project root and set the necessary environment variables.  Refer to `.env.example` or any documentation for the required variables.

## Usage

To run the chatbot:

```bash
python rag-chatbot.py
```

## Docker Compose (Optional)

You can also run the chatbot using Docker Compose.  Make sure you have Docker installed.

```bash
docker-compose up -d
```

## Project Structure

-   `.env`: Environment variables.
-   `rag-chatbot.py`: The main chatbot script.
-   `requirements.txt`:  List of Python dependencies.
-   `docker-compose.db.yml`: Docker Compose configuration file.
-   `.venv`:  Virtual environment directory (if created).
