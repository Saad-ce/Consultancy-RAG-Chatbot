# Project Name
Consultancy RAG Chatbot

> This is an older project I built during my Masters. Uploading to document my progress.


## Description
This project is an AI-powered Retrieval-Augmented Generation (RAG) system designed to act as a Senior Consultant. It extracts knowledge from internal company documents (PDF, DOCX, PPTX) and uses a vector database to provide context-aware answers to client inquiries. The system features a multi-turn conversation logic that remembers chat history and uses a "follow-up" mode to maintain context during long discussions.

## Technologies Used
Python: Core programming language.

FAISS: Vector database for high-speed similarity search.

Sentence-Transformers: To generate semantic text embeddings (all-MiniLM-L6-v2).

LangChain: For text splitting and document chunking.

Unstructured / python-docx / python-pptx: For parsing complex document formats.

Mistral API: The Large Language Model (LLM) used for generating responses.