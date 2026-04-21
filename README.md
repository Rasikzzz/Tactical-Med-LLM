# TCCC-RAG: Tactical Combat Casualty Care Knowledge Engine

A high-precision **Retrieval-Augmented Generation (RAG)** pipeline designed to ingest and query the **ATP 4-02.43** (Army Techniques Publication for Casualty Response, Tactical Combat Casualty Care, and First Aid).

## 🚀 Overview

This repository contains a technical implementation of a RAG system that transforms static military medical doctrine into a dynamic, queryable interface. By leveraging **LangChain** for orchestration, **ChromaDB** for vector indexing, and **Ollama** for local LLM inference, the system provides grounded responses for tactical medical protocols without external API dependencies.

## 🛠️ Technical Stack

* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Inference Engine:** [Ollama](https://ollama.com/) (Supports Llama 3, Mistral, etc.)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Document Loader:** PyPDFLoader
* **Text Splitting:** RecursiveCharacterTextSplitter

## 🏗️ Implementation Architecture

The pipeline follows a modular five-stage process:

1.  **Ingestion:** Extracts text from the ATP 4-02.43 PDF.
2.  **Semantic Chunking:** Splits documents into 1000-character segments with a 150-character overlap to preserve the continuity of multi-step medical procedures.
3.  **Vectorization:** Generates high-dimensional embeddings via Ollama.
4.  **Storage:** Indexes vectors in a local ChromaDB instance for persistent, fast retrieval.
5.  **Query Logic:** Uses a `RetrievalQA` chain to inject relevant document context into the LLM prompt, ensuring responses are doctrinally accurate.

## 📋 Features

* **Zero-Cloud Dependency:** Runs entirely on local hardware using Ollama.
* **Persistent Index:** ChromaDB stores the tokenized manual locally, so indexing is only required once.
* **Doctrinal Fidelity:** Uses custom prompt templates to force the model to stick strictly to the provided ATP context.

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* [Ollama](https://ollama.com/) installed and running
* The ATP 4-02.43 PDF file

### Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/TCCC-RAG.git](https://github.com/your-username/TCCC-RAG.git)
cd TCCC-RAG

# Install dependencies
pip install langchain langchain-community chromadb pypdf
