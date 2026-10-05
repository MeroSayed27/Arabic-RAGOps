# Arabic-RAGOps Architecture

Arabic-RAGOps is a multilingual Arabic/English RAG system.

## Current Core Flow

```text
Documents
   ↓
Ingestion
   ↓
Cleaning
   ↓
Chunking
   ↓
Embeddings
   ↓
Retrieval
   ↓
LLM
   ↓
Answer + Citations
   ↓
Evaluation   