# Arabic-RAGOps

### Multilingual AI Knowledge & Evaluation Assistant

Arabic-RAGOps is a Python portfolio project exploring multilingual Retrieval-Augmented Generation (RAG), with a current focus on document processing, Arabic-English retrieval, hybrid search, reranking, and retrieval evaluation.

The project investigates how different retrieval methods find relevant information in a small knowledge base and how their results can be evaluated systematically.

## Project Goals

- Support Arabic, English, and mixed-language queries.
- Process documents into searchable text chunks.
- Generate multilingual text embeddings.
- Compare semantic retrieval with keyword-based retrieval.
- Combine dense and sparse retrieval methods.
- Experiment with cross-encoder reranking.
- Evaluate retrieval quality using measurable metrics.
- Build a foundation for a more complete, grounded RAG application.

## Current Features

- Document ingestion
- Text chunking with overlap
- Multilingual sentence embeddings
- Dense semantic retrieval
- BM25 keyword retrieval
- Hybrid retrieval
- Cross-encoder reranking
- Retrieval evaluation metrics
- A small JSON evaluation dataset
- Automated tests using pytest
- Saved experimental baseline results

## Technology Stack

- Python
- Sentence Transformers
- Hugging Face
- `rank-bm25`
- Cross-encoder reranking
- pytest
- JSON evaluation data
- Git and GitHub

## Architecture

```text
Knowledge Documents
        |
        v
Document Ingestion
        |
        v
Text Chunking
        |
        v
Multilingual Embeddings
        |
        +----------------------+
        |                      |
        v                      v
  Dense Retrieval         BM25 Retrieval
        |                      |
        +-----------+----------+
                    |
                    v
             Hybrid Retrieval
                    |
                    v
             Cross-Encoder
                Reranking
                    |
                    v
            Retrieval Evaluation
```

**Current scope:** The implemented prototype focuses on the retrieval and evaluation stages. Grounded answer generation, a production API, and a user interface are not yet complete.

## Evaluation Results

The current evaluation dataset contains six questions. The following are preliminary results from the existing experiment.

| Retrieval Method | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| Dense | 0.6667 | 0.7500 | 0.9167 |
| BM25 | 0.4167 | 0.6667 | 0.6667 |
| Hybrid | 0.6667 | 1.0000 | 0.9167 |
| Reranked Retrieval | 0.2500 | 1.0000 | 0.7500 |

### Understanding the Metrics

- **Recall@1:** The proportion of expected relevant chunks found in the first result.
- **Recall@3:** The proportion of expected relevant chunks found within the first three results.
- **MRR (Mean Reciprocal Rank):** Measures how high the first relevant result appears. Higher values indicate that relevant results tend to appear earlier.

### Initial Findings

In this small test dataset, hybrid retrieval found all expected relevant chunks within the top three results. However, the reranked results had lower Recall@1, indicating that the ranking stage needs further investigation.

These results are experimental, not a general performance guarantee. A larger and more diverse evaluation dataset is needed before drawing broader conclusions.

## Project Structure

```text
Arabic-RAGOps/
├── app/
│   ├── config/
│   ├── ingestion/
│   ├── chunking/
│   ├── embeddings/
│   └── retrieval/
├── data/
│   └── sample/
├── evaluation/
│   ├── questions.json
│   ├── retrieval_metrics.py
│   ├── compare_retrieval_methods.py
│   └── results/
├── tests/
├── docs/
├── notebooks/
├── scripts/
├── .env.example
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Installation

These instructions are for Windows Command Prompt.

### 1. Open the project folder

Open a terminal in the `Arabic-RAGOps` project root.

### 2. Create a virtual environment

```cmd
python -m venv .venv
```

### 3. Activate the virtual environment

```cmd
.venv\Scripts\activate
```

### 4. Install the dependencies

```cmd
python -m pip install -r requirements.txt
```

Some pretrained models may download the first time they are used. An internet connection may be required for these downloads.

## Running the Project

Run the semantic retrieval demo:

```cmd
python -m app.retrieval.retrieval_demo
```

Run the BM25 retrieval test:

```cmd
python -m app.retrieval.test_bm25_index
```

Compare dense, BM25, and hybrid retrieval:

```cmd
python -m evaluation.compare_retrieval_methods
```

Run the retrieval metrics evaluation:

```cmd
python -m evaluation.test_retrieval_metrics
```

Run the reranked retrieval evaluation:

```cmd
python -m evaluation.test_reranked_metrics
```

## Running the Automated Tests

From the project root, run:

```cmd
python -m pytest -q
```

The existing automated test suite has passed in the development environment. Run the command again after changes to verify the current state.

## Limitations

- The sample knowledge base is small.
- The evaluation dataset currently contains only six questions.
- Dense retrieval uses an in-memory vector index.
- Results may change with different documents, queries, or model settings.
- Hybrid retrieval does not outperform dense retrieval on every metric.
- Reranking needs further improvement.
- Grounded answer generation and production deployment are not yet implemented.

## Future Development

Planned areas for improvement include:

- Expanding the Arabic-English evaluation dataset
- Improving and tuning reranking
- Testing retrieval across different document types
- Adding grounded answer generation with source citations
- Evaluating answer relevance and faithfulness
- Exploring persistent vector storage and an API
- Improving reproducibility and documenting experiments

## Responsible Evaluation

This project reports experimental results transparently and documents the limitations of its evaluation. Its small test dataset is intended for initial development and does not establish production-level accuracy or reliability.

## License

See the `LICENSE` file for the terms governing the use and distribution of this project.