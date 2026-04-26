# PowerLLM

A local-first RAG (Retrieval-Augmented Generation) application for legal document analysis.

## Introduction

PowerLLM is designed for analyzing legal documents using retrieval and generation techniques. It uses **LangGraph** for orchestrating the RAG pipeline and supports multiple retrieval modes including similarity search, BM25 keyword search, and hybrid fusion.

The system is built around a **"case" abstraction** where each case contains documents that are processed, embedded, and queried. Documents are ingested through a processing pipeline that handles PDFs with OCR, converts various formats to markdown, and creates searchable vector indices.

This project is the config tuning and benchmark scripts for PowerLLM's RAG pipeline.

## File Structure

```
.
├── src/powerllm/                       # Core source code
│   ├── retrieval/
│   │   ├── rag_graph.py                # LangGraph-based RAG pipeline
│   │   ├── pipeline.py                 # Embedding model registry & retrieval tools
│   │   ├── chunker.py                  # Legal-structure-aware text chunking
│   │   ├── query_builder.py            # LLM-based BM25 query rewriting
│   │   ├── query_normalizer.py         # Query normalization utilities
│   │   └── reranker.py                 # Reranking logic
│   └── models/
│       └── model_resolver.py           # Chat model registry
├── benchmark/                          # Benchmarking & tuning framework
│   ├── benchmark.py                    # Main benchmark evaluation entry point
│   ├── retrieval_config_tuning.py      # Optuna-based hyperparameter optimization
│   ├── generation_latency_benchmark.py # Generation latency testing
│   ├── benchmark_pipeline.py           # Benchmark pipeline utilities
│   ├── benchmark_metrics.py            # Retrieval metrics definitions
│   ├── benchmark_utils.py              # Data processing & I/O utilities
│   ├── dataset_cut3.py                 # LegalBench-RAG dataset sampling
│   ├── build_mix_dataset.py            # Mixed dataset builder
│   ├── gemini_batch.py                 # Gemini Batch API for BM25 query cache
│   ├── README.md                       # Detailed benchmark documentation
│   ├── reproduce_README.md             # Reproduction instructions
│   └── retrieval_config_tuning_README.md  # Optuna tuning guide
├── tests/                              # Unit tests
├── benchmark_results/                  # Benchmark output directory
├── cases_data/                         # Case storage (SQLite + Chroma vector stores)
├── pyproject.toml                      # Project metadata & dependencies
├── requirements.txt                    # Python package requirements
├── config.json                         # Runtime app configuration
└── uv.lock                             # uv lockfile
```

## Basic Usage

### Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Create a virtual environment and install all dependencies from the lockfile:

```bash
uv sync
```

Or install the package in editable mode with pip:

```bash
uv pip install -e .
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
GOOGLE_API_KEY="your_google_api_key"              # Required for Gemini models
LANGSMITH_API_KEY="..."                           # Optional, for LangSmith tracing
```

## Detailed Documentation

For comprehensive documentation on specific topics, see the dedicated READMEs in the `benchmark/` directory:

- **[benchmark/README.md](benchmark/README.md)** — Full benchmark documentation, including architecture, metrics, dataset sampling, and caching strategy.
- **[benchmark/reproduce_README.md](benchmark/reproduce_README.md)** — Step-by-step instructions for reproducing benchmark results, including dataset preparation and tuning/benchmark commands.
- **[benchmark/retrieval_config_tuning_README.md](benchmark/retrieval_config_tuning_README.md)** — Guide to Optuna-based retrieval hyperparameter tuning, including search-space presets, budget pruning, and result inspection.

### Files
manual.pdf
reproduce.pdf
report.pdf
