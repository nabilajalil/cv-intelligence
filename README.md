# CV Intelligence

**Evidence-grounded candidate search and analysis using Agentic RAG, hybrid retrieval, and multimodal AI.**

CV Intelligence is a Python-based AI engineering portfolio project exploring natural-language CV search, candidate evaluation, retrieval optimization, and reliable LLM response generation. It identifies explicitly documented skills and returns supporting evidence from source documents.

## Key Capabilities

| Capability | Implementation | Status |
|---|---|---|
| Document ingestion | PDF/DOCX extraction, section-aware chunking and candidate metadata | Implemented |
| Hybrid RAG | OpenAI embeddings, Chroma, BM25 and Reciprocal Rank Fusion (RRF) | Implemented |
| Candidate evaluation | Skill matching, candidate ranking and supporting evidence | Implemented |
| Agentic orchestration | Controlled sequential LangGraph workflow | Implemented |
| Response generation | LangChain, Ollama and Qwen2.5 3B/7B | Implemented; generative mode experimental |
| Inference controls | Structured outputs, estimated token budgets, temperature, HTTP timeouts and fallback | Implemented |
| Retrieval optimization | Baseline BM25, contextual retrieval and late chunking comparisons | Experimental |
| Long-context vs. RAG | Context and computational cost comparison | Experimental |
| GraphRAG | Candidate knowledge graph and graph-enhanced retrieval | Experimental |
| Multimodal retrieval | Text/vision processing and protected scanned-PDF search | Implemented prototype |
| Evidence security | Claim validation, source binding, SHA-256 integrity and abstention | Implemented in protected workflows |
| Privacy and security | Access control, data retention policies and sensitive information protection | Implemented |
| Prompt engineering | Versioned prompts, structured claims and known-pattern injection detection | Experimental |
| Public dataset benchmarking | Hugging Face dataset and automated retrieval evaluation | Implemented |
| Observability | LangSmith tracing, latency, token monitoring and fallback diagnostics | Implemented; partial instrumentation |
| Application | FastAPI, Streamlit and Docker configuration | Implemented |
| Automated testing | 164 passing tests | Implemented |

## Technology Stack

| Component | Technologies |
|---|---|
| Language | Python 3.12 |
| LLM framework and orchestration | LangChain, LangGraph |
| Models | Qwen2.5 3B/7B, OpenAI GPT-4.1 mini |
| Embeddings | OpenAI text-embedding-3-small |
| Local inference | Ollama |
| Retrieval | Chroma, BM25, RRF |
| Document processing | PyMuPDF, PDF/DOCX processing |
| Data validation | Pydantic |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Observability | LangSmith |
| Testing | Pytest, unittest |
| Containerization | Docker |

## Agentic RAG Workflow

1. **Understand:** Extract explicit skill requirements from the recruiter query.
2. **Retrieve:** Retrieve relevant CV chunks using hybrid lexical and semantic search.
3. **Evaluate:** Match documented skills, rank candidates and collect supporting evidence.
4. **Respond:** Produce a recruiter summary from structured candidate evidence.

The main application uses a fixed LangGraph workflow. Skill extraction, ranking and evidence selection are controlled by application logic rather than an LLM. LLM inference is optional and limited to response generation in this workflow.

## Retrieval Research and Experiments

| Strategy | Objective |
|---|---|
| Baseline BM25 | Establish a lexical retrieval baseline |
| Dense retrieval | Capture semantic similarity beyond exact keywords |
| Hybrid retrieval | Combine lexical precision and semantic relevance |
| Contextual retrieval | Reduce context loss in independently retrieved chunks |
| Late chunking | Explore preservation of document-level context |
| Long-context vs. RAG | Compare context utilization and estimated inference costs |
| GraphRAG | Explore relationships between candidates, skills, projects and experience |

Hybrid retrieval is used in the main application. Advanced retrieval strategies are maintained as separate experiments to investigate relevance, contextual preservation and computational trade-offs.

## Datasets

The public [Resume Parsing Vision](https://huggingface.co/datasets/sukhrobnurali/resume-parsing-vision) dataset from Hugging Face provides synthetic resumes and structured ground-truth annotations for retrieval benchmarking. The download pipeline requests 50 training samples by default, with a configurable limit and validation of records.

Additional synthetic CVs support candidate evaluation, GraphRAG, multimodal verification, prompt injection testing and LLM response evaluation.

## LLM Response Generation

| Mode | Description |
|---|---|
| Deterministic (default) | Python-generated summaries without LLM inference |
| Constrained | Qwen2.5 3B selects a response style; Python renders candidate facts |
| Generative (experimental) | Qwen2.5 7B generates recruiter summaries from structured candidate evidence |

Inference controls include structured output validation, estimated token budgets, temperature settings, HTTP timeouts, prompt versioning and deterministic fallback. Generated prose in experimental mode is not fully semantically validated.

## Multimodal Retrieval and Evidence Security

The protected scanned-CV workflow combines text extraction and visual verification, with candidate and source binding, claim-level validation, source-page provenance, SHA-256 document integrity checks and abstention when evidence is unavailable or invalid. Protected search uses previously prepared verification artifacts.

A separate prompt engineering experiment investigates unsupported skill claims and known-pattern prompt injection detection.

## Observability

- LangSmith tracing for controlled synthetic evaluations.
- Response generation latency and estimated token-budget monitoring.
- Actual token usage where supported, prompt versioning and fallback diagnostics.
- Streamlit inference diagnostics and API cost estimation.

## Evaluation and Testing

| Evaluation | Result |
|---|---|
| Automated test suite | 164 passed |
| GraphRAG — BM25 + graph | nDCG@5: 0.890 |
| Protected visual evidence verification | 6/6 supported |
| Protected multimodal search | 3/3 query scenarios passed |
| Document integrity scenarios | 2/2 passed |
| Grounded-answer regression | 27/27 passed |
| HTTP and grounded-answer tests | 39/39 passed |
| Constrained LLM response evaluation | 4/4 passed |

These results reflect controlled experimental conditions, not production-scale performance or generalization to real recruitment datasets.

## Installation

Requires Python 3.12, Git, an OpenAI API key for OpenAI-dependent workflows, and Ollama for optional local LLM inference.

```bash
git clone https://github.com/nabilajalil/cv-intelligence.git
cd cv-intelligence
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[agent]"
cp .env.example .env
```

Configure the required environment variables in `.env`. Do not commit API keys or sensitive CV data.

For optional local response generation:

```bash
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5:7b
```

## Running the Application

Start the FastAPI backend:

```bash
python -m uvicorn cv_intelligence.api:app --host 127.0.0.1 --port 8001
```

In a separate terminal, start Streamlit:

```bash
source .venv/bin/activate
CV_INTELLIGENCE_API_URL=http://127.0.0.1:8001 python -m streamlit run cv_intelligence/dashboard.py
```

- API documentation: http://127.0.0.1:8001/docs
- Streamlit dashboard: use the local URL printed by Streamlit.

To enable optional LLM generation, stop the backend and restart it with:

```bash
LANGSMITH_TRACING=false \
CV_ENABLE_LLM_POLISHING=true \
CV_RESPONSE_MODE=generative \
python -m uvicorn cv_intelligence.api:app --host 127.0.0.1 --port 8001
```

Use `CV_RESPONSE_MODE=constrained` for constrained response generation.

### Prepare the Public Benchmark

```bash
python -m scripts.data_preparation.download_public_cvs --limit 50
python -m scripts.data_preparation.inspect_public_cvs
python -m scripts.data_preparation.chunk_public_cvs
python -m scripts.data_preparation.build_public_benchmark
```

### Run Tests

```bash
python -m pytest -q
```

Some search, evaluation and multimodal workflows require locally prepared datasets, indexes and verification artifacts.

## Project Structure

```text
cv-intelligence/
├── cv_intelligence/          # Core application package
│   ├── api.py                # Main FastAPI application
│   ├── protected_api.py      # Protected multimodal endpoint
│   ├── dashboard.py          # Streamlit interface
│   ├── prompts/              # LLM prompts, response chains and validation
│   └── ...                   # Other application modules
├── scripts/                  # Data preparation and experimental workflows
│   ├── agentic_rag/          # Agentic RAG and response evaluation
│   ├── data_preparation/     # Dataset download, ingestion and chunking
│   ├── evaluation/           # Retrieval benchmarks and evaluation
│   ├── graphrag/             # Knowledge graphs and GraphRAG experiments
│   ├── llm/                  # Prompt engineering and claim validation
│   ├── multimodal/           # Visual evidence and document verification
│   ├── retrieval/            # Search and retrieval experiments
│   └── utilities/            # Supporting utilities
├── tests/                    # Automated tests and regression checks
├── data/                     # Local datasets and generated artifacts
├── pyproject.toml            # Project metadata and dependencies
├── Dockerfile                # Container build configuration
├── compose.yaml              # Container service configuration
└── README.md                 # Project documentation
```

## Roadmap

- **Incremental indexing:** Efficient BM25 and vector index updates for new, modified and deleted CVs.
- **Retrieval scalability:** Persistent lexical indexing, caching, index synchronization and large-scale performance benchmarking.
- **Persistent agent memory:** Session-aware conversations, LangGraph checkpointing and privacy-aware long-term memory.
- **Retrieval optimization:** Reranking, adaptive chunking and context compression.
- **Self-correcting RAG:** Explore bounded query reformulation and retrieval-quality assessment as a separate extension.
- **LLM reliability:** Semantic validation of generated responses and expanded adversarial testing.
- **Multimodal retrieval:** Broader visual-document benchmarks and ColPali-style retrieval.

## Limitations and Responsible Use

CV Intelligence is a research and portfolio prototype evaluated primarily on synthetic data. A documented skill mention does not establish professional proficiency, verified experience or suitability for employment. Generated responses may contain inaccuracies and require human review.

The application supports evidence-based candidate search, not autonomous hiring decisions. Production deployment requires further privacy, security, fairness and operational validation.
