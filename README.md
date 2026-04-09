# ESG Insights Dashboard

A privacy-preserving, multi-agent ESG data extraction pipeline. Upload any sustainability PDF — the pipeline masks sensitive data, extracts environmental metrics, and audits them for hallucinations.

## Features

- PII and financial data masking before LLM processing (Microsoft Presidio)
- Agent A extracts 10 environmental metrics — Scope 1/2/3, water, energy, waste, targets
- Agent B cross-verifies every metric and flags hallucinations with confidence scores
- Role-gated vector search — Guest sees masked data, Admin sees everything
- Redacted report download for Guest role

## Tech Stack

Streamlit, Groq (Llama 3.3 70B), Microsoft Presidio, FAISS, LangChain, PyMuPDF, Plotly

## Setup

```bash
git clone https://github.com/ak-kapur/-esg-pipeline.git
cd -esg-pipeline
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run:
```bash
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Project Structure

```
├── app.py              # Streamlit dashboard
├── agents.py           # Agent A (Extractor) + Agent B (Auditor)
├── privacy_layer.py    # Presidio PII and financial masking
├── pdf_ingestion.py    # PDF parsing and chunking
├── vector_store.py     # FAISS vector store with role-gated retrieval
└── config.py           # Configuration
```

## System Architecture Diagram

![ESG Pipeline Architecture](docs/System_architecture.png)