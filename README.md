# Outcome-QBank (Syllabus-Aware)

A Streamlit app that generates an outcome-aligned question bank from course PDFs. It uses retrieval over uploaded materials and sample papers, plans subtopics from syllabus context, detects subject style, and produces questions with CO/Bloom/difficulty tags, detailed answers, rubrics, and citations. A second LLM pass audits quality and alignment, then retries up to 3 iterations.

## Features

- Upload course PDFs and optional sample papers (style only, no copying).
- Chunking and retrieval over Chroma DB with SentenceTransformer embeddings.
- Subject detection to recommend question-type mix (theory, numerical, derivation, equation, diagram).
- Mark distribution controls (1/2/5-mark) and difficulty mix.
- LLM-based audit loop with structured critique and improvement logs.
- Coverage report + export to CSV/PDF.

## Quick Start (Local)

1) Create and activate a venv
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies
```powershell
pip install -r app/runs/requirements.txt
```

3) Set OpenAI key
Create `app/runs/.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY="your_api_key_here"
```

4) Run the app
```powershell
streamlit run app/ui_streamlit.py
```

## Streamlit Community Cloud Deployment

1) Push this repo to GitHub.
2) Go to https://share.streamlit.io and connect your GitHub account.
3) Select the repo and set:
   - Main file path: `app/ui_streamlit.py`
4) Add the secret:
   - `OPENAI_API_KEY` in the Streamlit Cloud secrets UI.
5) Deploy.

## Usage Flow

1) Upload PDFs (course material + optional sample papers).
2) Click "Ingest PDFs into Vector DB".
3) Load syllabus context, detect subject, and plan subtopics.
4) Retrieve context for subtopics.
5) Generate questions, review logs and audit issues.
6) Export CSV/PDF with coverage report.

## Notes

- Re-ingest PDFs if you change chunking settings or update the vector store schema.
- Sample papers are used for style only; the model is instructed not to copy them.
- PII detection warns but does not block ingestion.
