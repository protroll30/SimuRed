# SimuRed

FastAPI service that **mutates a user prompt**, **queries an LLM** for each variant, **scores semantic drift** against the baseline reply, and **optionally logs runs to Supabase**. The goal is lightweight red-team–style probing (typos and synonym-style perturbations) with embedding-based consistency checks.

## What it does

1. **Prompt mutation** (`app/services/mutator.py`): builds three strings—original, keyboard-noise typo, and WordNet synonym swap (via [nlpaug](https://github.com/makcedward/nlpaug)).
2. **LLM calls** (`app/services/llm_client.py`): [LiteLLM](https://github.com/BerriAI/litellm) async completion to **Google Gemini** (`gemini/gemini-2.5-flash` by default).
3. **Drift evaluation** (`app/services/evaluator.py`): [Sentence-Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) cosine similarity between each attack response and the **original** response; flags stability with a **0.85** similarity threshold.
4. **Persistence** (`app/services/database.py`): each simulation row is inserted into Supabase table **`simulations`** (failures are logged to stdout; the API still returns JSON).

## Requirements

- Python 3.10+ recommended  
- GPU optional (PyTorch / sentence-transformers will use CPU if needed)  
- NLTK corpora: first synonym augmentation may trigger **WordNet** / POS data downloads on the machine running the app.

## Environment variables

Create a `.env` in the project root:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google AI Studio key for LiteLLM → Gemini |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase service or anon key (must allow inserts used by the client) |

## Supabase schema

The code expects a table named **`simulations`** with columns compatible with:

- `original_prompt` (text)
- `attack_type` (text)
- `mutated_input` (text)
- `ai_output` (text)
- `similarity_score` (numeric; stored for attack rows)
- `is_stable` (boolean)

Adjust column types in Supabase to match your conventions (e.g. `float8`, `text`).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API

From the project root (so `app` resolves as a package):

```bash
uvicorn app.main:app --reload
```

- **Health**: `GET /` → `{"message": "SimuRed System Active"}`
- **Pipeline**: `POST /monitor/test-logic` with JSON body `{"prompt": "your text here"}`

Example:

```bash
curl -X POST http://127.0.0.1:8000/monitor/test-logic -H "Content-Type: application/json" -d "{\"prompt\": \"Hello, who are you?\"}"
```

Open **Swagger UI** at `http://127.0.0.1:8000/docs`.

## Project layout

```
app/
  main.py              # FastAPI app, mounts monitor router
  routers/monitor.py   # /monitor/test-logic
  services/
    mutator.py         # nlpaug typo + synonym attacks
    llm_client.py      # LiteLLM + Gemini
    evaluator.py       # embedding similarity / drift
    database.py        # Supabase insert helper
workers/tasks.py       # placeholder (empty); Celery not wired in app code yet
requirements.txt
```

## Dependencies note

`requirements.txt` includes **Celery** and **Redis** for a future task queue; the current HTTP flow runs entirely inside FastAPI. **scikit-learn**, **pandas**, and **instructor** are listed for broader ML / LLM tooling but are not required by the core `/monitor/test-logic` path described above.

## License

[MIT](LICENSE) — Copyright (c) 2026 protroll30.
