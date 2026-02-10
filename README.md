# lang-extract

Extract characters, emotions, and relationships from text using [LangExtract](https://github.com/google/langextract) and an LLM (gemma3:1b on Ollama).

## Setup

- **Ollama**: run `ollama serve` and pull the model `gemma3:1b`

## Run

```bash
python3 main.py
```

Runs extraction, and writes:

- `romeo_juliet_extractions.jsonl` — extracted entities
- `romeo_juliet_visualization.html` — interactive view
