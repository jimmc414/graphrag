# Using GraphRAG with the Local Codex Provider

This runbook explains how to run GraphRAG with the bundled `openai_codex_chat` model that simulates an OpenAI Codex interface without requiring an API key.

## Prerequisites
- Python 3.10–3.12
- Clone this repository or install from PyPI

## Install Dependencies
If working from source, install development dependencies with:

```bash
uv sync --extra dev
```

For a PyPI install, simply run:

```bash
pip install graphrag
```

## Initialize a Workspace
Create a workspace containing input data and default config files:

```bash
graphrag init --root ./ragtest
```

This creates `.env` and `settings.yaml` in `./ragtest`.

## Configure the Codex Model
Edit `ragtest/settings.yaml` and add a model definition for `openai_codex_chat`:

```yaml
models:
  codex_local:
    type: openai_codex_chat
    model: dummy-codex
```

The local provider returns placeholder text but can be extended to wrap a real Codex model or local inference server.

## Index Documents
Place your source text files under `ragtest/input` then run:

```bash
graphrag index --root ./ragtest
```

## Query the Data
Ask questions using the codex model:

```bash
graphrag query --root ./ragtest --method global --query "<your question>"
```

The system will respond using the local Codex provider, which requires no API key.
