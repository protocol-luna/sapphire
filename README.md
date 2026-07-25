# Sapphire

LLM gateway / middleware for Protocol Luna.

Classifies each user message as **GENERIC** (trivial) or **SEMANTIC** (serious), then routes to the appropriate [Krystal](https://github.com/protocol-luna/krystal) backend.

## Architecture

```
Bot (Jade / Pixieglow)
  │ POST /v1/chat/completions
  ▼
Sapphire (:3123)
  │ classify message → GENERIC or SEMANTIC
  ├─ GENERIC  → Krystal-GENERIC  (:3124, Luna 1.5B)
  └─ SEMANTIC → Krystal-SEMANTIC (:3125, Discord-Hermes-8B)
```

- **Classifier**: [addyo07/distilbert-query-classifier](https://huggingface.co/addyo07/distilbert-query-classifier) — DistilBERT multilingual, 134M params, ONNX INT8, 98.4% accuracy, P99 16.87ms
- **Protocol**: OpenAI-compatible `/v1/chat/completions` with streaming support

## Setup

```bash
git clone https://github.com/protocol-luna/sapphire
cd sapphire
bash start.sh
```

## Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `SAPPHIRE_PORT` | `3123` | Server port |
| `KRYSTAL_GENERIC_URL` | `http://127.0.0.1:3124` | Backend for trivial messages |
| `KRYSTAL_SEMANTIC_URL` | `http://127.0.0.1:3125` | Backend for serious messages |

## License

MIT
