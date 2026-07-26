# Sapphire

LLM gateway / middleware for Protocol Luna.

Classifies each user message as **FUTILE** (trivial) or **INTERESSANT** (serious) using embedding centroid similarity, then routes to the appropriate [Krystal](https://github.com/protocol-luna/krystal) backend.

## Architecture

```
Bot (Jade / Pixieglow)
  │ POST /v1/respond { username, text, session_id, stream }
  ▼
Sapphire (:3123)
  │ 1. embed text (BAAI/bge-small-en-v1.5)
  │ 2. classify → FUTILE or INTERESSANT
  │ 3. score emotion → valence / arousal
  │ 4. manage session (TTL, history, slot hash)
  │ 5. inject few-shot examples
  │ 6. route + proxy to Krystal
  │ 7. detect degenerate output, retry if needed
  │ 8. stream tokens back via SSE
  ├─ FUTILE      → Krystal-GENERIC  (:3124, Luna 1.5B)
  └─ INTERESSANT → Krystal-SEMANTIC (:3125, e.g. Hermes)
```

## Setup

```bash
git clone https://github.com/protocol-luna/sapphire
cd sapphire
pip install -r requirements.txt
python server.py
```

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/respond` | POST | Main endpoint — classify + session + few-shot + route + stream |
| `/v1/chat/completions` | POST | OpenAI-compatible (no sessions, no few-shot) |
| `/v1/reset` | POST | Reset one or all sessions |
| `/classify` | POST | Classify only, returns label + emotion scores |
| `/emotion/{conv_key}` | GET | Current emotional state (valence/arousal) |
| `/health` | GET | Server status + active sessions |

## Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `SAPPHIRE_PORT` | `3123` | Server port |
| `KRYSTAL_GENERIC_URL` | `http://127.0.0.1:3124` | Backend for FUTILE messages |
| `KRYSTAL_SEMANTIC_URL` | `http://127.0.0.1:3125` | Backend for INTERESSANT messages |
| `SAPPHIRE_EXAMPLES` | `./examples.yml` | Path to classification examples |
| `SAPPHIRE_EMOTION_EXAMPLES` | `./examples_emotion.yml` | Path to emotion examples |
| `SAPPHIRE_SYSTEM_PROMPT` | `"Your name is Luna..."` | System prompt |
| `SAPPHIRE_LLM_SESSION_TTL` | `600` | Session expiry in seconds |
| `SAPPHIRE_LLM_MAX_HISTORY` | `20` | Max conversation pairs |
| `SAPPHIRE_LLM_MAX_RETRIES` | `2` | Retries on degenerate output |
| `SAPPHIRE_FEW_SHOT_ENABLED` | `true` | Enable few-shot priming |
| `SAPPHIRE_FEW_SHOT_EXAMPLES` | `./few_shot_examples.yml` | Few-shot examples file |
| `SAPPHIRE_MIROSTAT_ENABLED` | `true` | Enable mirostat sampling |
| `SAPPHIRE_MIROSTAT_MODE` | `2` | Mirostat mode (1 or 2) |
| `SAPPHIRE_MIROSTAT_LR` | `0.1` | Mirostat learning rate |
| `SAPPHIRE_MIROSTAT_ENT` | `5.0` | Mirostat target entropy |
| `SAPPHIRE_EMOTION_DECAY` | `0.85` | EMA decay for emotion state |
| `SAPPHIRE_EMOTION_DEADZONE` | `0.06` | Emotion noise threshold |
| `SAPPHIRE_LLM_N_SLOTS` | `1` | Number of llama.cpp slots |

## License

MIT
