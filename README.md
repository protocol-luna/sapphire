# Sapphire

Sapphire is the LLM gateway for the Luna Protocol ecosystem. It sits between Emerald (the brain) and Krystal (llama.cpp), handling session management, few-shot example injection, emotion classification, degenerate response detection, and prompt construction.

> **Architecture**: `Emerald → Sapphire (HTTP) → Krystal (llama.cpp)`

## How It Works

1. Emerald sends user messages to Sapphire's `/v1/respond` HTTP endpoint
2. Sapphire classifies the message using fastembed + BAAI-bge-small-en-v1.5 centroid embeddings (FUTILE/INTERESSANT + emotional valence/arousal)
3. Sapphire retrieves or creates a conversation session with per-channel history
4. Sapphire injects few-shot examples from YAML files into the conversation (right after system prompt)
5. Sapphire constructs the prompt with system message, few-shot examples, and conversation history
6. Sapphire calls Krystal's `/v1/chat/completions` with emotion-aware sampling parameters
7. Sapphire checks the response for degenerate patterns and retries if needed (non-streaming)
8. Sapphire returns the response text (and optionally debug stats)

## Endpoints

### POST `/v1/respond`

Main endpoint called by Emerald. Accepts a text message and session ID, returns a generated response.

**Request**:
```json
{
  "username": "User",
  "text": "hello how are you?",
  "session_id": "jade:123456",
  "stream": false,
  "debug": false
}
```

**Response (non-streaming)**:
```json
{
  "text": "Hey! I'm good, how about you?",
  "label": "FUTILE",
  "backend": "http://127.0.0.1:3124",
  "valence": 0.12,
  "arousal": -0.04
}
```

**Response (debug mode)**:
```json
{
  "text": "Hey! I'm good, how about you?",
  "label": "FUTILE",
  "backend": "http://127.0.0.1:3124",
  "valence": 0.12,
  "arousal": -0.04,
  "debug_prompt_tokens": 450,
  "debug_completion_tokens": 12,
  "debug_time_ms": 3200,
  "debug_tokens_per_second": 3.75,
  "debug_emotion_state_valence": 0.6,
  "debug_emotion_state_arousal": 0.3,
  "debug_classification_confidence": 0.85
}
```

### Streaming (`stream: true`)

When `stream: true` is set, Sapphire returns an SSE (Server-Sent Events) stream:

```
data: Hey
data: !
data:  I'm
data:  good
data: , how
data:  about
data:  you
data: ?
data: {"text":"Hey! I'm good, how about you?","label":"FUTILE","backend":"http://127.0.0.1:3124","valence":0.12,"arousal":-0.04,"time_ms":3200,"tokens_per_second":3.8}
data: [DONE]
```

The stream contains:
1. One `data: <token>` line per content token
2. A final `data: {json}` line with complete metadata (includes debug fields when `debug: true`)
3. A `data: [DONE]` terminator

If the output is degenerate, the stream stops early -- no metadata, just `data: [DONE]`. The session is not updated with degenerate content.

### Other endpoints

- **POST `/v1/reset`** -- Reset a session (or all sessions)
- **POST `/classify`** -- Classify text without generating a response
- **GET `/emotion/{conv_key}`** -- Current emotional state for a conversation
- **GET `/health`** -- Health check with system status

## Features

### Emotion Classification

Each message is scored on two continuous axes:
- **Valence** (−1 to +1): negative to positive sentiment
- **Arousal** (−1 to +1): calm to aroused

Centroids are computed from `examples_emotion.yml` (83 positive, 79 negative, 85 high-arousal, 116 low-arousal samples).

### Emotion-Aware Sampling

When calling Krystal, Sapphire dynamically adjusts sampling parameters:

- **Temperature**: `clamp(0.7 + arousal * 0.3, 0.4, 1.0)` -- higher arousal = more randomness
- **Repeat penalty**: `clamp(1.15 - valence * 0.1, 1.0, 1.3)` -- higher valence = less penalty
- **Mirostat entropy**: `clamp(6.0 + arousal * 2.0, 3.0, 8.0)` -- higher arousal = more entropy
- **Emotion deadzone**: 0.005 -- per-message scores below this threshold don't update the running state

### Session Management

Per-channel/user conversation history with configurable parameters:
- `LLM_MAX_HISTORY` (default: 20) -- max messages kept
- `LLM_SESSION_TTL` (default: 600s) -- session expiry
- `LLM_N_SLOTS` (default: 1) -- concurrent slots

### Few-Shot Learning

Up to 5 example exchanges from `few_shot_examples.yml` are injected after the system prompt and before conversation history. Username prefix is prepended to user examples (e.g., `"User: hello"`).

### Degenerate Response Detection

Regex-based detection of empty, too-short, or character-repetitive outputs. In **non-streaming** mode, Sapphire retries up to `LLM_MAX_RETRIES` (default: 2) times. In **streaming** mode, degenerate output is discarded after buffering -- the client never receives it.

### Backend Routing

FUTILE messages route to `KRYSTAL_GENERIC_URL` and INTERESSANT messages route to `KRYSTAL_SEMANTIC_URL`. By default these point to different ports (`:3124` / `:3125`) -- set both to the same URL to run a single backend.

### System Prompt

The system prompt is configurable via `SAPPHIRE_SYSTEM_PROMPT` and includes:
- Bot name (from `SAPPHIRE_BOT_NAME`, default: "Luna")
- Personality description
- Explicit "never repeat or echo" instruction

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SAPPHIRE_PORT` | 3123 | HTTP port |
| `KRYSTAL_GENERIC_URL` | `http://127.0.0.1:3124` | Backend for FUTILE messages |
| `KRYSTAL_SEMANTIC_URL` | `http://127.0.0.1:3125` | Backend for INTERESSANT messages |
| `SAPPHIRE_BOT_NAME` | "Luna" | Bot persona name |
| `SAPPHIRE_EMOTION_DEADZONE` | 0.005 | Emotion update threshold |
| `SAPPHIRE_EMOTION_DECAY` | 0.85 | Emotion decay factor |
| `SAPPHIRE_FEW_SHOT_ENABLED` | true | Enable few-shot injection |
| `SAPPHIRE_LLM_MAX_RETRIES` | 2 | Degenerate retry count |
| `SAPPHIRE_MIROSTAT_ENABLED` | true | Enable mirostat sampling |

## Running

```bash
# Install
pip install -r requirements.txt

# Development
python server.py

# Production (PM2)
pm2 start
```
