# Sapphire

Sapphire is the LLM gateway for the Luna Protocol ecosystem. It sits between Emerald (the brain) and Krystal (llama.cpp), handling session management, few-shot example injection, emotion classification, degenerate response detection, and prompt construction.

> **Architecture**: `Emerald → Sapphire (HTTP) → Krystal (llama.cpp)`

## How It Works

1. Emerald sends user messages to Sapphire's `/v1/respond` HTTP endpoint
2. Sapphire classifies the message using fastembed + BAAI-bge-small-en-v1.5 centroid embeddings (emotional valence/arousal, conversation category)
3. Sapphire retrieves or creates a conversation session (last N messages for context)
4. Sapphire injects few-shot examples from YAML files matching the classified categories
5. Sapphire constructs the prompt with system prompt, few-shot examples, and conversation history
6. Sapphire calls Krystal's `/v1/chat/completions` with emotion-aware sampling parameters (temperature adjusted by arousal, repeat penalty by valence)
7. Sapphire checks the response for degenerate patterns and retries if needed
8. Sapphire returns the response text (and optionally debug stats: token counts, timing, emotion state, confidence)

## Features

- **Session management** — Per-channel/user conversation history with configurable context window
- **Few-shot learning** — Category-matched example injection from `few_shot_examples.yml`, `examples.yml`, `examples_emotion.yml`
- **Emotion classification** — Valence (positive/negative) and arousal (calm/aroused) on continuous axes
- **Category classification** — Multi-class conversation category via centroid embeddings
- **Emotion-aware sampling** — Temperature/repeat penalty/mirostat entropy dynamically adjusted by emotional state
- **Degenerate response detection** — Regex-based detection of loops, stutters, empty replies with automatic retry
- **Debug mode** — When `debug: true` in the request body, returns token counts, timing, emotion state, and confidence
- **Single-backend mode** — All routes forward to the same Krystal backend instance

## Endpoints

### POST `/v1/respond`

Request body:

```json
{
  "model": "default",
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "Hello!" },
    { "role": "assistant", "content": "Hi there!" }
  ],
  "channel_id": "123456",
  "user_id": "789012",
  "bot_username": "Luna",
  "behavior": {
    "mannerisms": { "enabled": true, "chance": 0.075 },
    "sleep": { ... }
  },
  "debug": false
}
```

Response (normal mode):

```json
{
  "text": "Hello! How can I help you today?"
}
```

Response (debug mode):

```json
{
  "text": "Hello! How can I help you today?",
  "debug": true,
  "debug_prompt_tokens": 450,
  "debug_completion_tokens": 12,
  "debug_time_ms": 3200,
  "debug_tokens_per_second": 3.75,
  "debug_emotion_state_valence": 0.6,
  "debug_emotion_state_arousal": 0.3,
  "debug_classification_confidence": 0.85
}
```

### Emotion-Aware Sampling

When calling Krystal, Sapphire dynamically adjusts these parameters:

- **Temperature**: `clamp(0.7 + arousal * 0.3, 0.4, 1.0)` — higher arousal = more randomness
- **Repeat penalty**: `clamp(1.15 - valence * 0.1, 1.0, 1.3)` — higher valence (positive) = less penalty
- **Mirostat entropy**: `clamp(6.0 + arousal * 2.0, 3.0, 8.0)` — higher arousal = more entropy

## Configuration

Key settings in `config.yml`:

```yaml
host: "0.0.0.0"
port: 3123
krystal_host: "localhost"
krystal_port: 3124
sessions:
  max_tokens: 3072
  max_messages: 20
model: "default"
few_shot:
  max_examples: 6
  min_similarity: 0.3
degenerate:
  max_length: 300
  retry_count: 2
```

## Running

```bash
# Install
pip install -r requirements.txt

# Development
python server.py

# Production (PM2)
pm2 start ecosystem.config.js
```
