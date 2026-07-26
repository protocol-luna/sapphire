"""
Test Sapphire emotion scoring — valence (positive/negative) and arousal (high/low).

Checks that /classify returns meaningful, directionally-correct emotion scores.
"""

import json
from urllib.request import Request, urlopen
from urllib.error import URLError

SAPPHIRE_URL = "http://127.0.0.1:3123/classify"
TIMEOUT = 10

def classify(text: str) -> dict:
    req = Request(
        SAPPHIRE_URL,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())

EXPECTED_VALENCE = [
    ("i love this", "positive"),
    ("you're amazing", "positive"),
    ("this is wonderful", "positive"),
    ("im so happy", "positive"),
    ("i hate everything", "negative"),
    ("this is terrible", "negative"),
    ("im so angry", "negative"),
    ("this sucks", "negative"),
    ("ok whatever", "neutral"),
    ("i guess", "neutral"),
    ("meh", "neutral"),
]

EXPECTED_AROUSAL = [
    ("OH MY GOD", "high"),
    ("AAAAA", "high"),
    ("im panicking", "high"),
    ("im so hyped", "high"),
    ("THIS IS INSANE", "high"),
    ("just chilling", "low"),
    ("im tired", "low"),
    ("nothing much", "low"),
    ("meh", "low"),
    ("bored af", "low"),
    ("taking it easy", "low"),
]


def main():
    ok_v = 0
    ok_a = 0

    print("=== Valence direction ===")
    for text, expected in EXPECTED_VALENCE:
        result = classify(text)
        v = result["valence"]
        if expected == "positive" and v > 0:
            ok_v += 1
        elif expected == "negative" and v < 0:
            ok_v += 1
        elif expected == "neutral" and abs(v) < 0.05:
            ok_v += 1
        else:
            print(f"  v={v:+.4f}  {text}  (expected {expected})")

    print(f"Valence: {ok_v}/{len(EXPECTED_VALENCE)}")

    print("\n=== Arousal direction ===")
    for text, expected in EXPECTED_AROUSAL:
        result = classify(text)
        a = result["arousal"]
        if expected == "high" and a > 0:
            ok_a += 1
        elif expected == "low" and a < 0:
            ok_a += 1
        else:
            print(f"  a={a:+.4f}  {text}  (expected {expected})")

    total = len(EXPECTED_VALENCE) + len(EXPECTED_AROUSAL)
    ok = ok_v + ok_a
    print(f"Arousal: {ok_a}/{len(EXPECTED_AROUSAL)}")
    print(f"\n=== Emotion TOTAL: {ok}/{total} ({ok/total*100:.1f}%) ===")


if __name__ == "__main__":
    main()
