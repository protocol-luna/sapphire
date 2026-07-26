"""
Test offline des axes émotionnels (valence, arousal).

Contrairement à test_classifier.py, ce script ne passe pas par le serveur
HTTP — il calcule les centroïdes et scorre directement en local, ce qui
permet de valider la séparation des pôles avant même de lancer Sapphire.

Usage:
  pip install fastembed pyyaml numpy
  python test_emotion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastembed import TextEmbedding

from sapphire.emotion import (
    load_emotion_examples,
    compute_emotion_centroids,
    _axis_score,
    get_default_emotion_examples_path,
)

# Phrases de validation, distinctes des exemples de calibration —
# pour vérifier que ça généralise et pas juste que ça "retrouve" ses exemples.
VALIDATION = [
    # (texte, valence_attendue, arousal_attendu)  — +1 haut, -1 bas
    ("thank you so much for this", +1, -1),
    ("i really appreciate it", +1, -1),
    ("get away from me", -1, -1),
    ("i cant stand you anymore", -1, +1),
    ("THIS IS THE BEST DAY EVER", +1, +1),
    ("im about to lose it", -1, +1),
    ("yeah thats fine i guess", 0, -1),
    ("k", 0, -1),
    ("wait what just happened??", 0, +1),
    ("im so proud of you right now", +1, +1),
]


def main():
    print("chargement de BAAI/bge-small-en-v1.5...")
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=128)

    path = get_default_emotion_examples_path()
    print(f"chargement des exemples depuis {path}")
    examples = load_emotion_examples(path)
    for pole, texts in examples.items():
        print(f"  {pole}: {len(texts)} exemples")

    print("calcul des centroïdes...")
    centroids = compute_emotion_centroids(embedder, examples)

    print("\n=== validation (hors exemples de calibration) ===")
    valence_ok = arousal_ok = 0
    for text, exp_val, exp_aro in VALIDATION:
        emb = next(embedder.query_embed(text))
        val = _axis_score(emb, centroids["positive"], centroids["negative"])
        aro = _axis_score(emb, centroids["high_arousal"], centroids["low_arousal"])

        val_sign = 1 if val > 0.02 else (-1 if val < -0.02 else 0)
        aro_sign = 1 if aro > 0.02 else (-1 if aro < -0.02 else 0)
        val_match = "OK" if val_sign == exp_val else "MISS"
        aro_match = "OK" if aro_sign == exp_aro else "MISS"
        valence_ok += val_match == "OK"
        arousal_ok += aro_match == "OK"

        print(
            f"  valence={val:+.3f} [{val_match:4}] arousal={aro:+.3f} [{aro_match:4}] | {text}"
        )

    n = len(VALIDATION)
    print(f"\nvalence: {valence_ok}/{n} corrects | arousal: {arousal_ok}/{n} corrects")
    print(
        "\nSi beaucoup de MISS : enrichir examples_emotion.yml avec des\n"
        "exemples plus proches du style Discord réel (plutôt que des phrases\n"
        "'propres' comme ci-dessus), et vérifier que les deux pôles de chaque\n"
        "axe ne se chevauchent pas trop (ex: 'negative' et 'high_arousal' qui\n"
        "capturent tous les deux la colère peuvent brouiller le signal)."
    )


if __name__ == "__main__":
    main()
