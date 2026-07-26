"""
Test offline du suivi d'état (EmotionState) sur des conversations simulées.

Contrairement à test_emotion.py (qui valide juste que positive/negative et
high_arousal/low_arousal se séparent bien message par message), ce script
vérifie le comportement de l'état PERSISTANT au fil d'une conversation :
- est-ce que l'état monte dans le bon sens quand plusieurs messages du même
  ton s'enchaînent ?
- est-ce que l'état redescend vers 0 (decay) quand la conversation redevient
  neutre ?
- est-ce que l'état reste borné (pas de dérive qui s'envole avec le temps) ?
- est-ce que deux conversations (conv_key différentes) restent isolées ?

Usage:
  pip install fastembed pyyaml numpy
  python test_emotion_state.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastembed import TextEmbedding

from sapphire.emotion import (
    load_emotion_examples,
    compute_emotion_centroids,
    score_axes,
    get_default_emotion_examples_path,
    EmotionState,
)

# Scénarios : liste de tours (texte) + ce qu'on attend de la trajectoire.
# "direction" est vérifiée sur la dernière valeur vs la première valeur
# non-nulle de la séquence "montante", pas message par message (le score
# d'un seul message est bruité, la state machine lisse via le decay).
SCENARIOS = {
    "escalade_colere": {
        "turns": [
            "hey whats up",
            "why didnt you do what i asked",
            "i told you already, this is annoying",
            "seriously stop ignoring me",
            "i cant stand this anymore",
        ],
        "expect_valence_end": "negative",
        "expect_arousal_end": "high",
    },
    "calme_puis_neutre": {
        "turns": [
            "thank you so much this is great",
            "yeah im really happy about it",
            "love how this turned out",
            "nothing new to report",
            "same as before",
            "no updates on my end",
            "still working on it",
            "havent had time to check",
        ],
        # les 3 premiers messages tirent valence+, arousal bas ;
        # les 5 suivants sont réellement plats (pas des affirmations
        # casuelles type "cool"/"sure", qui lisent mollement positif sur
        # ce modèle) -> l'état doit redescendre vers 0
        "expect_decay_after_turn": 3,
    },
}


def run_scenario(name: str, scenario: dict, embedder, centroids, decay: float, deadzone: float):
    state = EmotionState(decay=decay, deadzone=deadzone)
    trajectory = []

    for text in scenario["turns"]:
        emb = next(embedder.query_embed(text))
        valence, arousal = score_axes(emb, centroids)
        s = state.update(name, valence, arousal)
        trajectory.append((text, valence, arousal, s["valence"], s["arousal"]))

    print(f"\n=== {name} ===")
    for text, v, a, sv, sa in trajectory:
        print(f"  raw(v={v:+.3f} a={a:+.3f}) -> state(v={sv:+.3f} a={sa:+.3f}) | {text}")

    checks_ok = True

    if "expect_valence_end" in scenario:
        end_v = trajectory[-1][3]
        want_negative = scenario["expect_valence_end"] == "negative"
        ok = (end_v < -0.01) if want_negative else (end_v > 0.01)
        print(f"  [valence finale {'négative' if want_negative else 'positive'} attendue] "
              f"v={end_v:+.3f} -> {'OK' if ok else 'MISS'}")
        checks_ok &= ok

    if "expect_arousal_end" in scenario:
        end_a = trajectory[-1][4]
        want_high = scenario["expect_arousal_end"] == "high"
        ok = (end_a > 0.01) if want_high else (end_a < -0.01)
        print(f"  [arousal finale {'haute' if want_high else 'basse'} attendue] "
              f"a={end_a:+.3f} -> {'OK' if ok else 'MISS'}")
        checks_ok &= ok

    if "expect_decay_after_turn" in scenario:
        i = scenario["expect_decay_after_turn"]
        peak_v = trajectory[i - 1][3]
        final_v = trajectory[-1][3]
        # la valence doit se rapprocher de 0 après la séquence neutre
        ok = abs(final_v) < abs(peak_v)
        print(f"  [decay attendu après le tour {i}] peak={peak_v:+.3f} -> final={final_v:+.3f} "
              f"-> {'OK' if ok else 'MISS'}")
        checks_ok &= ok

    # bornage : même après une longue séquence, l'état ne doit jamais sortir de [-1, 1]
    out_of_bounds = [
        (sv, sa) for (_, _, _, sv, sa) in trajectory
        if abs(sv) > 1.0 or abs(sa) > 1.0
    ]
    if out_of_bounds:
        print(f"  [bornage] DEPASSEMENT détecté: {out_of_bounds}")
        checks_ok = False
    else:
        print("  [bornage] OK, état toujours dans [-1, 1]")

    return checks_ok


def test_isolation(embedder, centroids, decay: float, deadzone: float):
    """Deux conv_key différentes ne doivent pas se mélanger."""
    state = EmotionState(decay=decay, deadzone=deadzone)

    angry_emb = next(embedder.query_embed("i hate this so much"))
    calm_emb = next(embedder.query_embed("just chilling, all good"))

    v_angry, a_angry = score_axes(angry_emb, centroids)
    v_calm, a_calm = score_axes(calm_emb, centroids)

    state.update("conv_A", v_angry, a_angry)
    state.update("conv_B", v_calm, a_calm)

    a_state = state.get("conv_A")
    b_state = state.get("conv_B")

    print("\n=== isolation entre conversations ===")
    print(f"  conv_A (colère) -> v={a_state['valence']:+.3f} a={a_state['arousal']:+.3f}")
    print(f"  conv_B (calme)  -> v={b_state['valence']:+.3f} a={b_state['arousal']:+.3f}")

    ok = a_state["valence"] != b_state["valence"] and a_state != b_state
    print(f"  [isolation] {'OK' if ok else 'MISS — les deux états sont identiques'}")
    return ok


def main():
    print("chargement de BAAI/bge-small-en-v1.5...")
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=128)

    path = get_default_emotion_examples_path()
    print(f"chargement des exemples depuis {path}")
    examples = load_emotion_examples(path)
    centroids = compute_emotion_centroids(embedder, examples)

    decay = 0.85  # doit matcher SAPPHIRE_EMOTION_DECAY côté server.py
    deadzone = 0.06  # doit matcher SAPPHIRE_EMOTION_DEADZONE côté server.py
    results = [
        run_scenario(name, scenario, embedder, centroids, decay, deadzone)
        for name, scenario in SCENARIOS.items()
    ]
    results.append(test_isolation(embedder, centroids, decay, deadzone))

    total_ok = sum(results)
    print(f"\n=== TOTAL: {total_ok}/{len(results)} scénarios OK ===")
    if total_ok < len(results):
        print(
            "Si un scénario échoue : vérifier d'abord test_emotion.py (séparation\n"
            "des pôles au niveau message) avant de suspecter le decay ou l'état —\n"
            "un mauvais centroïde produit un signal bruité que le lissage ne peut\n"
            "pas corriger."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
