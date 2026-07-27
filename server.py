import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import yaml
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="[sapphire] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sapphire")

_config_path = Path(__file__).resolve().parent / "config.yml"
with open(_config_path) as _f:
    _cfg = yaml.safe_load(_f)


def build_centroids():
    from fastembed import TextEmbedding
    from sapphire.classifier import (
        load_examples,
        compute_centroids,
        save_centroids,
        get_default_examples_path,
    )
    from sapphire.emotion import (
        load_emotion_examples,
        compute_emotion_centroids,
        save_emotion_centroids,
        get_default_emotion_examples_path,
    )

    examples_path = str(_cfg.get("examples_path", get_default_examples_path()))
    emotion_path = str(_cfg.get("emotion_examples_path", get_default_emotion_examples_path()))

    log.info("loading embedding model BAAI/bge-small-en-v1.5...")
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=128)

    log.info("loading examples from %s", examples_path)
    futile, interessant = load_examples(examples_path)
    log.info("  %d futile, %d interessant examples", len(futile), len(interessant))

    log.info("computing classification centroids...")
    f_c, i_c = compute_centroids(embedder, futile, interessant)
    save_centroids(f_c, i_c)
    log.info("  saved")

    log.info("loading emotion examples from %s", emotion_path)
    emotion_examples = load_emotion_examples(emotion_path)
    for pole, texts in emotion_examples.items():
        log.info("  %s: %d examples", pole, len(texts))

    log.info("computing emotion centroids...")
    e_c = compute_emotion_centroids(embedder, emotion_examples)
    save_emotion_centroids(e_c)
    log.info("  saved")

    log.info("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="sapphire")
    parser.add_argument(
        "--build-centroids",
        action="store_true",
        help="Precompute classification and emotion centroids from YAML and exit",
    )
    args = parser.parse_args()

    if args.build_centroids:
        build_centroids()
        sys.exit(0)

    port = int(_cfg.get("port", 3123))
    uvicorn.run("sapphire.server:app", host="127.0.0.1", port=port, log_level="info")
