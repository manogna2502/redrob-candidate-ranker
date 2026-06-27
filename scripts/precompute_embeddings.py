#!/usr/bin/env python3
"""
One-time setup script: downloads and caches the sentence-transformers model
used by the semantic backend, so the timed `rank.py` run never needs network
access.

Run this ONCE, with network access, before running rank.py in a no-network
sandbox:

    python3 scripts/precompute_embeddings.py

This does NOT precompute candidate embeddings (those depend on which JD/pool
you're ranking and are cheap enough to compute fresh inside the 5-minute
budget -- see the runtime numbers in README.md). It only ensures the model
checkpoint itself is present in the local sentence-transformers cache
(typically ~/.cache/torch/sentence_transformers/ or HF_HOME, depending on
your environment).

If this script is never run, or fails (no network, disk issues, etc.),
rank.py will simply fall back to the TF-IDF backend automatically -- nothing
breaks, you just get a slightly lower semantic-quality ranking. See
src/redrob_ranker/semantic.py for the fallback logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redrob_ranker.semantic import SBERT_MODEL_NAME


def main() -> None:
    print(f"Attempting to download and cache: {SBERT_MODEL_NAME}")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(SBERT_MODEL_NAME)
        # Smoke-test encode to make sure the model actually works end-to-end,
        # not just that the files downloaded.
        vec = model.encode("smoke test sentence", convert_to_numpy=True)
        print(f"Success. Model cached and verified (embedding dim = {len(vec)}).")
        print("You can now run rank.py in a no-network environment; it will "
              "automatically use this cached model.")
    except ImportError:
        print(
            "sentence-transformers is not installed. Install it with:\n"
            "  pip install sentence-transformers --break-system-packages\n"
            "Or simply skip this step -- rank.py will use the TF-IDF fallback "
            "backend instead, with no further action required."
        )
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Failed to download/cache the model: {type(exc).__name__}: {exc}\n"
            "This is most likely a network issue. rank.py will use the TF-IDF "
            "fallback backend automatically if this model is unavailable at "
            "run time -- no further action is strictly required, but semantic "
            "match quality will be somewhat lower without the SBERT backend."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
