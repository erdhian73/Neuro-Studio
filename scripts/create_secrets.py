#!/usr/bin/env python3
"""Create the Modal secret this app expects, using exact key names.

Usage (never hard-code tokens in source control)::

    export MODEL_API_KEY=LLM_...          # Meta Model API key (Muse Spark/Image)
    export HF_TOKEN=hf_...                # Hugging Face token with Llama-3.2 access
    python scripts/create_secrets.py

The script is idempotent: re-running it updates the secret in place.
"""

from __future__ import annotations

import os
import sys

import modal

SECRET_NAME = "muse-model-api"
REQUIRED_KEYS = ("MODEL_API_KEY", "HF_TOKEN")


def main() -> int:
    missing = [key for key in REQUIRED_KEYS if not os.environ.get(key)]
    if missing:
        print(
            "Missing environment variable(s): " + ", ".join(missing) + "\n"
            "Export them first, e.g.\n"
            "  export MODEL_API_KEY='LLM_...'\n"
            "  export HF_TOKEN='hf_...'",
            file=sys.stderr,
        )
        return 2

    values = {key: os.environ[key].strip() for key in REQUIRED_KEYS}

    with modal.enable_output():
        # allow_existing makes the script idempotent (updates in place).
        modal.Secret.objects.create(SECRET_NAME, env_dict=values, allow_existing=True)

    print(f"Secret '{SECRET_NAME}' created/updated with keys: {', '.join(REQUIRED_KEYS)}")
    print("Now run:  modal deploy app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
