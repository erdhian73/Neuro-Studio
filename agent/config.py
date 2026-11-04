"""Static configuration and verified constants for the ad-pretest pipeline.

Every value here was confirmed against primary sources on 2026-11-04
(TRIBE v2 repo + HF config.yaml, Meta Model API docs, live endpoint probes).
Nothing in this file is inferred; see README.md "Verified facts" for evidence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# TRIBE v2 (source of truth: github.com/facebookresearch/tribev2)
# ---------------------------------------------------------------------------

TRIBE_REPO_ID = "facebook/tribev2"
TRIBE_GIT_URL = "git+https://github.com/facebookresearch/tribev2.git"
TRIBE_SRC_DIR = "/opt/tribev2-src"

# fsaverage5: 10_242 vertices per hemisphere, 20_484 total.
# TribeModel.predict returns (n_timesteps, n_vertices) and TRIBE splits
# hemispheres as data[:n//2] (left) and data[n//2:] (right) --
# verified in tribev2/plotting/base.py::get_stat_map.
N_VERTICES_PER_HEMI = 10_242
N_VERTICES_TOTAL = 2 * N_VERTICES_PER_HEMI

# TRIBE predictions are offset 5 s into the past to compensate the
# hemodynamic response lag (README: "offset by 5 seconds").
HEMODYNAMIC_LAG_S = 5.0

# The model's own TR is stored in its config; 1 s is the documented
# "per-TR" granularity used by the demo notebook (preds[:n_timesteps]).
DEFAULT_TR_S = 1.0

# Encoders TRIBE v2 loads, taken from the released config.yaml on the Hub.
ENCODER_ASSETS = {
    "text": "meta-llama/Llama-3.2-3B",
    "image": "facebook/dinov2-large",
    "video": "facebook/vjepa2-vitg-fpc64-256",
    "audio": "facebook/w2v-bert-2.0",
    "transcript": "whisperx large-v3 (via `uvx whisperx`)",
}

# Measured with HEAD on the Hub: best.ckpt is 708_856_138 bytes.
CHECKPOINT_BYTES = 708_856_138

# ---------------------------------------------------------------------------
# Meta Model API (source of truth: llama.developer.meta.com/docs)
# ---------------------------------------------------------------------------

MODEL_API_BASE_URL = "https://api.meta.ai/v1"
MODEL_API_KEY_ENV = "MODEL_API_KEY"

# Live-verified model catalogue (GET /v1/models, HTTP 200).
MUSE_SPARK_MODEL = "muse-spark-1.3"
MUSE_SPARK_CONTRIBUTOR_MODEL = "muse-spark-1.3-contributor"
MUSE_IMAGE_MODEL = "muse-image-1.0"

# "none" is rejected by Muse Spark (HTTP 400); minimal..max are supported.
# "max" is only available on muse-spark-1.3 (not the -contributor variant).
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")

# Muse Image constraints enforced by the API (probed: invalid values -> 400).
IMAGE_OUTPUT_FORMATS = ("png", "jpeg", "webp")
IMAGE_MAX_N = 10

# ---------------------------------------------------------------------------
# Pipeline thresholds (documented, overridable)
# ---------------------------------------------------------------------------

# Downsampled brain grid for the browser animation. Each hemisphere occupies a
# half-width slab, so a cell averages 10242 / (rows * cols_per_hemi) vertices.
# 10242 = 2 x 3 x 3 x 569, so rows*cols_per_hemi must divide it exactly;
# 64 rows x 80 cols per hemisphere (= 5120 cells, ~2 vertices each) does.
BRAIN_GRID_HEIGHT = 64
BRAIN_GRID_WIDTH = 160  # both hemispheres side by side (80 + 80)

# A clip is flagged as a "dead zone" when its predicted response stays in the
# bottom quartile of that clip's own dynamic range for at least this long.
LOW_ENGAGEMENT_QUANTILE = 0.25
LOW_ENGAGEMENT_MIN_S = 3.0

# Number of strongest / weakest windows sent to the LLM for interpretation.
N_KEY_WINDOWS = 5

DEFAULT_CAMPAIGN_BUDGET_USD = 0.0  # informational only, not used for routing


@dataclass(frozen=True)
class BrainGrid:
    """Geometry of the browser-friendly 2-D cortical strip."""

    width: int = BRAIN_GRID_WIDTH
    height: int = BRAIN_GRID_HEIGHT

    @property
    def n_pixels(self) -> int:
        return self.width * self.height


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable the way Modal users expect."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default
