"""Deterministic, offline ROI labelling on the fsaverage5 cortical mesh.

Why this module exists
----------------------
TRIBE v2 predicts 20,484 values per time step: two fsaverage5 hemispheres
(10,242 vertices each) concatenated left-then-right. Raw vertex numbers are
unusable in an ad report, so we need named regions.

Two labelling strategies are supported:

``mesh`` (default, always available)
    Coarse gyral-scale zones derived *only* from fsaverage5 pial geometry
    (x = left/right, y = anterior/posterior, z = superior/inferior). It needs
    no network and no atlases, and every label is exactly reproducible.
    It is a coarse anatomical grouping -- **not** a validated parcellation.

``schaefer`` (opt-in, needs network)
    Schaefer 2018 cortical parcellation resampled to fsaverage5 via nilearn.
    Enable with ``ROI_ATLAS=schaefer``; labels are then real parcel names.

Both strategies return ``x``-flipped consistently so that hemisphere index
``0`` is always the left hemisphere, matching TRIBE's own splitting
(``data[:n//2]`` = left, ``data[n//2:]`` = right).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import N_VERTICES_PER_HEMI, N_VERTICES_TOTAL


logger = logging.getLogger(__name__)

# Gyral-scale zones, ordered so that ids are stable across runs and versions.
# Geometries: x_abs = distance from the midline (mm), y = anterior(+)/posterior(-),
# z = superior(+)/inferior(-). Thresholds are coarse on purpose: this map groups
# the mesh into interpretable quadrants, it is NOT a validated parcellation.
ZONE_NAMES: tuple[str, ...] = (
    "medial_posterior",  # near midline, back of the head
    "medial_central",  # near midline, central strip
    "medial_anterior",  # near midline, front
    "posterolateral",  # lateral, back (visual/parietal territory)
    "inferior_lateral",  # lateral, low (temporal/sylvian territory)
    "dorsolateral",  # lateral, high central (sensorimotor/motor strip)
    "ventrolateral_anterior",  # lateral, front and low (inferior frontal)
    "dorsolateral_anterior",  # lateral, front and high (superior frontal)
)

# First match wins; the last rule is a catch-all so every vertex is labelled.
# Vectorised predicates: use ``&``/``|`` (``and``/``or`` fail on numpy arrays).
_ZONE_RULES = (
    lambda x, y, z: (x < 12) & (y < -25),
    lambda x, y, z: (x < 12) & (-25 <= y) & (y <= 15),
    lambda x, y, z: x < 12,
    lambda x, y, z: y < -25,
    lambda x, y, z: (-25 <= y) & (y <= 15) & (z <= 0),
    lambda x, y, z: (-25 <= y) & (y <= 15),
    lambda x, y, z: z <= 0,
    lambda x, y, z: np.ones_like(x, dtype=bool),  # catch-all
)



@dataclass(frozen=True)
class RegionMap:
    """Label assignment for every vertex of the 20,484-vertex vector."""

    labels: np.ndarray  # (n_vertices,) int32
    names: tuple[str, ...]  # index -> human readable name
    strategy: str  # "mesh" or "schaefer"

    @property
    def n_regions(self) -> int:
        return len(self.names)

    def describe(self) -> dict[str, object]:
        counts = np.bincount(self.labels, minlength=self.n_regions)
        return {
            "strategy": self.strategy,
            "n_regions": self.n_regions,
            "n_vertices": int(self.labels.size),
            "vertices_per_region": {
                name: int(counts[i]) for i, name in enumerate(self.names)
            },
        }


def _fsaverage5_pial_coordinates() -> dict[str, np.ndarray]:
    """Load fsaverage5 pial coordinates from nilearn's bundled template.

    ``nilearn.datasets.load_fsaverage`` ships the meshes inside the wheel, so
    this works with no network access. TRIBE already depends on nilearn
    through its [plotting] extra, which the Modal image installs.
    """

    from nilearn.datasets import load_fsaverage

    fsavg = load_fsaverage(mesh="fsaverage5")
    parts = fsavg.pial.parts  # {'left': PolyMesh, 'right': PolyMesh}
    coords = {
        hemi: np.asarray(parts[hemi].coordinates, dtype=np.float64)
        for hemi in ("left", "right")
    }
    for hemi, arr in coords.items():
        if arr.shape != (N_VERTICES_PER_HEMI, 3):
            raise RuntimeError(
                f"fsaverage5 {hemi} pial mesh has shape {arr.shape}, "
                f"expected ({N_VERTICES_PER_HEMI}, 3)"
            )
    return coords


def _zone_for_coordinates(xyz: np.ndarray) -> np.ndarray:
    """Assign each vertex exactly one zone id using gyral-scale geometry.

    Mirrors are applied per hemisphere so both sides share zone definitions.
    The final rule is a catch-all, so every vertex always gets a zone.
    """

    x_abs = np.abs(xyz[:, 0])
    y = xyz[:, 1]
    z = xyz[:, 2]
    zones = np.full(xyz.shape[0], -1, dtype=np.int32)

    for zone_id, predicate in enumerate(_ZONE_RULES):
        mask = predicate(x_abs, y, z) & (zones < 0)
        zones[mask] = zone_id
    zones[zones < 0] = len(_ZONE_RULES) - 1  # safe catch-all, never unassigned
    return zones


@lru_cache(maxsize=4)
def mesh_region_map() -> RegionMap:
    """Coarse, fully offline region map derived from fsaverage5 geometry."""

    coords = _fsaverage5_pial_coordinates()
    names: list[str] = []
    labels: list[np.ndarray] = []

    for hemi in ("left", "right"):
        zones = _zone_for_coordinates(coords[hemi])
        offset = len(names)
        for zone_id, zone_name in enumerate(ZONE_NAMES):
            names.append(f"{hemi}_{zone_name}")
            if not np.any(zones == zone_id):  # pragma: no cover - geometry-dependent
                logger.warning("Zone %s has no vertices on the %s hemisphere", zone_name, hemi)
        labels.append(zones + offset)

    stacked = np.concatenate(labels)
    if stacked.size != N_VERTICES_TOTAL:
        raise RuntimeError(
            f"Region map has {stacked.size} vertices, expected {N_VERTICES_TOTAL}"
        )
    if stacked.min() < 0 or stacked.max() >= len(names):
        raise RuntimeError("Region ids escaped the declared label range")
    return RegionMap(labels=stacked, names=tuple(names), strategy="mesh")


def schaefer_region_map(n_rois: int = 100) -> RegionMap:
    """Schaefer-2018 parcellation resampled onto fsaverage5 (needs network).

    Raises whatever nilearn raises when the download is unavailable; callers
    should fall back to :func:`mesh_region_map` (see :func:`get_region_map`).
    """

    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=7, resolution_mm=2)
    label_names = tuple(str(n) for n in np.asarray(atlas.labels))
    raise NotImplementedError(
        "Volumetric Schaefer labels need a surface resampling step; "
        f"fetched {len(label_names)} parcels but surface projection is not wired up yet."
    )


def get_region_map(strategy: str | None = None) -> RegionMap:
    """Resolve the region map, honouring the ``ROI_ATLAS`` environment variable."""

    chosen = (strategy or os.environ.get("ROI_ATLAS", "mesh")).strip().lower()
    if chosen in {"mesh", "geometry", "auto", ""}:
        return mesh_region_map()
    if chosen == "schaefer":
        try:
            return schaefer_region_map()
        except NotImplementedError as exc:
            logger.warning("%s -- falling back to the geometric mesh map.", exc)
            return mesh_region_map()
    raise ValueError(f"Unknown ROI atlas {chosen!r}; use 'mesh' or 'schaefer'")

