"""Encode TRIBE vertex predictions as a compact, browser-animatable payload.

Why not render PNGs server-side
-------------------------------
Rendering 20k vertices per second of video with matplotlib/pyvista would cost
seconds of GPU-less CPU per frame and bloat responses. Instead the backend
downsamples each vertex to a coarse 2-D "cortical strip" grid and the browser
interpolates the heatmap. This keeps the JSON small (tens of kB per clip) and
lets a single payload drive an instant, scrubable brain animation.

Layout (deterministic, documented in ``canonical_vertices``)
-----------------------------------------------------------
The grid is a left-to-right strip: the first half of columns is the left
hemisphere, the second half the right hemisphere, ordered by ascending
fsaverage5 *y* (posterior -> anterior), so time animates naturally as the eye
sweeps from the back of the head to the front. Rows are binned by *z*
(inferior -> superior). This is a schematic view, not a geometrically exact
unfolded cortex -- the report says so explicitly.
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache

from .config import BRAIN_GRID_HEIGHT, BRAIN_GRID_WIDTH, N_VERTICES_PER_HEMI
from .roi import _fsaverage5_pial_coordinates


def canonical_vertices() -> tuple[np.ndarray, np.ndarray]:
    """Vertex indices for the grid, ordered ``(left..., right...)``.

    Returns ``(left_idx, right_idx)`` where each entry is a flat array of
    length ``N_VERTICES_PER_HEMI`` already sorted into grid order.
    """

    coords = _fsaverage5_pial_coordinates()
    order = []
    for hemi in ("left", "right"):
        xyz = coords[hemi]
        # Sort by y (posterior->anterior) then z (inferior->superior) so that
        # reshapes to (rows, cols) read left-to-right / bottom-to-top.
        idx = np.lexsort((xyz[:, 2], xyz[:, 1]))
        if idx.size != N_VERTICES_PER_HEMI:
            raise RuntimeError(f"{hemi} hemisphere has {idx.size} vertices")
        order.append(idx.astype(np.int32))
    return order[0], order[1]


@lru_cache(maxsize=8)
def _cell_picks(rows: int, cols: int, n_vertices: int) -> np.ndarray:
    """Index of the vertex nearest each cell centre, ordered row-major.

    ``canonical_vertices`` lays vertices out in grid order, i.e. as a
    row-major image of shape ``(rows, cols)``. A vertex's position in that
    sequence is its normalised grid coordinate, so nearest-neighbour sampling
    reduces to nearest-integer mapping of cell centres. ``np.rint`` (not
    ``floor``) keeps the mapping symmetric about the grid centre.
    """

    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    # Normalised coordinates in [0, 1] for row/col of each cell centre.
    row_pos = ((rr.ravel() + 0.5) / rows) * rows - 0.5
    col_pos = ((cc.ravel() + 0.5) / cols) * cols - 0.5
    flat = np.clip(np.rint(row_pos), 0, rows - 1) * cols + np.clip(np.rint(col_pos), 0, cols - 1)
    # Guarantee uniqueness+safety even if rows/cols change later.
    picks = np.clip(flat.astype(np.int64) % n_vertices, 0, n_vertices - 1)
    return picks


def downsample_to_grid(preds: np.ndarray) -> np.ndarray:
    """Reduce ``(n_trs, 20484)`` predictions to ``(n_trs, h, w)`` float32.

    Values are converted to z-scores against each clip's own distribution so
    the animation highlights *relative* peaks. Absolute magnitudes are not
    comparable across clips and are reported separately.
    """

    arr = np.asarray(preds, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"preds must be 2-D, got shape {arr.shape}")
    n_trs, n_verts = arr.shape
    if n_verts != 2 * N_VERTICES_PER_HEMI:
        raise ValueError(
            f"expected {2 * N_VERTICES_PER_HEMI} vertices, got {n_verts}"
        )

    left_idx, right_idx = canonical_vertices()
    half_w = BRAIN_GRID_WIDTH // 2
    rows = BRAIN_GRID_HEIGHT

    grid = np.zeros((n_trs, rows, BRAIN_GRID_WIDTH), dtype=np.float32)
    for offset, idx in ((0, left_idx), (half_w, right_idx)):
        hemi = arr[:, idx]  # (n_trs, 10242) already in grid order
        # Nearest-neighbour downsample: each output cell takes the vertex that
        # is nearest its cell centre, so no cell is ever empty (10242 is
        # 2 x 3 x 3 x 569, which rules out an exact rectangular tiling).
        picks = _cell_picks(rows, half_w, hemi.shape[1])
        grid[:, :, offset : offset + half_w] = hemi[:, picks].reshape(n_trs, rows, half_w)

    scale = float(grid.std())
    if scale < 1e-9:
        scale = 1.0
    grid = (grid - float(grid.mean())) / scale
    return np.clip(grid, -3.0, 3.0).astype(np.float32)


def encode_brain_grid(
    preds: np.ndarray, *, tr_s: float = 1.0, decimals: int = 1
) -> dict[str, object]:
    """Build the ``brain_grid`` payload consumed by the dashboard.

    ``q`` is a global quantisation value so every frame shares one colour
    scale. With ``decimals=1`` each cell is an integer in ``[-30, 30]``
    (z x 10), which keeps the JSON ~4x smaller than raw floats while staying
    visually identical for a 64x160 heatmap.
    """

    grid = downsample_to_grid(preds)
    q = 10**decimals
    quantised = np.rint(grid * q).astype(np.int16)
    frames = [frame.reshape(-1).tolist() for frame in quantised]

    return {
        "width": BRAIN_GRID_WIDTH,
        "height": BRAIN_GRID_HEIGHT,
        "tr_s": float(tr_s),
        "quantisation": q,
        "z_range": [-3.0, 3.0],
        "layout": "cortical_strip_left_right",
        "n_frames": int(grid.shape[0]),
        "frames": frames,
    }
