# Audit — real gaps, prioritised

Written after reading every source file and running the checks below.
Every finding names the file and line that proves it. Nothing here is inferred.

**Baseline measured on this tree (2026-11-05):**

| Check | Command | Result |
|---|---|---|
| Python tests | `python -m pytest tests -q` | **57 passed** in 6.76 s |
| Type check | `npx tsc --noEmit` | exit **0** |
| Lint | `npx eslint src` | exit **0** |
| Root git repo | `git status` | **absent** (`fatal: not a git repository`) |

---

## P0 — fix before anyone else reads this repo

### P0-1. README test count is stale
`README.md:461` says **"53 passed"**. The suite is now **57 passed**.
Trivial to fix, but this document's value is that it never overstates.

### P0-2. `scripts/smoke_test.py` documents a hostname shape that no longer exists
`scripts/smoke_test.py:6` and `:25` use `...--tribe-ad-pretest-healthz.modal.run`.
The real shape (per `frontend/src/lib/api.ts:57`) includes the class segment:
`...--tribe-ad-pretest-tribeadanalyzer-healthz.us-east.modal.run`.
The `endswith("-healthz.modal.run")` check at line 25 still passes, so this is a
**doc bug that hides a URL bug** — a user copying line 6 gets a 404.

### P0-3. No root git repo; `frontend/.git` is a stray nested repo
- Root: `fatal: not a git repository`.
- `frontend/.git` holds one commit (`3314d3f Initial commit from Create Next App`) and **no remote**.
- `frontend/` carries 549 MB `node_modules`, 84 MB `.next`, and 19 MB of `public/` assets, all uncommitted.

Nothing can be pushed until this is resolved.

---

## P1 — correctness / trust issues a careful reviewer will find

### P1-1. `region_mean_z` is 0.0 for *every* region, in *every* clip

This is **not** a stub, but it *looks* exactly like one, and no consumer can tell
the difference.

**Proof.** In all three cache files every one of the 16 region means is `±0.0`:

```
sample_2 ventrolateral: {'left_ventrolateral_anterior': 0.0, 'right_ventrolateral_anterior': 0.0}
sample_2 zeros: all 16 regions (medial/posterolateral/inferior/dorsolateral/...)
```

**Mechanism** (`agent/analysis.py:56-61`, `:143-144`, `:300-302`):

```python
region_mean_z = {name: float(region_z[i].mean())  ...}
```

`region_z` comes from `_zscore_rows`, which subtracts each region's own
time-mean. The mean over time of a row z-scored over time is **0 by
construction** — so this field is mathematically guaranteed to be ~0 for every
region of every clip, forever.

**Why it matters.** `agent/studio.py:363-376` feeds exactly these values into
`networks.static_region_mean_z`, and `frontend/src/app/page.tsx:148-160` renders
them as a greyed **"Language / Broca — clip mean"** bar. That bar is therefore
**always 0%**, in every report, and carries no information. The README and code
comments present it as an honest fallback measurement.

**Fix (pick one):**
- **(a) Cheap and correct:** mean the *raw* region timeseries for the static
  mean instead of the z-scored one. `region_timeseries(...)` already exists and
  is called in `compute_diagnostics` (`agent/analysis.py:238`). Emit
  `region_mean_raw` next to `region_mean_z` and point `studio.py` at the raw
  value.
- **(b) Cheap and honest:** keep `0.0`, rename to `region_mean_z_centered`, and
  render the bar as `n/a` rather than `0%`.

Option (a) is strictly better and is a few lines. Either way, the current state
is a metric that can never vary — the one thing this project is otherwise
careful about.

### P1-2. `hemisphere_split` mixes units with `global_z` in the same object

`agent/studio.py:118-128` puts two different quantities in one record:

| Field | Units | sample_2 range |
|---|---|---|
| `global_z` | z-scored across the clip | −0.803 … 1.326 |
| `hemisphere_split.left` | **raw** mean, constant | 0.0471 … 0.0471 (flat) |
| `hemisphere_split.right` | **raw** mean, constant | 0.0561 … 0.0561 (flat) |

Two problems:
1. **Different scale** in adjacent fields of one object.
2. **It is a constant repeated on every row.** `agent/studio.py:111` reads
   `hemisphere_mean` (a clip-wide scalar pair) and repeats it per TR. The comment
   at `:121-122` is honest about this, and the frontend doesn't plot it
   (`grep` finds no consumer outside `types.ts`) — so today the cost is a
   misleading field name plus ~n_trs wasted objects, not a wrong chart.

**Fix:** rename to `hemisphere_mean_clip` and move it up to the `fmri_curves`
level (once), or fill it with real per-TR means from `hemisphere_timeseries()`
(`agent/analysis.py:81-88`), which already computes them and is currently
discarded.

---

## P2 — design weaknesses, worth stating before a judge finds them

### P2-1. The campaign "score" is a heuristic with one dominant term

`agent/pipeline.py:194-197`:

```
engagement = 100 * clip(dynamic_range / 3.0, 0, 1)   # weight 0.5
focus      = 100 * clip(|laterality_ratio|, 0, 1)    # weight 0.2
continuity = 100 * clip(1 - dead_time_fraction, 0,1) # weight 0.3
score      = 0.5*engagement + 0.2*focus + 0.3*continuity
```

Honest in the payload (`build_report_payload` publishes `method.scoring`
verbatim) and tested for monotonicity (`tests/test_diagnostics.py:122-148`). But:

- **`dynamic_range / 3.0` is a magic constant.** Nothing explains why 3.0
  saturates; a clip with range 3.0 and one with 30.0 score identically.
- **`focus` is `|laterality|`**, i.e. "how asymmetric", with no evidence that
  asymmetry is *good*. `sample_4` has the largest `|laterality|` (0.5261) of the
  three samples and `sample_2` the smallest (0.0871). A reader who notices the
  ranking rewards lopsidedness will ask why.
- **All three terms derive from the same n_trs-length vector**, so they are not
  independent; calling them "components" implies more structure than exists.

Fine for a demo, but the README should say plainly: *"the ranking is a
transparent heuristic, not a validated index"* — the same sentence it already
uses for ROI labels.

### P2-2. The `schaefer` ROI path is a declared dead end
`agent/roi.py:165-179` fetches the atlas, then raises `NotImplementedError`
unconditionally; `get_region_map` catches it and falls back to `mesh`
(`agent/roi.py:188-193`) with only a log line. So `ROI_ATLAS=schaefer` **never
works**. Implement it or delete it — an env var that silently no-ops is worse
than a missing one.

### P2-3. `/campaign` cannot reach the concurrency it claims
`app.py:784-797` runs creatives via `loop.run_in_executor(None, ...)` inside a
**single** GPU container's event loop, calling `self._predict_path` — the same
`self`, same container, same model, same GPU. The docstring says creatives "fan
out to *separate* GPU containers". Three 25 s clips will take ~3×95 s, not ~95 s.
This is the one place the README overstates the architecture.

### ~~P2-4. `MAX_EVENT_CHARS` is declared but never enforced~~ — RETRACTED
I initially flagged this, then checked it and **I was wrong.** `MAX_EVENT_CHARS`
is enforced in `JobLog.append` (`app.py:104`) and re-checked inside the trimming
loop (`app.py:110`), which drops the oldest non-terminal events and sets
`truncated: True`. The guard is real and the design is sound. Recording the
retraction here rather than deleting it, so the rest of this document can be
read as checked rather than asserted.

---

## P3 — polish

- `frontend/README.md` is still untouched `create-next-app` boilerplate; says
  nothing about `NEXT_PUBLIC_TRIBE_API`, the cache, or the stage contract.
- `frontend/AGENTS.md` is auto-generated by `next dev` and will keep reappearing
  as an uncommitted change (its own text says so).
- `agent/studio.py:444-447` re-exports private aliases (`_zones_for_network`, …)
  so the older script/tests keep importing. Harmless, but migrate, don't preserve.
- `scripts/generate_sample_cache.py` is 699 lines — the largest file after
  `app.py`; much of it should move into `agent/studio.py` now that the builder
  lives there.

---

## Suggested order of work

1. **P0-1, P0-2** (2 lines) — restore the docs' credibility.
2. **P0-3** — root repo, park `frontend/.git`, get ignored build dirs right.
3. **P1-1 option (a)** — make the static mean a real number, add a test asserting
   it varies across clips.
4. **P1-2** — stop putting a constant next to a z-score.
5. **P2-3** — either fix `/campaign`'s docstring or make it actually parallel.
6. **P2-1** — one paragraph naming the ranking as a heuristic.

Items 3 and 5 are the ones that change what a sceptical reader concludes about
whether this project measures what it says it measures.
