# Decisions log

Spec v4 is frozen (README §21). Deviations and engineering choices made during runs are recorded
here with the date and the run they first applied to, not as spec revisions.

Informal expectations (e.g. "a plain tanh RNN lands around 1.7–2.0") may live here. They never
appear in the scoreboard or in public claims (§2).

| Date | Decision | Applies from | Why |
|---|---|---|---|
| 2026-09-13 | Path gate thresholds: reachable ≥ 0.95, p90 ≤ microsteps × 3 chars | first build_graph | §6 leaves the numbers to engineering; logged in gate.json |
| 2026-09-13 | Degree-preserving plateau: min 10×E swaps, then chunks of E until survival drops < 1e-3/chunk; two seeds must agree within 0.01 | first build_graph | §4 |
| 2026-09-13 | Reference losses use add-1 smoothing with counts from the train split | prepare_data | §2 |
| 2026-09-13 | Trim step (§3.3 D): when the post-trim SCC is short of target, re-trim to target + deficit rather than lowering k (k is already the largest core ≥ target, so lowering it cannot help). Final size may exceed target by a few nodes; logged as final_neurons | first build_graph | spec D as written loops to k=0 on a one-node shortfall |
| 2026-09-13 | MaleCNS is read from the official bulk release (`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`, CC-BY 4.0, md5-verified against the bucket listing), not neuPrint | data/fly/fetch_malecns.py | neuprint.janelia.org is unreachable from this node (TCP timeout, sandbox or not); the bucket is the primary release and needs no token. The verified neuPrint dataset name is `male-cns:v1.0` (was guessed as `male-cns`); kept as the `--neuprint` fallback |
| 2026-09-13 | Region column is `superclass`; region = central_brain iff superclass starts with `cb_` (cb_intrinsic, cb_sensory, cb_motor, cb_endocrine, cb_efferent, cb_sensory_tbc = 37,229 bodies). `vnc_*` -> vnc, `ol_*` -> optic_lobe; ascending/descending, visual_projection/centrifugal, ENS, sensory_ascending -> other (span compartments, excluded); NaN -> unannotated | data/fly/build_edges.py | §3.1: only unambiguous central-brain annotations enter the pool; the full crosstab is printed by build_edges.py |
| 2026-09-13 | Edge table comes from the full `connectome-weights-...-minconf-0.5.feather` (151.9M segment pairs), filtered to annotated bodies by the extractor. On the cb_ pool it differs from the `traced-only` file by 988 of 7.29M edges (0.014%), all touching cb_-annotated bodies whose status is orphan/leaf/anchor rather than traced; these are low-degree and are removed by the dense-core step | first build_graph | User asked for the full graph; the difference is logged rather than silently chosen |
