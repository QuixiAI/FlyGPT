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
