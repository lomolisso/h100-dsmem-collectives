# roofline/ — where the DSMEM workloads sit on the H100 NVL roofline

Design and registry for a roofline study (TPUv1-paper style) with two markers per workload: the DSMEM
version and its non-DSMEM counterpart, measured on one H100 NVL (GPU 3, SM clock locked at 1635 MHz).
Results: `analysis/roofline.ipynb` (executed, figures stored), `results/ceilings.csv` and `results/points.csv`;
`ROADMAP.md` records tasks R0–R8 (run in one conversation) and their key results, `CLAUDE.md` the protocol and the
change log of every fix made to an artifact.

| Path | Content |
|---|---|
| `CLAUDE.md` | Agent brief: one-conversation protocol, server protocol (from `../docs/ENVIRONMENT.md`), workflow and rules |
| `ROADMAP.md` | Task status table and one card per task (R0 environment, R1 ceilings, R2 runner + CREDIT, R3–R7 workloads, R8 final notebook) |
| `docs/DESIGN.md` | Scope decision (operator vs full workload), ceilings, metrics, measurement protocol, figures, CSV schema |
| `docs/WORKLOADS.md` | Every workload pair: entry points in `../third_party/`, the non-DSMEM counterpart that exists, traps, priority |
| `docs/ADDING_A_WORKLOAD.md` | Manifest field reference and checklist for adding a new DSMEM / non-DSMEM pair |
| `workloads/*.yaml` | One manifest per workload (15 initial) — the only thing to add for a new workload |
| `analysis/roofline.ipynb` | Reads every manifest + `results/ceilings.csv` + `results/points.csv`; draws the classic and hierarchical rooflines and per-workload bars |
| `patches/` | `git diff` patches applied to submodules to obtain a variant (referenced by manifests) |
| `harness/` | `run_manifest.py` (build → validate → time (nsys) → ncu → `points.csv`), ceiling microbenchmarks (`ceilings/`), per-workload wrappers (`workloads/`), ncu roofline cross-check |
| `tools/` | `run`: the `h100 run` wrapper with the study's cache/`PYTHONPATH` variables (the image is the shared `~/h100` one) |
| `results/` | git-ignored measurement outputs, except `ceilings.csv` and `points.csv` |

## Adding a workload pair in three steps

1. Copy `workloads/_template.yaml` to `workloads/<name>.yaml`; fill shapes, FLOP / byte formulas, and one
   `variants` entry per version (`role: dsmem`, `role: non_dsmem`, optional `role: control`).
2. Make each variant buildable and runnable from the manifest's `build` / `run` commands (a patch file
   under `patches/` if a submodule must be edited).
3. Run the manifest with `harness/run_manifest.py` (R2); the notebook picks up the new rows automatically.
