# roofline/ — where the DSMEM workloads sit on the H100 NVL roofline

Design and registry for a roofline study (TPUv1-paper style) with two markers per workload: the DSMEM
version and its non-DSMEM counterpart. Nothing here has been run; `CLAUDE.md` briefs the agent that
runs it on the shared H100 NVL server.

| Path | Content |
|---|---|
| `CLAUDE.md` | Agent brief with the IMPORTANT server-protocol banner, workflow and rules |
| `docs/DESIGN.md` | Scope decision (operator vs full workload), ceilings, metrics, measurement protocol, figures, CSV schema |
| `docs/WORKLOADS.md` | Every workload pair: entry points in `../third_party/`, the non-DSMEM counterpart that exists, traps, priority |
| `docs/ADDING_A_WORKLOAD.md` | Manifest field reference and checklist for adding a new DSMEM / non-DSMEM pair |
| `workloads/*.yaml` | One manifest per workload (15 initial) — the only thing to add for a new workload |
| `analysis/roofline.ipynb` | Reads every manifest + `results/ceilings.csv` + `results/points.csv`; draws the classic and hierarchical rooflines and per-workload bars |
| `patches/` | `git diff` patches applied to submodules to obtain a variant (referenced by manifests) |
| `results/` | git-ignored measurement outputs |

## Adding a workload pair in three steps

1. Copy `workloads/_template.yaml` to `workloads/<name>.yaml`; fill shapes, FLOP / byte formulas, and one
   `variants` entry per version (`role: dsmem`, `role: non_dsmem`, optional `role: control`).
2. Make each variant buildable and runnable from the manifest's `build` / `run` commands (a patch file
   under `patches/` if a submodule must be edited).
3. Run the manifest with the agent's harness; the notebook picks up the new rows automatically.
