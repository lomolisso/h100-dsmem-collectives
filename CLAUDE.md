# h100-dsmem-collectives — instructions for Claude Code

## What this repository is
A catalogue of DSMEM (thread-block-cluster) collective algorithms and of the LLM kernels that use them,
plus two **independent** studies on one H100 NVL, each with its own roadmap:

| Study | Brief | Roadmap | Question |
|---|---|---|---|
| `collectives/` | `collectives/CLAUDE.md` | `collectives/ROADMAP.md` (tasks C1–C7) | Are the 29 collective implementations correct, and what are their latency and bandwidth vs cluster size and message size? |
| `roofline/` | `roofline/CLAUDE.md` | `roofline/ROADMAP.md` (tasks R0–R8) | Where does each workload's DSMEM version sit on the H100 roofline relative to its non-DSMEM counterpart? |

`kernels/` (verbatim extractions with docstrings) and `third_party/` (pinned submodules) are reference
material. Neither study modifies them; `roofline/` builds variants of the submodules through patch files only.

The two roadmaps do not depend on each other: either can be run first, and neither waits for the other's
results. They share only the environment (`docs/ENVIRONMENT.md`), the vendored microbench harness
(`docs/REUSE.md`, `common/h100mb/`) and the GPU lock.

## Starting a new conversation
One conversation = one roadmap task. A new conversation needs no further prompt:
1. Decide the study: the one the user names, else the directory the conversation started in, else ask.
2. Open `<study>/ROADMAP.md` → **Task status**. Work on the first row that is not `done`:
   - `in review`: the previous task is committed but not approved. Ask the user for their review, apply
     the requested changes, and push once they approve.
   - `todo` / `in progress`: this is the task. Its prerequisites (the card's "Prerequisites") must hold;
     if they do not, tell the user.
3. Work in **plan mode** first (enter it if the conversation did not start in it). Read the files the task
   card lists, then present a plan: code to write, runs, expected GPU time, result files with sizes, and what
   will be committed. After the user approves, carry it out in auto mode.
4. Always read, in this order: this file, `docs/ENVIRONMENT.md`, `<study>/CLAUDE.md`, `<study>/ROADMAP.md`
   (the whole status table, including "Key results" of earlier rows), the task card, then the documents it
   cites. Read `docs/REUSE.md` before writing any harness code.

## Finishing a task
1. Set the task's row in `<study>/ROADMAP.md` to `in review` and fill in "Key results" (numbers, file paths
   and decisions later tasks need). Update the study's `CLAUDE.md` change log if algorithm or artifact code
   was fixed.
2. Commit everything of the task in one commit on the checked-out branch:
   `<study>/<task id>: <what was done>; <main finding>`. Never commit raw data (`<study>/results/` except the
   files the study's `CLAUDE.md` names), binaries, `.ncu-rep`/`.nsys-rep`, build trees. **Do not push.**
3. After the user has reviewed and explicitly approved: apply fixes, set the row to `done`, commit, push.
   Never force-push. Create, switch or merge branches only when asked.

## Subagents and tools
- Spawning subagents to parallelise work *within* a task is encouraged (independent files, analysis,
  figure code, documentation lookups). Every subagent uses **Claude Opus 5.5 (`model: opus`) with effort `high`**.
- **Only one GPU job at a time**, across this repository and h100-microbench (shared lock). Subagents never
  start GPU runs in parallel. If `tools/run` refuses because the lock is held, wait and retry; never switch
  GPU on your own.
- Web search is allowed for PTX ISA, CUDA, Nsight, PyTorch and artifact documentation.

## Non-negotiable rules
- Environment, GPU selection and safety: `docs/environment` of h100-microbench as summarised in
  `docs/ENVIRONMENT.md` — everything through `tools/run`, **GPU 3**, locked clock for timed and ncu runs,
  never `docker run` by hand or prune, never touch other users' processes.
- Raw data: written crash-safely into `<study>/results/`, never deleted or overwritten, never committed
  (`docs/ENVIRONMENT.md` §5).
- Honesty: report what was measured, including failures, skipped configurations, discarded launches (with
  counts) and partial runs. Never tune parameters, loosen tolerances or drop data to reach an expected value.
