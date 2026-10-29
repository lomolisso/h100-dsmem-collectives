# ADDING_A_WORKLOAD.md — manifest reference

A workload is one YAML file in `workloads/`. The harness and the notebook discover manifests by
globbing `workloads/*.yaml` (files starting with `_` are ignored). Field reference:

```yaml
name: <unique snake_case id>            # legend label and `workload` column in points.csv
paper: <paper or origin>
model: <model / application the kernel comes from>
runnable: true | false | conditional    # false -> plotted from `reported`; conditional -> see `notes`
precision: fp16_tensor | bf16_tensor | fp8_tensor | fp32_cuda | int32   # picks the compute ceiling
scope_operator:
  description: <what the region computes, one line>
  shapes:                                # named shapes; every formula below is evaluated per shape
    <shape_id>: {hidden: 4096, seq_len: 4096, ...}
  flops: "<python expression over the shape keys>"         # or int_ops: for integer workloads
  min_hbm_bytes: "<python expression>"                      # compulsory HBM traffic (upper bound of intensity)
  validate:
    reference: <what the outputs are compared against>
    tolerance: <abs/rel tolerance or 'bit-exact'>
  variants:                              # >= 2 entries, exactly one role: dsmem and one role: non_dsmem
    - id: <variant id>
      role: dsmem | non_dsmem | control
      description: <one line>
      build: {cwd: third_party/<repo>, cmd: "<build command>", patch: patches/<file>.patch}   # patch optional
      run:   {cwd: third_party/<repo>, cmd: "<command that runs ONE iteration of the variant on {shape}>"}
      kernels: [<regex of kernel names to capture with ncu -k>]
      cluster_size: <int or null>
scope_layer:                             # optional
  description: <one line>
  extra_kernels: [<kernels added to both variants>]
  run: {cwd: ..., cmd: ...}
measure: {reps: 100, warmup: 20, flush_l2: true}
reported:                                # only when runnable: false
  <variant id>: {throughput: <FLOP/s or op/s>, hbm_bytes: <B>, flops: <FLOP>, source: "<Fig/Table>"}
notes: <traps, caveats>
```

Conventions the runner (`roofline/harness/run_manifest.py`, written in R2; its docstring is the reference) applies:
- `build.cwd` names the submodule. The runner creates one git worktree per (submodule, patch) at the pinned
  commit, `roofline/build/src/<repo>__<patch stem | pin>/`, applies `build.patch` there and runs `build.cmd` in
  it (once per distinct command; a stamp file records it). Variants with the same submodule and patch share the
  worktree. `third_party/` is never modified. `build.gpu: true` when the build needs a visible GPU (e.g.
  ClusterFusion's `setup.py` reads the device architecture); `build.env` adds environment variables.
- `build.cmd` leaves its result in the worktree or under `{pyenv}` = `roofline/build/pyenv/<repo>__<patch>`
  (no `pip install` into the image venv: containers are `--rm`). `run.pythonpath: [...]` (formatted) is put on
  `PYTHONPATH` in front of `roofline/harness`.
- Commands are format strings over the shape keys plus `{P}` (one element of a `cluster_size` list; the runner
  makes one configuration per element), `{warmup}`, `{iters}`, `{worktree}`, `{pyenv}`, `{repo}`, `{harness}`.
  `run.cmd` runs `warmup + iters` iterations; `run.env` sets environment variables.
- **Every iteration starts with the L2-flush kernel `roofline_l2_flush_kernel`** (Python: `flush.flush_l2()`,
  done by `wrapper.py`; C++ artifacts: a patch, e.g. `patches/credit__roofline_hooks.patch`). It flushes L2 and
  delimits the iterations in the nsys kernel trace. Per iteration the runner sums the durations of the kernels
  matching `kernels` (regexes, matched against the demangled names nsys/ncu report).
- Validation: `validate.cmd` (default: `run.cmd` with warmup 0, iters 1) must exit 0, and its output must match
  every regex in `validate.expect`. `skip: {shape_id: reason}` declares configurations that do not exist; they
  are recorded as SKIP with the reason.
- Python artifacts: a wrapper `roofline/harness/workloads/<manifest>.py` built on `roofline/harness/wrapper.py`
  (`setup` / `step` / `check`; `--variant`, `--warmup`, `--iters`, `--validate`, one `--<key>` per shape key).
  `step` runs exactly one iteration of the variant (only its kernels plus what the region needs), `check`
  compares against the reference of `validate`.
- Everything runs through `roofline/tools/run` (`../../docs/ENVIRONMENT.md`): `--no-gpu` for `--stage build`,
  `--gpu 3 --profile --lock-sm-mhz 1635` for validate / time / ncu.

Checklist before a manifest is "done":
0. Any new Python or system dependency is added to the shared image (`~/h100/docker/requirements.txt` →
   `h100 image lock` → `h100 image build`, or the Dockerfile), with the user's approval, and `docs/DESIGN.md` §6
   lists it.
1. Both `build` commands succeed from a fresh checkout (plus the referenced patch).
2. The `run` command for every variant produces the output the `validate.reference` can check, and
   validation passes at the stated tolerance.
3. `flops` / `int_ops` evaluates for every shape and agrees with ncu instruction counters within 10 %.
4. `kernels` regexes capture exactly the kernels of the variant under `ncu -k regex:...` (count them).
5. `points.csv` has rows for every (variant, shape) with `scope = operator`, and `scope = layer` if
   `scope_layer` exists.
6. The notebook shows the new markers with an arrow from `non_dsmem` to `dsmem`.
