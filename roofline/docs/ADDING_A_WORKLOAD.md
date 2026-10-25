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

Conventions the runner (`roofline/harness/run_manifest.py`, written in roadmap task R2) applies:
- `build.cwd` / `run.cwd` name the submodule; the runner executes them in that variant's git worktree
  `roofline/build/src/<manifest>__<variant>/` with `build.patch` applied there, never in `third_party/`.
  Patch files are named `patches/<manifest>__<variant>.patch` (the initial five use shorter names; keep
  them, the manifest field is what counts).
- `build.cmd` must leave the result in the worktree or under `roofline/build/pyenv/<manifest>__<variant>`
  (no `pip install` into the image venv: containers are `--rm`); `run.cmd` gets that prefix on `PYTHONPATH`.
- Commands that need a wrapper (`# agent script` in the initial manifests) point to a script under
  `roofline/harness/workloads/<manifest>.py` that runs exactly one iteration of the variant on `{shape}`.
- Everything runs through `roofline/tools/run` (`../../docs/ENVIRONMENT.md`).

Checklist before a manifest is "done":
0. Any new Python or system dependency is added to `roofline/docker/Dockerfile` (+ its lock file), the
   image is rebuilt and re-tagged with `roofline/tools/build_image.sh`, and `docs/DESIGN.md` §6 lists it.
1. Both `build` commands succeed from a fresh checkout (plus the referenced patch).
2. The `run` command for every variant produces the output the `validate.reference` can check, and
   validation passes at the stated tolerance.
3. `flops` / `int_ops` evaluates for every shape and agrees with ncu instruction counters within 10 %.
4. `kernels` regexes capture exactly the kernels of the variant under `ncu -k regex:...` (count them).
5. `points.csv` has rows for every (variant, shape) with `scope = operator`, and `scope = layer` if
   `scope_layer` exists.
6. The notebook shows the new markers with an arrow from `non_dsmem` to `dsmem`.
