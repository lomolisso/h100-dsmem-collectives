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

Checklist before a manifest is "done":
1. Both `build` commands succeed from a fresh checkout (plus the referenced patch).
2. The `run` command for every variant produces the output the `validate.reference` can check, and
   validation passes at the stated tolerance.
3. `flops` / `int_ops` evaluates for every shape and agrees with ncu instruction counters within 10 %.
4. `kernels` regexes capture exactly the kernels of the variant under `ncu -k regex:...` (count them).
5. `points.csv` has rows for every (variant, shape) with `scope = operator`, and `scope = layer` if
   `scope_layer` exists.
6. The notebook shows the new markers with an arrow from `non_dsmem` to `dsmem`.
