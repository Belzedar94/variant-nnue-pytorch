# AtomicNNUEV3 bootstrap executor (H9.3l-k Slice2)

This slice freezes the execution and restart contract for the four bootstrap
runs without adding a native provider or a network serializer. Both are later,
explicit injections; there is no Python fallback dataset reader.

The four run IDs are `lambda-0`, `lambda-025`, `lambda-050` and
`lambda-linear-015-050`. Every run has exactly 37 epochs. Historical loader
equivalence treats 20,000,000 training positions and 1,000,000 validation
positions as requested minima and yields complete logical batches: 1,221
training steps (20,004,864 accepted positions) and 62 validation batches
(1,015,808 accepted positions) per epoch. The effective batch is 16,384, formed
from exactly 128 microbatches of 128. Seed 42, random training skip 3, fp32, one
GPU, one torch thread and one provider worker are fixed. Both training and
validation use deterministic random skip 3, matching the archived Atomic
training command; the accepted-position budgets above are measured after that
skip. The linear schedule
includes both advertised endpoints: epoch 0 is 0.15 and epoch 36 is 0.5.

The optimizer is the trainer's single historical Ranger instance. All
parameters except FC2 use learning rate `1.5e-3`; FC2 uses `1.5e-4`. Its betas
are `(0.9, 0.999)`, epsilon `1e-7`, Lookahead alpha `0.5`, `k=6`, no weight
decay and no gradient centralization. `StepLR(gamma=0.987, step_size=1)` advances
once after each completed epoch. Sparse table gradients are coalesced after
each microbatch and densified exactly once immediately before Ranger.
`N_sma_threshold=5` is frozen and checkpoint-validated with the rest of
historical Ranger. A
microbatch mean is multiplied by `microbatch_samples / effective_step_samples`.
The normative provider must deliver every one of the 128 microbatches in an
optimizer step at exactly 128 positions; partial batches fail closed.

The training provider owns one continuous cyclic logical cursor. The executor
does not recreate or reset it at epoch boundaries. Validation uses a fresh
provider restored to the identical authenticated zero cursor every epoch; it
never inherits training state and repeats the same seed-42, skip-3 accepted
prefix.

`prepare_production_run` is the only normative construction path. It calls
`seed_training(42)` before constructing the model or either provider, performs
one complete semantic canary batch per role, restores both cursors byte/field
exactly to canonical zero, and records the model/cursor hashes in the checkpoint
configuration. `create_shared_initial_state` may be called once and its returned
state supplied to all four preparations, guaranteeing identical initial
parameters. Per-microbatch execution performs only shape/sample-count checks;
the expensive board reconstruction never runs in the CUDA hot path.

## Checkpoints

Only `<output-dir>/last.ckpt` is written. A temporary file is flushed and
atomically replaced, so a failed replace leaves the previous checkpoint
intact. The checkpoint binds:

- complete frozen run, optimizer and scheduler configuration;
- bootstrap receipt, selection and semantic-evidence hashes;
- all ordered 29 training and one validation manifest paths, hashes and counts;
- trainer, engine-submodule and bootstrap-verifier commits;
- shared initial-state identity, model, Ranger, StepLR, Python, NumPy, torch
  CPU/CUDA RNG states;
- continuous provider logical cursor and cumulative counters.

Loading is fail-closed on backend/network identity or any config, dataset or
commit mismatch. It also verifies the provider's post-restore cursor exactly
and derives all step, accepted-sample, validation-batch and lambda counters from
the completed epoch count. Checkpoints retain the receipt's
`non-publication-bootstrap` provenance and both publication/release flags as
exactly false. They cannot be used as publication evidence.

Checkpoint and dataset roots are independent, configurable absolute paths and
may reside on different volumes. The expected production output root can
therefore be `D:\NNUE training\Atomic-v2` without moving or rewriting dataset
provenance.

## Dry-run CLI

`train_atomic_v3.py` is deliberately dry-run-only in this slice. It fully
authenticates the bootstrap receipt and its 29+1 manifest/evidence closure,
validates one of the four parameter sets and prints its dataset/commit
pre-binding. The final checkpoint binding additionally receives the shared
initial-state and both provider cursor hashes from preparation. Dry-run does not
allocate a model, create a provider or train:

```powershell
python train_atomic_v3.py `
  --run lambda-linear-015-050 `
  --bootstrap-source C:\path\bootstrap-receipt.json <receipt-sha256> `
  --output-dir "D:\NNUE training\Atomic-v2\lambda-linear-015-050" `
  --trainer-commit <40-digit-commit> `
  --microbatch-size 128 `
  --dry-run
```

Omitting `--dry-run` fails before the dataset is opened. Production execution
is available through the injected `prepare_production_run` → `run_production`
API only; the
native provider and final `.nnue` serializer remain explicit subsequent work.

## Production CLI integration gate

The provider/serializer integration commit must replace the refusal above with
one executable command exposing explicit `--provider-library`,
`--bootstrap-source RECEIPT SHA256`, `--output-dir`, `--trainer-commit`,
`--shared-initial-state`, `--resume` and `--device cuda` arguments plus a final
serializer destination. It must construct both authenticated role factories,
then call only `prepare_production_run` followed by `run_production`, serialize
only the completed epoch-37 checkpoint, and fail closed before training when
the provider DLL, receipt, shared state, commit binding, CUDA device or output
directory is incompatible. Until `atomic_v3.native_provider` and the V3
serializer land together, this slice deliberately does not expose flags that
would describe a launcher it cannot execute.
