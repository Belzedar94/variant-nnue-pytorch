# AtomicNNUEV3 bootstrap production executor

This path freezes and executes the restart contract for the four bootstrap
runs. It uses only the authenticated native Atomic BIN V2 provider and the
strict AtomicNNUEV3 serializer; there is no Python fallback dataset reader,
alternate loss, or alternate optimizer.

The four run IDs are `lambda-0`, `lambda-025`, `lambda-050` and
`lambda-linear-015-050`. Every run has exactly 37 epochs. Historical loader
equivalence treats 20,000,000 training positions and 1,000,000 validation
positions as requested minima and yields complete logical batches: 1,221
training steps (20,004,864 accepted positions) and 62 validation batches
(1,015,808 accepted positions) per epoch. The effective batch is 16,384, formed
from exactly 128 microbatches of 128. Seed 42, random skip 3 for both roles,
fp32, one GPU, one torch thread and one provider worker are fixed. The linear
schedule includes both advertised endpoints: epoch 0 is 0.15 and epoch 36 is
0.5.

The optimizer is the trainer's single historical Ranger instance. All
parameters except FC2 use learning rate `1.5e-3`; FC2 uses `1.5e-4`. Its betas
are `(0.9, 0.999)`, epsilon `1e-7`, Lookahead alpha `0.5`, `k=6`,
`N_sma_threshold=5`, no weight decay and no gradient centralization.
`StepLR(gamma=0.987, step_size=1)` advances once after each completed epoch.
Sparse table gradients are coalesced after each microbatch and densified once
immediately before Ranger. A microbatch mean is multiplied by
`microbatch_samples / effective_step_samples`. Every one of the 128
microbatches in an optimizer step must contain exactly 128 positions.

The training provider owns one continuous cyclic logical cursor. The executor
does not recreate or reset it at epoch boundaries. Validation uses a fresh
provider restored to the identical authenticated zero cursor every epoch; it
never inherits training state and repeats the same seed-42, skip-3 accepted
prefix.

`prepare_production_run` is the only normative construction path. It calls
`seed_training(42)` before constructing the model or either provider, performs
one complete semantic canary batch per role, restores both cursors exactly to
canonical zero, and records the model/cursor hashes in the checkpoint
configuration. Per-microbatch execution performs only shape/sample-count
checks; expensive board reconstruction never runs in the CUDA hot path.
The historical physical microbatch remains fixed at 128; progress reporting
does not change the accumulation contract.

## Checkpoints and final artifacts

Only `runs/<run-id>/last.ckpt` is written during training. A temporary file is
flushed and atomically replaced, so a failed replace leaves the previous
checkpoint intact. The checkpoint binds:

- complete frozen run, optimizer and scheduler configuration;
- bootstrap receipt, selection and semantic-evidence hashes;
- all ordered 29 training and one validation manifest paths, hashes and counts;
- trainer, engine-submodule and bootstrap-verifier commits;
- the exact native provider DLL SHA-256 used to construct both streams;
- shared initial-state identity, model, Ranger, StepLR, Python, NumPy, torch
  CPU/CUDA RNG states;
- continuous provider logical cursor and cumulative counters.

Loading is fail-closed on backend/network identity or any config, dataset or
commit mismatch. It verifies the provider's post-restore cursor exactly and
derives all step, accepted-sample, validation-batch and lambda counters from
the completed epoch count. Checkpoints retain the receipt's
`non-publication-bootstrap` provenance and both publication/release flags as
false. They cannot be used as publication evidence.

Only the completed epoch-37 network is serialized, to
`artifacts/atomic-v3-<run-id>-epoch-37.nnue`. A no-overwrite final receipt binds
the bootstrap receipt, provider DLL, trainer and engine commits, shared
initialization, checkpoint, and network hashes. Rolling `status.json` files and
the root `training-summary.json` are machine-readable.

Checkpoint, dataset, and shared-initial-state roots may reside on different
volumes. The expected production output root can therefore be
`D:\NNUE training\Atomic-v2` without moving or rewriting dataset provenance.

Every 32 committed optimizer steps, and again at validation, epoch checkpoint
and completion boundaries, `status.json` is atomically replaced. It records
epoch/global steps, accepted training and validation samples, lambda, learning
rates, the committed native cursor, UTC timestamps, invocation throughput and
ETA. Epoch checkpoint records also include train/validation losses. The hot
path reads only integer counters and the committed native cursor; it never
copies a loss tensor to the host or introduces a per-microbatch CUDA
synchronization.

## Production CLI

`train_atomic_v3.py` authenticates the bootstrap receipt and its 29+1
manifest/evidence closure, loads the provider DLL by an explicit path, and
calls `prepare_production_run` then `run_production`. After epoch 37 it invokes
the strict serializer exactly once. The four-run form is:

The native binding canonicalizes that explicit path and binds it to the
SHA-256 measured both before and after `CDLL` loading. A process-wide lock
serializes the first load and every reuse rehashes the same path. Replacing,
removing, or rebuilding those bytes after load fails closed and requires a
fresh process; the SHA passed into checkpoint and receipt identity must equal
the cached native-module SHA. The historical no-path loader remains only for
legacy callers and is authenticated immediately after `nnue_dataset` imports
its already-selected library; production never uses that weaker entry point.

```powershell
python train_atomic_v3.py `
  --all-runs `
  --bootstrap-source `
    "F:\Atomic-Stockfish-Datasets\atomic-bootstrap-ob68-375m\receipts\atomic-v3-bootstrap-training-receipt-v1.json" `
    2548d1ec912315e80494b3a115e3e1e60376db36e6a2545089d3c3daf8536ad8 `
  --provider-library C:\path\training_data_loader.dll `
  --output-root "D:\NNUE training\Atomic-v2" `
  --shared-initial-state "D:\NNUE training\Atomic-v2\shared-initial-state.pt" `
  --trainer-commit <40-digit-commit> `
  --device cuda:0
```

Use `--run lambda-0` (or another run ID) instead of `--all-runs` for one
network. An existing `last.ckpt` always requires `--resume`; without it the
launcher fails before constructing a model or provider and leaves the
checkpoint byte-for-byte untouched. With `--resume`, an existing checkpoint
is restored, while an absent checkpoint starts a fresh run. A checkpoint made
with a different provider DLL hash is rejected. Already-complete runs are
accepted only when their final receipt, network and checkpoint hashes all
verify. `--dry-run` remains a receipt/config authentication command and never
loads CUDA or the provider.

The launcher creates or loads one atomic `shared-initial-state.pt`, verifies
its tensor-domain SHA-256, and supplies that exact state to every selected run.
Native providers and CUDA allocations are released between runs.

The native provider exposes the exact `ResumableBatchProvider` protocol:
`next_batch` returns `ProviderBatch`, the logical cursor is the last committed
native cursor, and restore creates and verifies a replacement stream before it
destroys the old stream. Validation reset uses that same transactional restore
to the authenticated zero cursor.

## Restore-and-continue GPU canary

Before a 37-epoch launch, the standalone canary exercises the real provider,
one exact 16,384-position forward/backward/optimizer step, rolling checkpoint,
fresh provider/model restore, a second real optimizer step, a second rolling
checkpoint, final serializer, strict checker and strict loader. It reapplies
and verifies the frozen one-thread torch setting before both steps. The final
committed cursor must report 32,768 accepted samples and batch sequence 256.
It cannot call `run_production` and therefore cannot start an epoch:

```powershell
python scripts\canary_atomic_v3_production.py `
  --bootstrap-source `
    "F:\Atomic-Stockfish-Datasets\atomic-bootstrap-ob68-375m\receipts\atomic-v3-bootstrap-training-receipt-v1.json" `
    2548d1ec912315e80494b3a115e3e1e60376db36e6a2545089d3c3daf8536ad8 `
  --provider-library C:\path\training_data_loader.dll `
  --work-dir "D:\NNUE training\Atomic-v2\canary" `
  --shared-initial-state "D:\NNUE training\Atomic-v2\shared-initial-state.pt" `
  --trainer-commit <40-digit-commit> `
  --device cuda:0
```

The work directory must be empty. Its `canary-result.json` contains all input,
restore/final checkpoint, network, restored-cursor and final-cursor hashes and
counters.
