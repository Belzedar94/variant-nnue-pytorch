# AtomicNNUEV3 trainer performance incident

## Summary

The first production bootstrap run was stopped after the trainer proved about
41 times slower than one historically contended legacy run, and about 49 to 65
times slower than the normal three-to-four-minute legacy epoch observed on the
same machine.  AtomicNNUEV3 is larger than the legacy model, but that does not
explain the regression.  The primary failure was architectural: Atomic V3
became a second training runtime instead of a feature/model adapter over the
proven trainer path.

No network was released.  The 375M-position dataset, manifests, validation
receipt and shared initial state remain intact.  The interrupted epoch-one
checkpoint is retained only as diagnostic evidence and must not be resumed by
the repaired trainer.

## Timeline and measured impact

- Production started at 2026-07-17 02:19:56 CEST.
- The first durable epoch checkpoint was written at 05:42:29.
- Epoch two reached validation at 08:53:31, but was stopped before a second
  durable checkpoint was written.
- The status file reported 40,009,728 accepted training samples, 2,442
  optimizer steps, 23,614 seconds elapsed and 0.1034 optimizer steps per
  second: approximately 1,694 accepted positions per second.
- A 20,004,864-position epoch therefore required about 197 minutes of training
  before validation and checkpointing.
- Historical logs for four concurrent legacy runs show a median of about 289
  seconds per epoch per run.  The normal uncontended observation is three to
  four minutes.
- Four serial 37-epoch Atomic V3 runs would have taken roughly twenty days.

## Technical causes

1. The runtime fixed the physical batch to 128 and accumulated 128 backwards
   for every 16,384-position optimizer step.  The proven CUDA path uses a
   physical batch of 16,384.
2. The audit-grade Atomic BIN V2 reader was used in the epoch hot path.  Every
   raw record paid descriptor verification, semantic decode, FEN/Position
   reconstruction, legality checks and a byte-exact re-encode before the
   deterministic skip discarded three records out of four.
3. Generic sparse embeddings materialized gathered feature rows and produced
   sparse gradients.  The proven trainer uses a fused CUDA sparse feature
   transformer with direct dense gradients.
4. Each 128-position batch crossed native/NumPy/Torch/CUDA boundaries through
   many separately allocated and pinned tensors, without useful prefetch.
5. CUDA tensor predicates in the dense and loss paths forced host
   synchronization per microbatch.
6. Training fake quantization promoted gathered weights to float64 on an RTX
   3080, where FP64 throughput is deliberately low.
7. Sparse gradients were repeatedly coalesced after every microbatch and then
   densified for Ranger.
8. Validation incorrectly used random skip 3, multiplying its raw reader work
   by about four.

## Process causes

- Correctness, provenance and fail-closed resume behavior were frozen before
  throughput viability was measured.
- A CPU-safe batch of 128 was incorrectly documented and tested as the
  historical GPU batch.
- Unit tests and the real-GPU canary proved forward/backward, cursor, resume
  and serialization, but imposed no samples-per-second or epoch-ETA gate.
- The production status already exposed an impossible ETA, but no circuit
  breaker converted it into a failure.  The user, not automation, detected the
  incident.

## Repair boundary

The proven legacy/current Stockfish trainer is the execution platform.  Atomic
V3 supplies only:

- authenticated Atomic BIN V2 ingestion;
- HM, CapturePair, KingBlastEP and BlastRing feature extraction;
- Atomic mixed-width feature-transformer rules;
- the Atomic loss and four lambda schedules;
- the AtomicNNUEV3 serializer and engine compatibility checks.

The production hot path must use the proven 16,384-position CUDA batching,
fused sparse accumulation, FP32 quantization-aware training, buffered data
loading and asynchronous pinned transfer.  Receipt, manifest and shard hashes
are authenticated at setup boundaries, never once per record.

## Mandatory gates

Before another multi-epoch campaign:

1. Compare accepted record identity and feature indices against the frozen
   scalar/provider oracle.
2. Verify bit-exact shared initialization and equal starting state for all four
   lambda runs.
3. Verify CPU reference versus CUDA fused forward and gradients, including HM
   base/virtual rows and mixed i16/i8 quantization boundaries.
4. Warm up and measure at least ten production-shaped optimizer steps.  Report
   loader, transfer, forward, backward, optimizer and clipping time, accepted
   positions per second, VRAM and projected epoch duration.
5. Target at most five minutes per epoch.  Automatically reject a projected
   epoch strictly above ten minutes.
6. Complete one real epoch, checkpoint, resume, serialize, strict-check and
   load the network in Atomic-Stockfish before launching 37 epochs.
7. Keep a rolling production circuit breaker; lack of stderr is never a health
   criterion.

The first repaired campaign starts fresh from the authenticated shared initial
state.  The old optimizer, scheduler, RNG and microbatch cursor states are not
scientifically compatible with the repaired runtime.
