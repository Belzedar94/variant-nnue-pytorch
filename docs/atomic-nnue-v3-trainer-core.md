# AtomicNNUEV3 trainer core (H9.3l-j)

`atomic_v3/` is an isolated trainer backend for the wire-v1 architecture
frozen by Atomic-Stockfish ADR 0004. It does not register a generic feature
name, change the legacy command line or import `atomic_v2`.

The feature transformer has six parameter tensors:

- 1,024 signed-i16 accumulator biases;
- factorized HalfKAv2AtomicHM training rows `[24576, 1032]` plus virtual rows
  `[768, 1032]`;
- CapturePair `[40012, 1024]` with signed-i8 export precision;
- KingBlastEP `[2304, 1024]` with signed-i16 export precision;
- BlastRing `[10240, 1024]` with signed-i8 export precision.

Only HM owns the final eight columns. Its bucket row and virtual row are added
before mixed fake quantization, including PSQT, matching the frozen 12-to-11
export rule. Relation slices have no PSQT parameters. The dense tail is the
eight-stack SFNNv15 graph shared with AtomicNNUEV2, including pairwise
multiplication, squared paths and the FC0 `[30] - [31]` skip.

## Dataset boundary

A campaign filename and a Python object are not authentication roots.
`inspect_campaign_roles` and `create_role_provider` require the path of the
strict H9.3l-a publication receipt plus its exact expected SHA-256. That digest
must arrive out of process from the authenticated controller or content-addressed
store; accepting a digest supplied by arbitrary code already running in the
trainer process does not add authenticity. Accordingly, arbitrary same-process
Python code is explicitly inside this boundary and there is no issuer,
unforgeable-handle or duck-typed capability claim.

Every `create_role_provider` call starts again from the external receipt digest.
It opens stable regular-file descriptors, rejects symlinks and Windows reparse
points in the complete path, and re-reads and re-hashes the receipt, campaign and
every ordered train/validation Atomic BIN V2 manifest immediately before calling
the explicit `provider_factory`. A prior `CampaignInspectionSnapshot` is for
display and diagnostics only and cannot be passed as authority. There is no
loader auto-detection or legacy fallback.

The factory receives the fresh immutable manifest bytes as `manifest_payloads`,
together with paths, expected hashes and record counts. A native adapter MUST
parse those supplied bytes rather than reopen a manifest as unauthenticated
metadata; the accompanying metadata paths are provenance labels, not authority.
It MUST also authenticate each shard named by the manifest immediately at its
own I/O boundary; authenticated manifest bytes do not authenticate a later,
mutable shard path. This milestone adds the strict receipt JSON contract, not a
signing system: key management and signatures remain the controller/CAS layer's
responsibility.

The repository fixture is separately SHA-pinned and retains canonical sample
order. Its train and validation roles are disjoint and are validated
independently before any model allocation. The fixture is forced to LF by
`.gitattributes`, so the authenticated bytes do not change on Windows.

The isolated backend remains import-compatible with CPython 3.9 through 3.12.
Its runtime modules avoid PEP 604 unions and dataclass options introduced after
3.9; CI compiles and imports every `atomic_v3` module at each supported minor.

Every batch must contain exactly `piece_count` active HM rows in both
perspectives. Each perspective is decoded back to an absolute-color board and
must contain one king per color, at most 16 pieces per color and at most one
piece per square; the WHITE and BLACK reconstructions must agree exactly.
Outcomes are restricted to `0`, `0.5` or `1`, while scores must be integral
signed-i32 values represented in the float32 training tensor.

The controlled optimizer helper clips immediately before the forward and
immediately after the update. Mixed i16/i8 tables, the inward float32-safe PSQT
i32 envelope, all dense signed-i32 biases and both HM/FC0 factor sums are
therefore exportable at every persistent step boundary. FC0 bias and weight
limits apply to the serialized base+factor value, not just to each training
factor independently. FC0 biases are coalesced and saturated in float64, written
back as coordinated float32 operands, and then checked using both float32 and
float64 addition before control returns. This prevents both the `+2147483648`
float32 half-ulp overflow and the `-2147483776` value exposed only by float64
coalescing. No signed-i32 cast is attempted before the range check, and
non-finite dense biases fail closed.

## Deliberate milestone limits

H9.3l-j proves indexing, factorization, mixed feature forward, loss and a finite
deterministic production-shape CPU step. The step uses state-free SGD only to
exercise sparse table gradients without pre-empting the first-run optimizer
discussion. It is not the production schedule.

Checkpoint/network serialization, final dependency/environment binding,
controlled execution evidence and the first real training run belong to
H9.3l-k and later milestones. Nothing in this backend claims training
publication readiness.
