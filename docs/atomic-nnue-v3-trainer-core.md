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

Each authenticated read freezes one absolute lexical provenance label before
opening the descriptor and returns that label together with the descriptor
identity and immutable bytes. No `resolve()` occurs after authentication, so a
post-read symlink swap cannot redirect the reported campaign parent or manifest
discovery into another tree. Labels remain non-authoritative; consumers use the
authenticated manifest bytes and hashes.

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

The first 375-million-position pilot has a separate
`atomic-v3-bootstrap-training-receipt-v1` boundary. It authenticates exactly 29
ordered training manifests and one validation manifest, their semantic
validation JSONL and the observed 6-thread/30-thread generation topology. The
receipt is always `provenance_class=non-publication-bootstrap`, with both
`dataset_publication_ready` and `release_candidate_eligible` exactly false.
Those values are passed unchanged to the provider and cannot be promoted by a
caller. The publication loader rejects a bootstrap receipt and the bootstrap
loader rejects a publication receipt.

`atomic_v3.dataset_source` exposes mutually exclusive publication/bootstrap
API objects and CLI arguments. There is no content sniffing, auto-detection or
fallback between them. The bootstrap route reauthenticates its externally
pinned receipt, all 30 ordered manifest byte snapshots and the semantic JSONL
on every provider creation. Manifest and shard path, digest and filesystem
identity reuse across the 29+1 roles is rejected.

The isolated backend remains import-compatible with CPython 3.9 through 3.12.
Its runtime modules avoid PEP 604 unions and dataclass options introduced after
3.9; CI compiles and imports every `atomic_v3` module at each supported minor.

Every batch must contain exactly `piece_count` active HM rows in both
perspectives. Each perspective is decoded back to an absolute-color board and
must contain one king per color, at most 16 pieces per color and at most one
piece per square; the WHITE and BLACK reconstructions must agree exactly.
Outcomes are restricted to `0`, `0.5` or `1`, while scores must be integral
signed-i32 values represented in the float32 training tensor.

## Bootstrap sequential provider

`atomic_v3.native_provider` is the deliberately conservative provider for the
first 29-train/1-validation bootstrap selection. It uses the native C1 Atomic
BIN V2 reader pinned through the `420c9f352` engine submodule. There is no
random-access index, block shuffle, PRP, smart filter, loader auto-detection or
background worker pool. One synchronous native worker walks manifests, shards
and records in authenticated order and emits exact local V3 indices from
`emit_full_refresh` for both king perspectives. The executor owns any effective
batch accumulation; the provider's default transfer microbatch is 128.

Training is one continuously living cyclic stream. Crossing the final training
manifest increments its logical epoch and continues at the first manifest; a
trainer epoch must never reconstruct or rewind it. Validation is non-cyclic,
reports an explicit EOF (including a final partial batch), and may be recreated
from its fixed initial cursor before each validation pass. For the bootstrap
run, the executor takes the fixed one-million-accepted-position prefix from the
dedicated validation manifest.

The only sample selector is deterministic random FEN skipping with the
historical default `random_fen_skipping=3` (one record retained in expectation
out of four) for both roles. Validation requests one million *accepted*
positions, so it normally reads about four million raw records; it does not
shrink to 250,000 positions. Resetting its seed and raw cursor reproduces the
same accepted million exactly. The old trainer's nominal filtered/smart-FEN
path is intentionally not reproduced because its `do_filter()` implementation
returned false for all records. Adding a new position filter here would
silently change the historical sampling policy.

The versioned C ABI owns explicit create, fetch, EOF, error, batch destruction,
stream destruction, commit and committed-cursor operations. Fetch advances a
working cursor; the persisted cursor advances only when the executor calls
`commit()` after a successful optimizer step. The cursor binds the ordered
manifest hashes/counts plus batch size, skip value, seed and cyclic role, so an
exact resume cannot be applied to a different provider contract. Training
reset is rejected; validation reset is explicit.

At each manifest boundary the provider re-reads and hashes the exact immutable
manifest bytes before C1 resolves any shard. C1 then copies, hashes and decodes
only the current shard through its private auto-deleting snapshot. This snapshot
is required security state, not an extra dataset cache: no persistent provider
copy is created, completed/error/destroyed readers close it, and at most one
snapshot is live. A production 12.5-million-record shard occupies
`96 + 12,500,000 * 64 = 800,000,096` bytes (about 763 MiB), which is therefore
the provider's peak temporary-disk requirement independent of whether the
authenticated source lives on another volume.

The controlled optimizer helper clips immediately before the forward and
immediately after the update. Mixed i16/i8 tables, the inward float32-safe PSQT
i32 envelope, all dense signed-i32 biases and both HM/FC0 factor sums are
therefore exportable at every persistent step boundary. FC0 bias and weight
limits apply to the serialized base+factor value, not just to each training
factor independently. FC0 biases and factorized HM PSQT weights are coalesced
and saturated in float64, written back as coordinated float32 operands, and
then checked using the real wire rounding from both float32 and float64 addition
before control returns. This prevents both the FC0 `+2147483648` float32
half-ulp overflow / `-2147483776` float64 overflow and the analogous HM PSQT
opposite-sign cancellation at the 9,600 scale. No signed-i32 cast is attempted
before the range check, and non-finite dense biases or factorized i32 sums fail
closed.

The opt-in `cuda-required` CI job exercises this same clamp on CUDA with both
opposite-sign edge pairs, then runs an Atomic V3 forward/backward and controlled
optimizer step. Setting `ATOMIC_REQUIRE_CUDA_TESTS=1` makes a missing CUDA/CuPy
runtime a session error rather than a skip.

## Bootstrap execution contract

H9.3l-k Slice2 adds the strict four-run Ranger/StepLR executor, exact
microbatch accumulation and rolling atomic checkpoint/resume seam. See
[`atomic-nnue-v3-bootstrap-executor.md`](atomic-nnue-v3-bootstrap-executor.md)
for the frozen parameters and dry-run command. This does not weaken the
bootstrap receipt's non-publication scope.

## Deliberate milestone limits

H9.3l-j proves indexing, factorization, mixed feature forward, loss and a finite
deterministic production-shape CPU step. The step uses state-free SGD only to
exercise sparse table gradients without pre-empting the first-run optimizer
discussion. It is transparent to the caller's CPU RNG and to every already
initialized CUDA RNG, including on exceptions, and it never initializes CUDA
just to run the CPU healthcheck. The caller's thread count and deterministic
algorithm mode, including PyTorch's independent `warn_only` flag, are restored
as well. It is not the production schedule.

The rolling training checkpoint is now implemented, but final network
serialization, the audited native provider, controlled execution evidence and
the first real training run remain later milestones. Nothing in this backend
claims training publication readiness.
