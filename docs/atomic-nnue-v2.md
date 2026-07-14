# AtomicNNUEV2 trainer

`atomic_v2/` is an additive trainer backend. The historical root `model.py`,
`train.py`, and `serialize.py` remain the LegacyAtomicV1 path.

The backend is pinned to the engine contract from Atomic-Stockfish commit
`0818886d77328fe25850a3187e6460adaa980316`. Its vendored JSON has Git blob
`8a8003cbf5f7b97b50470de898ae16074aad7bca` and raw SHA-256
`70ea8da4cdd2f209fafc25f9419d3b3f226711b00701bf2fc02587dba65b82d7`.
The SFNNv15 topology and quantization behavior are adapted by intent from
official `nnue-pytorch` commit
`b8512291deb4cd18afa67003bb6bc53dd522cbf0`; orthodox `HalfKAv2_hm` and
`FullThreats` are deliberately absent.

The physical graph is HalfKAv2Atomic (45,056 real features), a 1,024-value
accumulator per perspective, pairwise multiplication to 1,024 transformed
values, and the SFNNv15 `1024 -> 32`, `64 -> 32`, `128 -> 1` layer stacks.
Both squared and linear clipped activation paths and the `fc0[-2]-fc0[-1]`
short skip are mandatory even though the legacy wire hash cannot distinguish
all of those physical details.

Use `train_atomic_v2.py` with separate atomic-bin-v2 train and validation
manifests. The V2 entrypoint requires and preflights the canonical
`.atbin.manifest.json` suffix, format and both schema hashes before invoking the
dual-format native loader, so a Legacy V1 or mixed train/validation pair cannot
silently bypass overlap validation. It then uses the loader's plural capability
handshake and ten-tensor batch ABI, seeds before model construction, and starts
V2 from scratch.

The default batch size is 128 on CPU and 16,384 on CUDA. Validation is a
sample-weighted aggregate up to the explicit `--validation-samples` budget
(default 1,000,000), or to dataset EOF when shorter. The budget, actual sample
count and loss are persisted in the checkpoint and JSON summary.

`serialize_atomic_v2.py` converts only tagged V2 checkpoints and V2 network
files. Both entrypoints reject implicit overwrites; neither accepts or silently
upgrades a V1 network. The writer validates every physical tensor shape before
emitting header bytes and is pinned to the engine's nonzero diagnostic fixture
SHA-256 `4DEB05CFF79B5D5EBA51C560F64ED24224671C188B6C5DB27521033E587C87C6`.
That fixture diagnoses both main feature dimensions and PSQT buckets 0 through
6, activates ten paired FT dimensions across SIMD and sparse-group boundaries,
and carries nonzero weights through every dense layer and both squared/linear
activation paths. Its eight layer stacks have distinct raw/public outputs from
`10587/661` through `11112/694`, detecting stack permutation or fixed-bucket
errors. PSQT bucket 7 remains zero.

PSQT output follows the engine's signed integer contract: the perspective
difference is divided by two with truncation toward zero. The training graph
uses a straight-through estimator, so odd raw differences of `+1` and `-1`
both produce zero rather than unrepresentable half-unit evaluations while
retaining their gradient.

The current strong V1 network remains appropriate as the generator's `pure`
teacher. That does not make it a V2 initialization checkpoint.
