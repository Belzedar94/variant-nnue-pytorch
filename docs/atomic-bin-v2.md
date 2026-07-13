# Atomic BIN V2 trainer input

The trainer accepts Atomic BIN V2 only through its canonical
`*.atbin.manifest.json` sidecar. Raw `*.atbin` shards are deliberately rejected.
The native loader delegates manifest parsing, shard snapshots, SHA-256 checks,
record decoding, Atomic legality and byte-exact canonical validation to the
pinned Atomic-Stockfish reader.

The engine submodule is fixed to
`1e64c6f16e8c327be6ee5e3de57ed54d1079f060`. Configuration fails if the
gitlink, checkout, origin URL, schema hashes or `origin/main` ancestry differ,
or if the submodule is dirty.

Because the native loader statically links this pinned Atomic-Stockfish code,
the resulting component is distributed under `GPL-3.0-or-later`. The complete
license is installed alongside the loader from the repository's
`Copying.txt`; the recursively checked-out trainer and engine commits are its
corresponding source.

Legacy Atomic V1 remains available as a headerless 72-byte `.bin` stream. The
singular schema handshake is byte-for-byte unchanged; the plural handshake
advertises both formats.

V2 preserves signed 32-bit scores, unsigned 32-bit sample plies, unsigned
16-bit rule-50 counters, unsigned 32-bit fullmove numbers, Atomic960 castling
origins and score/result semantics relative to the side to move. Scores are
converted explicitly to the trainer's float tensor only when a batch is built.

Trainer-side smart filtering is never applied to V2. Its orthodox geometric
check approximation is retained only for historical Legacy datasets. V2 uses
and reports the authenticated generator policy in the manifest. The recommended
Atomic generation profile is:

```text
Use NNUE=pure
eval_limit=32000
filter_captures=true
filter_promotions=true
filter_checks=false
```

Other manifest policies remain valid for controlled experiments. Training and
validation manifests are rejected when any shards share a SHA-256 digest,
normalized path or filesystem identity, unless the explicit smoke/debug escape
hatch is used.
