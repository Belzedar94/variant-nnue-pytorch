#ifndef ATOMIC_V3_PROVIDER_H_INCLUDED
#define ATOMIC_V3_PROVIDER_H_INCLUDED

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
    #define ATOMIC_V3_PROVIDER_EXPORT __declspec(dllexport)
    #define ATOMIC_V3_PROVIDER_CDECL __cdecl
#else
    #define ATOMIC_V3_PROVIDER_EXPORT
    #define ATOMIC_V3_PROVIDER_CDECL
#endif

extern "C" {

inline constexpr std::uint32_t AtomicV3ProviderAbiVersion = 1;

enum AtomicV3ProviderStatus : std::int32_t {
    ATOMIC_V3_PROVIDER_OK    = 0,
    ATOMIC_V3_PROVIDER_EOF   = 1,
    ATOMIC_V3_PROVIDER_ERROR = 2,
};

struct AtomicV3ManifestInputV1 {
    const char*         pathUtf8;
    std::uint64_t       pathBytes;
    const std::uint8_t* payload;
    std::uint64_t       payloadBytes;
    const char*         sha256Hex;
    std::uint64_t       sha256Bytes;
    std::uint64_t       expectedRecords;
};

// Canonical cursor points to the next raw record. The binding authenticates
// the ordered manifests and all sequence-affecting provider options. Native
// worker and device choices are deliberately absent from that binding.
struct AtomicV3ProviderCursorV1 {
    std::uint32_t abiVersion;
    std::uint32_t structSize;
    std::uint8_t  bindingSha256[32];
    std::uint64_t epoch;
    std::uint32_t manifestIndex;
    std::uint32_t reserved0;
    std::uint64_t recordIndex;
    std::uint64_t acceptedSamples;
    std::uint64_t nextBatchSequence;
    std::uint8_t  eof;
    std::uint8_t  reserved1[7];
};

struct AtomicV3ProviderConfigV1 {
    std::uint32_t                      abiVersion;
    std::uint32_t                      structSize;
    const AtomicV3ManifestInputV1*     manifests;
    std::uint32_t                      manifestCount;
    std::uint32_t                      batchSize;
    std::uint32_t                      randomFenSkipping;
    std::uint32_t                      nativeWorkers;
    std::uint64_t                      seed;
    std::uint8_t                       cyclic;
    std::uint8_t                       reserved[7];
    const AtomicV3ProviderCursorV1*    resumeCursor;
};

struct AtomicV3SparseSliceViewV1 {
    const std::int32_t* indices;
    const float*        values;
    std::uint32_t       width;
    std::uint32_t       reserved;
};

struct AtomicV3PerspectiveViewV1 {
    const std::int64_t*         ownKingSquares;
    AtomicV3SparseSliceViewV1   hm;
    AtomicV3SparseSliceViewV1   capturePair;
    AtomicV3SparseSliceViewV1   kingBlastEp;
    AtomicV3SparseSliceViewV1   blastRing;
};

struct AtomicV3BatchViewV1 {
    std::uint32_t                abiVersion;
    std::uint32_t                structSize;
    std::uint32_t                size;
    std::uint32_t                reserved;
    const float*                 sideToMoveWhite;
    const std::int64_t*          pieceCounts;
    AtomicV3PerspectiveViewV1    white;
    AtomicV3PerspectiveViewV1    black;
    const float*                 outcome;
    const float*                 score;
    const std::int64_t*          bucketIndices;
    AtomicV3ProviderCursorV1     cursorAfter;
};

struct AtomicV3ProviderStreamV1;
struct AtomicV3ProviderBatchV1;

ATOMIC_V3_PROVIDER_EXPORT std::uint32_t ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_abi_version();

ATOMIC_V3_PROVIDER_EXPORT const char* ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_creation_error();

ATOMIC_V3_PROVIDER_EXPORT AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_create(const AtomicV3ProviderConfigV1* config,
                          AtomicV3ProviderStreamV1**       output);

ATOMIC_V3_PROVIDER_EXPORT void ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_destroy(AtomicV3ProviderStreamV1* stream);

ATOMIC_V3_PROVIDER_EXPORT AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_fetch(AtomicV3ProviderStreamV1* stream, AtomicV3ProviderBatchV1** output);

ATOMIC_V3_PROVIDER_EXPORT const char* ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_error(const AtomicV3ProviderStreamV1* stream);

ATOMIC_V3_PROVIDER_EXPORT AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_batch_view(const AtomicV3ProviderBatchV1* batch, AtomicV3BatchViewV1* output);

ATOMIC_V3_PROVIDER_EXPORT void ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_destroy_batch(AtomicV3ProviderBatchV1* batch);

// Commit is intentionally explicit. The executor calls it only after the
// optimizer step that consumed every microbatch since the previous commit.
ATOMIC_V3_PROVIDER_EXPORT AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_commit(AtomicV3ProviderStreamV1* stream);

ATOMIC_V3_PROVIDER_EXPORT AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_committed_cursor(const AtomicV3ProviderStreamV1* stream,
                                    AtomicV3ProviderCursorV1*       output);

}  // extern "C"

#endif  // ATOMIC_V3_PROVIDER_H_INCLUDED
