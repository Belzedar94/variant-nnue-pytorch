#include "atomic_v3_provider.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../external/Atomic-Stockfish/src/atomic_init.h"
#include "../external/Atomic-Stockfish/src/data/atomic_bin_v2_manifest.h"
#include "../external/Atomic-Stockfish/src/data/atomic_bin_v2_reader.h"
#include "../external/Atomic-Stockfish/src/data/sha256.h"
#include "../external/Atomic-Stockfish/src/nnue/atomic_v3/full_refresh.h"

namespace {

namespace AtomicData = Stockfish::Data;
namespace AtomicV3   = Stockfish::Eval::NNUE::AtomicV3;

constexpr std::uint64_t MaximumManifestBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::uint32_t MaximumBatchSize      = 1U << 20;
constexpr std::uint32_t MaximumManifests      = 100000;

thread_local std::array<char, 1024> CreationError{};

void set_creation_error(std::string_view message) noexcept {
    const std::size_t count = std::min(message.size(), CreationError.size() - 1);
    std::memcpy(CreationError.data(), message.data(), count);
    CreationError[count] = '\0';
}

bool lower_sha256(std::string_view value) {
    if (value.size() != 64)
        return false;
    return std::all_of(value.begin(), value.end(), [](char character) {
        return (character >= '0' && character <= '9')
            || (character >= 'a' && character <= 'f');
    });
}

void update_u32(AtomicData::Sha256& hash, std::uint32_t value) {
    std::array<std::uint8_t, 4> bytes{};
    for (unsigned index = 0; index < bytes.size(); ++index)
        bytes[index] = std::uint8_t(value >> (8 * index));
    hash.update(bytes.data(), bytes.size());
}

void update_u64(AtomicData::Sha256& hash, std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (unsigned index = 0; index < bytes.size(); ++index)
        bytes[index] = std::uint8_t(value >> (8 * index));
    hash.update(bytes.data(), bytes.size());
}

std::string read_manifest_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::invalid_argument("cannot open Atomic V3 manifest " + path.u8string());
    const std::streamoff end = input.tellg();
    if (end <= 0 || std::uint64_t(end) > MaximumManifestBytes)
        throw std::invalid_argument("Atomic V3 manifest has an invalid byte count");
    input.seekg(0);
    std::string bytes(std::size_t(end), '\0');
    input.read(bytes.data(), end);
    if (!input)
        throw std::invalid_argument("cannot read complete Atomic V3 manifest");
    return bytes;
}

struct ManifestSpec {
    std::filesystem::path path;
    std::string           payload;
    std::string           sha256;
    std::uint64_t         records = 0;
    std::uint64_t         firstRecord = 0;
};

std::unique_ptr<AtomicData::AtomicBinV2DatasetReader>
open_exact_manifest(const ManifestSpec& spec) {
    // The Python trust seam supplies immutable bytes. Re-read the named file at
    // this native boundary and require byte identity before allowing C1 to
    // resolve and authenticate its shard paths.
    const std::string current = read_manifest_bytes(spec.path);
    AtomicData::Sha256 digest;
    digest.update(current);
    if (digest.hex_digest() != spec.sha256 || current != spec.payload)
        throw std::invalid_argument("Atomic V3 manifest changed after authentication");

    std::unique_ptr<AtomicData::AtomicBinV2DatasetReader> reader;
    AtomicData::DataResult opened =
      AtomicData::AtomicBinV2DatasetReader::open(spec.path, reader);
    if (!opened)
        throw std::invalid_argument("cannot open Atomic V3 manifest through C1: " + opened.message);

    std::string canonical;
    AtomicData::DataResult rendered =
      AtomicData::render_atomic_bin_v2_manifest(reader->manifest(), canonical);
    if (!rendered || canonical != spec.payload)
        throw std::invalid_argument("C1 manifest metadata differs from authenticated bytes");
    if (reader->manifest().records != spec.records)
        throw std::invalid_argument("C1 manifest record count differs from provider contract");
    return reader;
}

std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

bool keep_record(std::uint64_t seed,
                 std::uint64_t epoch,
                 std::uint64_t roleRecord,
                 std::uint32_t randomFenSkipping) noexcept {
    if (randomFenSkipping == 0)
        return true;
    const std::uint64_t bound = std::uint64_t(randomFenSkipping) + 1;
    const std::uint64_t threshold = (0ULL - bound) % bound;
    std::uint64_t lane = 0;
    for (;;)
    {
        const std::uint64_t counter = seed
                                    ^ (epoch * 0xD1B54A32D192ED03ULL)
                                    ^ (roleRecord * 0x94D049BB133111EBULL)
                                    ^ (lane++ * 0x9E3779B97F4A7C15ULL);
        const std::uint64_t value = splitmix64(counter);
        if (value >= threshold)
            return value % bound == 0;
    }
}

Stockfish::Piece piece_from_wire(Stockfish::u8 value) {
    using namespace AtomicData;
    switch (value)
    {
    case ATOMIC_BIN_V2_EMPTY :
        return Stockfish::NO_PIECE;
    case ATOMIC_BIN_V2_WHITE_PAWN :
        return Stockfish::W_PAWN;
    case ATOMIC_BIN_V2_WHITE_KNIGHT :
        return Stockfish::W_KNIGHT;
    case ATOMIC_BIN_V2_WHITE_BISHOP :
        return Stockfish::W_BISHOP;
    case ATOMIC_BIN_V2_WHITE_ROOK :
        return Stockfish::W_ROOK;
    case ATOMIC_BIN_V2_WHITE_QUEEN :
        return Stockfish::W_QUEEN;
    case ATOMIC_BIN_V2_WHITE_KING :
        return Stockfish::W_KING;
    case ATOMIC_BIN_V2_BLACK_PAWN :
        return Stockfish::B_PAWN;
    case ATOMIC_BIN_V2_BLACK_KNIGHT :
        return Stockfish::B_KNIGHT;
    case ATOMIC_BIN_V2_BLACK_BISHOP :
        return Stockfish::B_BISHOP;
    case ATOMIC_BIN_V2_BLACK_ROOK :
        return Stockfish::B_ROOK;
    case ATOMIC_BIN_V2_BLACK_QUEEN :
        return Stockfish::B_QUEEN;
    case ATOMIC_BIN_V2_BLACK_KING :
        return Stockfish::B_KING;
    default :
        throw std::invalid_argument("Atomic V3 record contains an unknown piece code");
    }
}

struct PerspectiveRows {
    std::int64_t              ownKing = 0;
    std::vector<std::int32_t> hm;
    std::vector<std::int32_t> capturePair;
    std::vector<std::int32_t> kingBlastEp;
    std::vector<std::int32_t> blastRing;
};

struct PackedSample {
    float         sideToMoveWhite = 0.0F;
    std::int64_t  pieceCount = 0;
    PerspectiveRows white;
    PerspectiveRows black;
    float         outcome = 0.0F;
    float         score = 0.0F;
    std::int64_t  bucket = 0;
};

PerspectiveRows rows_from_emission(const AtomicV3::FullRefreshEmission& emission) {
    PerspectiveRows rows;
    rows.ownKing = std::int64_t(emission.hm.orientation.ownKing);
    rows.hm.reserve(emission.hm.size);
    rows.capturePair.reserve(emission.capturePairs.size);
    rows.kingBlastEp.reserve(emission.kingBlastEp.size);
    rows.blastRing.reserve(emission.blastRing.size);
    for (std::uint32_t index = 0; index < emission.hm.size; ++index)
        rows.hm.push_back(std::int32_t(emission.hm.features[index].trainingIndex));
    for (std::uint32_t index = 0; index < emission.capturePairs.size; ++index)
        rows.capturePair.push_back(std::int32_t(emission.capturePairs.features[index].localIndex));
    for (std::uint32_t index = 0; index < emission.kingBlastEp.size; ++index)
        rows.kingBlastEp.push_back(std::int32_t(emission.kingBlastEp.features[index].localIndex));
    for (std::uint32_t index = 0; index < emission.blastRing.size; ++index)
        rows.blastRing.push_back(std::int32_t(emission.blastRing.features[index].localIndex));
    if (!std::is_sorted(rows.capturePair.begin(), rows.capturePair.end())
        || !std::is_sorted(rows.kingBlastEp.begin(), rows.kingBlastEp.end())
        || !std::is_sorted(rows.blastRing.begin(), rows.blastRing.end()))
        throw std::logic_error("Atomic V3 relation oracle returned noncanonical order");
    return rows;
}

PackedSample pack_sample(const AtomicData::AtomicBinV2DecodedRecord& decoded) {
    const auto& fields = decoded.fields;
    AtomicV3::CapturePairSnapshot snapshot{};
    snapshot.sideToMove = fields.position.sideToMove == AtomicData::ATOMIC_BIN_V2_WHITE_TO_MOVE
                            ? Stockfish::WHITE
                            : Stockfish::BLACK;
    snapshot.epSquare = fields.position.enPassantSquare == AtomicData::AtomicBinV2NoSquare
                        ? Stockfish::SQ_NONE
                        : Stockfish::Square(fields.position.enPassantSquare);
    for (std::size_t square = 0; square < fields.position.board.size(); ++square)
        snapshot.board[square] = piece_from_wire(fields.position.board[square]);

    AtomicV3::FullRefreshEmission whiteEmission{};
    AtomicV3::FullRefreshEmission blackEmission{};
    const auto whiteError = AtomicV3::emit_full_refresh(snapshot, Stockfish::WHITE, whiteEmission);
    const auto blackError = AtomicV3::emit_full_refresh(snapshot, Stockfish::BLACK, blackEmission);
    if (whiteError != AtomicV3::CapturePairError::None
        || blackError != AtomicV3::CapturePairError::None)
        throw std::invalid_argument(
          std::string("Atomic V3 full refresh rejected record: white=")
          + AtomicV3::full_refresh_error_message(whiteError) + " black="
          + AtomicV3::full_refresh_error_message(blackError));
    if (whiteEmission.hm.size != blackEmission.hm.size
        || whiteEmission.hm.networkBucket != blackEmission.hm.networkBucket)
        throw std::logic_error("Atomic V3 perspectives disagree on piece count or network bucket");

    const float score = static_cast<float>(fields.score);
    if (static_cast<std::int64_t>(score) != std::int64_t(fields.score))
        throw std::invalid_argument("Atomic V3 score is not exactly representable as float32");

    PackedSample sample;
    sample.sideToMoveWhite = snapshot.sideToMove == Stockfish::WHITE ? 1.0F : 0.0F;
    sample.pieceCount = std::int64_t(whiteEmission.hm.size);
    sample.white       = rows_from_emission(whiteEmission);
    sample.black       = rows_from_emission(blackEmission);
    sample.outcome     = (float(fields.result) + 1.0F) / 2.0F;
    sample.score       = score;
    sample.bucket      = std::int64_t(whiteEmission.hm.networkBucket);
    return sample;
}

struct SparseStorage {
    std::uint32_t            width = 1;
    std::vector<std::int32_t> indices;
    std::vector<float>        values;
};

struct PerspectiveStorage {
    std::vector<std::int64_t> ownKingSquares;
    SparseStorage             hm;
    SparseStorage             capturePair;
    SparseStorage             kingBlastEp;
    SparseStorage             blastRing;
};

template<typename Selector>
SparseStorage pack_sparse(const std::vector<PackedSample>& samples, Selector selector) {
    SparseStorage storage;
    for (const PackedSample& sample : samples)
        storage.width = std::max<std::uint32_t>(
          storage.width, std::uint32_t(selector(sample).size()));
    if (samples.size() > std::numeric_limits<std::size_t>::max() / storage.width)
        throw std::length_error("Atomic V3 sparse batch size overflow");
    const std::size_t cells = samples.size() * storage.width;
    storage.indices.assign(cells, -1);
    storage.values.assign(cells, 0.0F);
    for (std::size_t row = 0; row < samples.size(); ++row)
    {
        const auto& values = selector(samples[row]);
        for (std::size_t column = 0; column < values.size(); ++column)
        {
            const std::size_t offset = row * storage.width + column;
            storage.indices[offset] = values[column];
            storage.values[offset]  = 1.0F;
        }
    }
    return storage;
}

PerspectiveStorage pack_perspective(const std::vector<PackedSample>& samples, bool white) {
    PerspectiveStorage storage;
    storage.ownKingSquares.reserve(samples.size());
    for (const PackedSample& sample : samples)
        storage.ownKingSquares.push_back(white ? sample.white.ownKing : sample.black.ownKing);
    const auto perspective = [white](const PackedSample& sample) -> const PerspectiveRows& {
        return white ? sample.white : sample.black;
    };
    storage.hm = pack_sparse(samples, [&](const PackedSample& sample) -> const auto& {
        return perspective(sample).hm;
    });
    storage.capturePair = pack_sparse(samples, [&](const PackedSample& sample) -> const auto& {
        return perspective(sample).capturePair;
    });
    storage.kingBlastEp = pack_sparse(samples, [&](const PackedSample& sample) -> const auto& {
        return perspective(sample).kingBlastEp;
    });
    storage.blastRing = pack_sparse(samples, [&](const PackedSample& sample) -> const auto& {
        return perspective(sample).blastRing;
    });
    return storage;
}

struct BatchImpl {
    std::vector<float>        sideToMoveWhite;
    std::vector<std::int64_t> pieceCounts;
    PerspectiveStorage        white;
    PerspectiveStorage        black;
    std::vector<float>        outcome;
    std::vector<float>        score;
    std::vector<std::int64_t> bucketIndices;
    AtomicV3ProviderCursorV1  cursorAfter{};

    BatchImpl(std::vector<PackedSample> samples, const AtomicV3ProviderCursorV1& cursor) :
        white(pack_perspective(samples, true)),
        black(pack_perspective(samples, false)),
        cursorAfter(cursor) {
        sideToMoveWhite.reserve(samples.size());
        pieceCounts.reserve(samples.size());
        outcome.reserve(samples.size());
        score.reserve(samples.size());
        bucketIndices.reserve(samples.size());
        for (const PackedSample& sample : samples)
        {
            sideToMoveWhite.push_back(sample.sideToMoveWhite);
            pieceCounts.push_back(sample.pieceCount);
            outcome.push_back(sample.outcome);
            score.push_back(sample.score);
            bucketIndices.push_back(sample.bucket);
        }
    }

    static AtomicV3SparseSliceViewV1 view(const SparseStorage& storage) {
        return {storage.indices.data(), storage.values.data(), storage.width, 0};
    }

    static AtomicV3PerspectiveViewV1 view(const PerspectiveStorage& storage) {
        return {storage.ownKingSquares.data(), view(storage.hm), view(storage.capturePair),
                view(storage.kingBlastEp), view(storage.blastRing)};
    }

    AtomicV3BatchViewV1 view() const {
        AtomicV3BatchViewV1 result{};
        result.abiVersion      = AtomicV3ProviderAbiVersion;
        result.structSize      = sizeof(result);
        result.size            = std::uint32_t(sideToMoveWhite.size());
        result.sideToMoveWhite = sideToMoveWhite.data();
        result.pieceCounts     = pieceCounts.data();
        result.white           = view(white);
        result.black           = view(black);
        result.outcome         = outcome.data();
        result.score           = score.data();
        result.bucketIndices   = bucketIndices.data();
        result.cursorAfter     = cursorAfter;
        return result;
    }
};

class StreamImpl {
   public:
    explicit StreamImpl(const AtomicV3ProviderConfigV1& config) :
        batchSize(config.batchSize),
        randomFenSkipping(config.randomFenSkipping),
        seed(config.seed),
        cyclic(config.cyclic != 0) {
        Stockfish::initialize_atomic_core();
        load_manifests(config);
        binding = compute_binding();
        initialize_cursor(config.resumeCursor);
    }

    std::unique_ptr<BatchImpl> next() {
        if (poisoned)
            throw std::runtime_error("Atomic V3 provider is poisoned after an earlier error");
        if (working.eof)
            return nullptr;

        std::vector<PackedSample> samples;
        samples.reserve(batchSize);
        while (samples.size() < batchSize)
        {
            AtomicData::AtomicBinV2DecodedRecord decoded{};
            std::uint64_t sourceEpoch = 0;
            std::uint64_t sourceRoleRecord = 0;
            if (!next_raw(decoded, sourceEpoch, sourceRoleRecord))
                break;
            if (!keep_record(seed, sourceEpoch, sourceRoleRecord, randomFenSkipping))
                continue;
            samples.push_back(pack_sample(decoded));
            ++working.acceptedSamples;
        }
        if (samples.empty())
            return nullptr;
        ++working.nextBatchSequence;
        lastDelivered = working;
        hasDelivered  = true;
        return std::make_unique<BatchImpl>(std::move(samples), working);
    }

    void commit() {
        if (!hasDelivered)
            throw std::logic_error("Atomic V3 provider has no delivered batch to commit");
        committed = lastDelivered;
        hasDelivered = false;
    }

    const AtomicV3ProviderCursorV1& committed_cursor() const noexcept { return committed; }

    void set_error(std::string message) noexcept {
        error = std::move(message);
        poisoned = true;
        reader.reset();
    }
    const std::string& last_error() const noexcept { return error; }

   private:
    void load_manifests(const AtomicV3ProviderConfigV1& config) {
        if (config.manifestCount == 0 || config.manifestCount > MaximumManifests)
            throw std::invalid_argument("Atomic V3 provider manifest count is outside bounds");
        if (config.manifests == nullptr)
            throw std::invalid_argument("Atomic V3 provider manifest array is null");
        specs.reserve(config.manifestCount);
        std::uint64_t first = 0;
        for (std::uint32_t index = 0; index < config.manifestCount; ++index)
        {
            const AtomicV3ManifestInputV1& input = config.manifests[index];
            if (input.pathUtf8 == nullptr || input.payload == nullptr || input.sha256Hex == nullptr
                || input.pathBytes == 0 || input.pathBytes > 32768
                || input.payloadBytes == 0 || input.payloadBytes > MaximumManifestBytes
                || input.sha256Bytes != 64
                || input.expectedRecords == 0)
                throw std::invalid_argument("Atomic V3 manifest descriptor is invalid");
            const std::string pathBytes(input.pathUtf8, std::size_t(input.pathBytes));
            if (pathBytes.find('\0') != std::string::npos)
                throw std::invalid_argument("Atomic V3 manifest path contains NUL");
            const std::string expectedHash(input.sha256Hex, std::size_t(input.sha256Bytes));
            if (!lower_sha256(expectedHash))
                throw std::invalid_argument("Atomic V3 manifest SHA-256 is not lowercase hex");
            const std::string payload(reinterpret_cast<const char*>(input.payload),
                                      std::size_t(input.payloadBytes));
            AtomicData::Sha256 payloadHash;
            payloadHash.update(payload);
            if (payloadHash.hex_digest() != expectedHash)
                throw std::invalid_argument("Atomic V3 supplied manifest bytes fail SHA-256");

            ManifestSpec spec;
            spec.path        = std::filesystem::u8path(pathBytes);
            spec.payload     = payload;
            spec.sha256      = expectedHash;
            spec.records     = input.expectedRecords;
            spec.firstRecord = first;
            if (first > std::numeric_limits<std::uint64_t>::max() - spec.records)
                throw std::overflow_error("Atomic V3 aggregate record count overflows uint64");
            first += spec.records;

            AtomicData::AtomicBinV2Manifest trusted{};
            AtomicData::DataResult parsed =
              AtomicData::parse_atomic_bin_v2_manifest(spec.payload, spec.path, trusted);
            if (!parsed)
                throw std::invalid_argument("cannot parse supplied Atomic V3 manifest: "
                                            + parsed.message);
            if (trusted.records != spec.records)
                throw std::invalid_argument("supplied Atomic V3 manifest record count differs");
            // Validate every manifest at native creation. Shards remain lazy:
            // C1 stages and hashes the current shard before exposing its first
            // record, keeping the 29-manifest bootstrap from allocating 29
            // full private snapshots simultaneously.
            (void) open_exact_manifest(spec);
            specs.push_back(std::move(spec));
        }
        totalRecords = first;
    }

    std::array<std::uint8_t, 32> compute_binding() const {
        AtomicData::Sha256 hash;
        // Include the terminating NUL without relying on a hand-counted byte
        // literal. This is part of the persisted cursor binding.
        static constexpr char Domain[] = "atomic-v3-sequential-provider-v1";
        hash.update(Domain, sizeof(Domain));
        update_u32(hash, std::uint32_t(specs.size()));
        for (const ManifestSpec& spec : specs)
        {
            hash.update(spec.sha256);
            update_u64(hash, spec.records);
        }
        update_u32(hash, batchSize);
        update_u32(hash, randomFenSkipping);
        update_u64(hash, seed);
        update_u32(hash, cyclic ? 1U : 0U);
        return hash.digest();
    }

    AtomicV3ProviderCursorV1 initial_cursor() const {
        AtomicV3ProviderCursorV1 cursor{};
        cursor.abiVersion = AtomicV3ProviderAbiVersion;
        cursor.structSize = sizeof(cursor);
        std::copy(binding.begin(), binding.end(), cursor.bindingSha256);
        return cursor;
    }

    void initialize_cursor(const AtomicV3ProviderCursorV1* resume) {
        working = initial_cursor();
        if (resume != nullptr)
        {
            if (resume->abiVersion != AtomicV3ProviderAbiVersion
                || resume->structSize != sizeof(AtomicV3ProviderCursorV1)
                || resume->reserved0 != 0
                || std::any_of(std::begin(resume->reserved1), std::end(resume->reserved1),
                               [](std::uint8_t value) { return value != 0; })
                || !std::equal(binding.begin(), binding.end(), resume->bindingSha256))
                throw std::invalid_argument("Atomic V3 resume cursor contract/binding mismatch");
            working = *resume;
        }
        if (!cyclic && working.epoch != 0)
            throw std::invalid_argument("Atomic V3 non-cyclic resume epoch must be zero");
        if (working.eof)
        {
            if (cyclic || working.manifestIndex != specs.size() || working.recordIndex != 0)
                throw std::invalid_argument("Atomic V3 resume EOF cursor is not canonical");
        }
        else
        {
            if (working.manifestIndex >= specs.size()
                || working.recordIndex >= specs[working.manifestIndex].records)
                throw std::invalid_argument("Atomic V3 resume cursor is outside ordered manifests");
            if (working.recordIndex != 0)
            {
                open_current();
                for (std::uint64_t index = 0; index < working.recordIndex; ++index)
                {
                    AtomicData::AtomicBinV2DecodedRecord decoded{};
                    bool hasRecord = false;
                    const AtomicData::DataResult result = reader->next(decoded, hasRecord);
                    if (!result || !hasRecord || decoded.globalIndex != index)
                        throw std::invalid_argument(
                          "cannot seek C1 reader to Atomic V3 resume cursor"
                          + (result ? std::string() : ": " + result.message));
                }
            }
        }
        committed = working;
    }

    void open_current() {
        if (reader)
            return;
        reader = open_exact_manifest(specs[working.manifestIndex]);
    }

    void finish_manifest() {
        AtomicData::AtomicBinV2DecodedRecord extra{};
        bool hasRecord = false;
        const AtomicData::DataResult result = reader->next(extra, hasRecord);
        if (!result || hasRecord)
            throw std::invalid_argument(
              "C1 failed to reconcile Atomic V3 manifest EOF"
              + (result ? std::string() : ": " + result.message));
        reader.reset();
        working.recordIndex = 0;
        ++working.manifestIndex;
        if (working.manifestIndex == specs.size())
        {
            if (cyclic)
            {
                working.manifestIndex = 0;
                if (working.epoch == std::numeric_limits<std::uint64_t>::max())
                    throw std::overflow_error("Atomic V3 provider epoch overflow");
                ++working.epoch;
            }
            else
            {
                working.eof = 1;
            }
        }
    }

    bool next_raw(AtomicData::AtomicBinV2DecodedRecord& decoded,
                  std::uint64_t& sourceEpoch,
                  std::uint64_t& sourceRoleRecord) {
        if (working.eof)
            return false;
        open_current();
        const ManifestSpec& spec = specs[working.manifestIndex];
        const std::uint64_t local = working.recordIndex;
        bool hasRecord = false;
        const AtomicData::DataResult result = reader->next(decoded, hasRecord);
        if (!result)
            throw std::invalid_argument("C1 rejected Atomic V3 shard/record: " + result.message);
        if (!hasRecord || decoded.globalIndex != local)
            throw std::logic_error("C1 ended before the manifest-declared Atomic V3 record count");
        sourceEpoch      = working.epoch;
        sourceRoleRecord = spec.firstRecord + local;
        ++working.recordIndex;
        if (working.recordIndex == spec.records)
            finish_manifest();
        return true;
    }

    std::vector<ManifestSpec> specs;
    std::unique_ptr<AtomicData::AtomicBinV2DatasetReader> reader;
    std::array<std::uint8_t, 32> binding{};
    std::uint64_t totalRecords = 0;
    std::uint32_t batchSize = 0;
    std::uint32_t randomFenSkipping = 0;
    std::uint64_t seed = 0;
    bool cyclic = false;
    bool poisoned = false;
    bool hasDelivered = false;
    std::string error;
    AtomicV3ProviderCursorV1 working{};
    AtomicV3ProviderCursorV1 committed{};
    AtomicV3ProviderCursorV1 lastDelivered{};
};

}  // namespace

struct AtomicV3ProviderStreamV1 {
    explicit AtomicV3ProviderStreamV1(const AtomicV3ProviderConfigV1& config) : impl(config) {}
    StreamImpl impl;
};

struct AtomicV3ProviderBatchV1 {
    explicit AtomicV3ProviderBatchV1(std::unique_ptr<BatchImpl> value) : impl(std::move(value)) {}
    std::unique_ptr<BatchImpl> impl;
};

extern "C" {

std::uint32_t ATOMIC_V3_PROVIDER_CDECL atomic_v3_provider_abi_version() {
    return AtomicV3ProviderAbiVersion;
}

const char* ATOMIC_V3_PROVIDER_CDECL atomic_v3_provider_creation_error() {
    return CreationError.data();
}

AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_create(const AtomicV3ProviderConfigV1* config,
                          AtomicV3ProviderStreamV1**       output) {
    CreationError[0] = '\0';
    if (output != nullptr)
        *output = nullptr;
    if (config == nullptr || output == nullptr)
    {
        set_creation_error("Atomic V3 provider config/output must not be null");
        return ATOMIC_V3_PROVIDER_ERROR;
    }
    if (config->abiVersion != AtomicV3ProviderAbiVersion
        || config->structSize != sizeof(AtomicV3ProviderConfigV1)
        || config->batchSize == 0 || config->batchSize > MaximumBatchSize
        || config->nativeWorkers != 1
        || (config->cyclic != 0 && config->cyclic != 1)
        || std::any_of(std::begin(config->reserved), std::end(config->reserved),
                       [](std::uint8_t value) { return value != 0; }))
    {
        set_creation_error("Atomic V3 provider ABI/configuration is invalid");
        return ATOMIC_V3_PROVIDER_ERROR;
    }
    try
    {
        *output = new AtomicV3ProviderStreamV1(*config);
        return ATOMIC_V3_PROVIDER_OK;
    }
    catch (const std::exception& exception)
    {
        set_creation_error(exception.what());
        return ATOMIC_V3_PROVIDER_ERROR;
    }
    catch (...)
    {
        set_creation_error("unknown Atomic V3 provider creation error");
        return ATOMIC_V3_PROVIDER_ERROR;
    }
}

void ATOMIC_V3_PROVIDER_CDECL atomic_v3_provider_destroy(AtomicV3ProviderStreamV1* stream) {
    delete stream;
}

AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_fetch(AtomicV3ProviderStreamV1* stream, AtomicV3ProviderBatchV1** output) {
    if (output != nullptr)
        *output = nullptr;
    if (stream == nullptr || output == nullptr)
        return ATOMIC_V3_PROVIDER_ERROR;
    try
    {
        std::unique_ptr<BatchImpl> batch = stream->impl.next();
        if (!batch)
            return ATOMIC_V3_PROVIDER_EOF;
        *output = new AtomicV3ProviderBatchV1(std::move(batch));
        return ATOMIC_V3_PROVIDER_OK;
    }
    catch (const std::exception& exception)
    {
        stream->impl.set_error(exception.what());
        return ATOMIC_V3_PROVIDER_ERROR;
    }
    catch (...)
    {
        stream->impl.set_error("unknown Atomic V3 provider fetch error");
        return ATOMIC_V3_PROVIDER_ERROR;
    }
}

const char* ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_error(const AtomicV3ProviderStreamV1* stream) {
    return stream == nullptr ? "null Atomic V3 provider stream" : stream->impl.last_error().c_str();
}

AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_batch_view(const AtomicV3ProviderBatchV1* batch,
                              AtomicV3BatchViewV1*            output) {
    if (batch == nullptr || batch->impl == nullptr || output == nullptr)
        return ATOMIC_V3_PROVIDER_ERROR;
    *output = batch->impl->view();
    return ATOMIC_V3_PROVIDER_OK;
}

void ATOMIC_V3_PROVIDER_CDECL atomic_v3_provider_destroy_batch(AtomicV3ProviderBatchV1* batch) {
    delete batch;
}

AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_commit(AtomicV3ProviderStreamV1* stream) {
    if (stream == nullptr)
        return ATOMIC_V3_PROVIDER_ERROR;
    try
    {
        stream->impl.commit();
        return ATOMIC_V3_PROVIDER_OK;
    }
    catch (const std::exception& exception)
    {
        stream->impl.set_error(exception.what());
        return ATOMIC_V3_PROVIDER_ERROR;
    }
    catch (...)
    {
        stream->impl.set_error("unknown Atomic V3 provider commit error");
        return ATOMIC_V3_PROVIDER_ERROR;
    }
}

AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_committed_cursor(const AtomicV3ProviderStreamV1* stream,
                                    AtomicV3ProviderCursorV1*       output) {
    if (stream == nullptr || output == nullptr)
        return ATOMIC_V3_PROVIDER_ERROR;
    *output = stream->impl.committed_cursor();
    return ATOMIC_V3_PROVIDER_OK;
}

}  // extern "C"

static_assert(sizeof(AtomicV3ProviderCursorV1) == 88);
