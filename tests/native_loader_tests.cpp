#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "../training_data_loader.cpp"
#include "../lib/atomic_v3_provider.h"
#include "../external/Atomic-Stockfish/src/atomic_init.h"
#include "../external/Atomic-Stockfish/src/data/sha256.h"
#include "../external/Atomic-Stockfish/src/nnue/atomic_v3/full_refresh.h"

#ifdef _WIN32
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess();
extern "C" __declspec(dllimport) int __stdcall GetProcessHandleCount(
    void*, unsigned long*);
#endif

using bin::TrainingDataEntry;
using namespace chess;
namespace AtomicData = Stockfish::Data;
namespace AtomicV3 = Stockfish::Eval::NNUE::AtomicV3;

struct TemporaryDirectory
{
    std::filesystem::path path;

    explicit TemporaryDirectory(const std::string& label)
    {
        const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path()
            / ("atomic-trainer-v2-" + label + "-" + std::to_string(nonce));
        assert(std::filesystem::create_directories(path));
    }

    ~TemporaryDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

struct V2Dataset
{
    std::filesystem::path shard;
    std::filesystem::path manifest;
    AtomicData::AtomicBinV2Manifest metadata;
};

struct V2MultiDataset
{
    std::vector<std::filesystem::path> shards;
    std::filesystem::path manifest;
    AtomicData::AtomicBinV2Manifest metadata;
};

static std::string utf8_path(const std::filesystem::path& path)
{
    return path.u8string();
}

static AtomicData::TrainingDataSample ordinary_v2_sample()
{
    AtomicData::TrainingDataSample sample;
    sample.fen = "7k/8/8/8/8/8/4P3/K7 w - - 32767 100000";
    sample.move = Stockfish::Move(Stockfish::SQ_E2, Stockfish::SQ_E3);
    sample.score = std::numeric_limits<std::int32_t>::max();
    sample.ply = std::numeric_limits<std::uint32_t>::max();
    sample.result = -1;
    return sample;
}

static AtomicData::AtomicBinV2Manifest v2_metadata(
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& shard_path,
    std::string sha256,
    const AtomicData::TrainingDataSample& sample)
{
    AtomicData::AtomicBinV2Manifest manifest;
    manifest.manifestPath = manifest_path;
    manifest.engineCommit = "1e64c6f16e8c327be6ee5e3de57ed54d1079f060";
    manifest.engineVersion = "Atomic-Stockfish trainer integration test";
    manifest.networkPath = manifest_path.parent_path() / "atomic.nnue";
    manifest.networkSha256 = std::string(64, '1');
    manifest.resolvedSeed = 20260713;
    manifest.atomic960 = bool(sample.flags & AtomicData::TRAINING_DATA_CHESS960);
    manifest.threads = 1;
    manifest.hashMb = 16;
    manifest.options.searchDepthMin = 3;
    manifest.options.searchDepthMax = 3;
    manifest.options.evalLimit = 32000;
    manifest.options.evalDiffLimit = 64000;
    manifest.options.randomMoveMinPly = 1;
    manifest.options.randomMoveMaxPly = 24;
    manifest.options.randomMoveCount = 5;
    manifest.options.randomMultiPv = 5;
    manifest.options.randomMultiPvDiff = 100;
    manifest.options.randomMultiPvDepth = 3;
    manifest.options.writeMinPly = 1;
    manifest.options.writeMaxPly = 4096;
    manifest.options.requestedRecords = 1;
    manifest.options.recordsPerShard = 1;
    manifest.options.keepDraws = "0.5";
    manifest.options.filterCaptures = true;
    manifest.options.filterPromotions = true;
    manifest.options.filterChecks = false;
    manifest.records = 1;
    manifest.draws = sample.result == 0 ? 1 : 0;

    AtomicData::AtomicBinV2ManifestShard shard;
    shard.path = shard_path;
    shard.records = 1;
    shard.bytes = AtomicData::AtomicBinV2HeaderSize + AtomicData::AtomicBinV2RecordSize;
    shard.sha256 = std::move(sha256);
    manifest.shards.push_back(std::move(shard));
    return manifest;
}

static bool write_v2_manifest(const AtomicData::AtomicBinV2Manifest& manifest)
{
    std::string json;
    if (!AtomicData::render_atomic_bin_v2_manifest(manifest, json))
        return false;
    std::ofstream output(manifest.manifestPath, std::ios::binary | std::ios::trunc);
    output.write(json.data(), static_cast<std::streamsize>(json.size()));
    return bool(output);
}

static V2Dataset create_v2_dataset(
    const std::filesystem::path& directory,
    const std::string& stem,
    const AtomicData::TrainingDataSample& sample)
{
    V2Dataset dataset;
    dataset.shard = directory / (stem + ".atbin");
    dataset.manifest = directory / (stem + ".atbin.manifest.json");

    AtomicData::AtomicBinV2Header header{};
    AtomicData::AtomicBinV2Record record{};
    assert(AtomicData::encode_atomic_bin_v2_header(1, header));
    assert(AtomicData::encode_atomic_bin_v2(sample, record));
    {
        std::ofstream output(dataset.shard, std::ios::binary | std::ios::trunc);
        output.write(
            reinterpret_cast<const char*>(header.data()),
            static_cast<std::streamsize>(header.size()));
        output.write(
            reinterpret_cast<const char*>(record.data()),
            static_cast<std::streamsize>(record.size()));
        assert(output);
    }

    std::string sha256;
    Stockfish::u64 size = 0;
    assert(AtomicData::sha256_file(dataset.shard, sha256, size));
    dataset.metadata = v2_metadata(dataset.manifest, dataset.shard, sha256, sample);
    assert(write_v2_manifest(dataset.metadata));
    return dataset;
}

static V2MultiDataset create_v2_multishard_dataset(
    const std::filesystem::path& directory,
    const std::string& stem,
    const std::vector<std::vector<AtomicData::TrainingDataSample>>& shard_samples)
{
    assert(shard_samples.size() >= 2);
    assert(!shard_samples.front().empty());

    V2MultiDataset dataset;
    dataset.manifest = directory / (stem + ".atbin.manifest.json");
    dataset.metadata = v2_metadata(
        dataset.manifest,
        directory / (stem + ".atbin"),
        std::string(64, '0'),
        shard_samples.front().front());
    dataset.metadata.shards.clear();
    dataset.metadata.records = 0;
    dataset.metadata.draws = 0;
    dataset.metadata.options.recordsPerShard = shard_samples.front().size();

    for (std::size_t shard_index = 0; shard_index < shard_samples.size(); ++shard_index)
    {
        const auto& samples = shard_samples[shard_index];
        assert(!samples.empty());
        if (shard_index + 1 < shard_samples.size())
            assert(samples.size() == shard_samples.front().size());
        else
            assert(samples.size() <= shard_samples.front().size());

        const auto shard_path = directory
            / (shard_index == 0
                ? stem + ".atbin"
                : stem + "-" + std::to_string(shard_index) + ".atbin");
        AtomicData::AtomicBinV2Header header{};
        assert(AtomicData::encode_atomic_bin_v2_header(samples.size(), header));
        {
            std::ofstream output(shard_path, std::ios::binary | std::ios::trunc);
            output.write(
                reinterpret_cast<const char*>(header.data()),
                static_cast<std::streamsize>(header.size()));
            for (const auto& sample : samples)
            {
                AtomicData::AtomicBinV2Record record{};
                assert(AtomicData::encode_atomic_bin_v2(sample, record));
                output.write(
                    reinterpret_cast<const char*>(record.data()),
                    static_cast<std::streamsize>(record.size()));
                dataset.metadata.draws += sample.result == 0 ? 1 : 0;
            }
            assert(output);
        }

        std::string sha256;
        Stockfish::u64 size = 0;
        assert(AtomicData::sha256_file(shard_path, sha256, size));
        AtomicData::AtomicBinV2ManifestShard shard;
        shard.path = shard_path;
        shard.index = shard_index;
        shard.records = samples.size();
        shard.bytes = size;
        shard.sha256 = std::move(sha256);
        dataset.metadata.shards.push_back(std::move(shard));
        dataset.metadata.records += samples.size();
        dataset.shards.push_back(shard_path);
    }

    dataset.metadata.options.requestedRecords = dataset.metadata.records;
    assert(write_v2_manifest(dataset.metadata));
    return dataset;
}

static Square sq(int file, int rank)
{
    return static_cast<Square>(rank * static_cast<int>(File::FILE_NB) + file);
}

static void set_bits(bin::nodchip::PackedSfen& sfen, int offset, std::uint32_t value, int width)
{
    for (int bit = 0; bit < width; ++bit)
    {
        const auto mask = static_cast<std::uint8_t>(1U << ((offset + bit) & 7));
        auto& byte = sfen.data[(offset + bit) / 8];
        if ((value >> bit) & 1U)
            byte |= mask;
        else
            byte &= static_cast<std::uint8_t>(~mask);
    }
}

static void set_raw_move(bin::nodchip::PackedSfenValue& record, std::uint16_t raw)
{
    static_assert(sizeof(record.move) == sizeof(raw));
    std::memcpy(&record.move, &raw, sizeof(raw));
}

static bin::nodchip::PackedSfenValue valid_legacy_record()
{
    bin::nodchip::PackedSfenValue record{};
    int cursor = 0;
    auto write = [&](std::uint32_t value, int width) {
        set_bits(record.sfen, cursor, value, width);
        cursor += width;
    };

    write(0, 1);   // White to move.
    write(4, 7);   // White king e1.
    write(60, 7);  // Black king e8.

    for (int rank = static_cast<int>(Rank::RANK_MAX);
         rank >= static_cast<int>(Rank::RANK_1);
         --rank)
    {
        for (int file = static_cast<int>(File::FILE_A);
             file <= static_cast<int>(File::FILE_MAX);
             ++file)
        {
            const auto square = sq(file, rank);
            if (square == sq(4, 0) || square == sq(4, 7))
                continue;
            if (square == sq(0, 1))
            {
                write(0b00001, 5);  // Pawn.
                write(0, 1);        // White.
            }
            else
                write(0, 1);        // Empty.
        }
    }

    for (Color color : {Color::White, Color::Black})
        for (PieceType pt = PieceType::Pawn; pt <= PieceType::MaxPiece; ++pt)
        {
            (void) color;
            write(0, DATA_SIZE > 512 ? 7 : 5);
        }

    for (int i = 0; i < 4; ++i)
        write(0, 1);  // No castling rights.
    write(0, 1);      // No en-passant square.
    write(0, 6);      // Rule 50 low bits.
    write(1, 8);      // Fullmove low bits.
    write(0, 8);      // Fullmove high bits.
    write(0, 1);      // Rule 50 high bit.
    assert(cursor <= DATA_SIZE);

    set_raw_move(record, static_cast<std::uint16_t>(16 | (8 << 6)));  // a2a3.
    record.game_result = 0;
    return record;
}

static void expect_invalid_record(
    const bin::nodchip::PackedSfenValue& record,
    const char* expected_message)
{
    try
    {
        (void) bin::packedSfenValueToTrainingDataEntry(record);
        assert(false && "corrupt Legacy72 record was accepted");
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()).find(expected_message) != std::string::npos);
    }
}

static TrainingDataEntry quiet_entry()
{
    TrainingDataEntry entry{};
    entry.pos.place(make_piece(PieceType::King, Color::White), sq(4, 0));
    entry.pos.place(make_piece(PieceType::King, Color::Black), sq(4, 7));
    entry.pos.place(make_piece(PieceType::Pawn, Color::White), sq(0, 1));
    entry.pos.setSideToMove(Color::White);
    entry.move = Move{sq(0, 1), sq(0, 2), MoveType::Normal};
    return entry;
}

static void test_smart_filter()
{
    auto entry = quiet_entry();
    assert(!is_smart_filtered(entry));
    auto predicate = make_skip_predicate(true, 0, 0);
    assert(!predicate(entry));

    entry.pos.place(make_piece(PieceType::Knight, Color::Black), sq(0, 2));
    assert(is_smart_filtered(entry));
    assert(predicate(entry));

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Pawn, Color::White), sq(6, 6));
    entry.pos.place(make_piece(PieceType::Rook, Color::Black), sq(7, 7));
    entry.move = Move{
        sq(6, 6),
        sq(7, 7),
        MoveType::Promotion,
        make_piece(PieceType::Queen, Color::White)
    };
    assert(is_smart_filtered(entry));

    entry = quiet_entry();
    entry.move = Move{sq(0, 1), sq(1, 2), MoveType::EnPassant};
    assert(is_smart_filtered(entry));

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Rook, Color::White), sq(7, 0));
    entry.move = Move{sq(4, 0), sq(7, 0), MoveType::Castle};
    assert(!is_smart_filtered(entry));

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Pawn, Color::Black), sq(3, 1));
    assert(is_smart_filtered(entry)); // black pawn attacks the white king on e1

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Knight, Color::Black), sq(5, 2));
    assert(is_smart_filtered(entry));

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Rook, Color::Black), sq(4, 5));
    assert(is_smart_filtered(entry));
    entry.pos.place(make_piece(PieceType::Pawn, Color::White), sq(4, 3));
    assert(!is_smart_filtered(entry));

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Bishop, Color::Black), sq(7, 3));
    assert(is_smart_filtered(entry));
    entry.pos.place(make_piece(PieceType::Pawn, Color::White), sq(6, 2));
    assert(!is_smart_filtered(entry));

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Rook, Color::White), sq(4, 6));
    assert(!is_smart_filtered(entry)); // attacking the non-moving king is irrelevant

    entry = TrainingDataEntry{};
    entry.pos.setSideToMove(Color::White);
    entry.move = Move{sq(0, 1), sq(0, 2), MoveType::Normal};
    assert(!is_smart_filtered(entry)); // missing king is handled safely

    entry = quiet_entry();
    entry.pos.place(Piece::None, sq(4, 7));
    entry.pos.place(make_piece(PieceType::King, Color::Black), sq(4, 1));
    // The legacy smart-skip heuristic deliberately uses geometric orthodox
    // check detection; it is not an Atomic legality oracle.
    assert(is_smart_filtered(entry));
}

static void test_seeded_random_skipping()
{
    const auto entry = quiet_entry();
    auto first = make_skip_predicate(false, 3, 0x123456789abcdef0ULL);
    auto second = make_skip_predicate(false, 3, 0x123456789abcdef0ULL);
    auto different = make_skip_predicate(false, 3, 0x0fedcba987654321ULL);

    std::uint64_t first_bits = 0;
    std::uint64_t second_bits = 0;
    std::uint64_t different_bits = 0;
    for (int i = 0; i < 64; ++i)
    {
        first_bits |= static_cast<std::uint64_t>(first(entry)) << i;
        second_bits |= static_cast<std::uint64_t>(second(entry)) << i;
        different_bits |= static_cast<std::uint64_t>(different(entry)) << i;
    }
    assert(first_bits == second_bits);
    assert(first_bits != different_bits);
    assert(first_bits == 0xaf46d7fda3c76fffULL);

    auto maximum = make_skip_predicate(false, std::numeric_limits<int>::max(), 1);
    const auto start = std::chrono::steady_clock::now();
    (void) maximum(entry);
    assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
}

static void test_c_api_validation()
{
    assert(create_sparse_batch_stream_with_seed("HalfKAv2", 0, "unused.bin", 1, false, false, 0, 0) == nullptr);
    assert(create_sparse_batch_stream_with_seed("HalfKAv2", 1, "unused.bin", 0, false, false, 0, 0) == nullptr);
    assert(create_sparse_batch_stream_with_seed("HalfKAv2", 1, "unused.bin", 1, false, false, -1, 0) == nullptr);
    assert(create_sparse_batch_stream_with_seed(nullptr, 1, "unused.bin", 1, false, false, 0, 0) == nullptr);
    assert(create_sparse_batch_stream_with_seed("unknown", 1, "unused.bin", 1, false, false, 0, 0) == nullptr);
    assert(create_sparse_batch_stream_with_seed("HalfKAv2", 1, "unused.txt", 1, false, false, 0, 0) == nullptr);
    assert(create_sparse_batch_stream_with_seed("HalfKAv2", 1, "does-not-exist.bin", 1, false, false, 0, 0) == nullptr);

    const auto unique_suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
    );
    const auto malformed = std::filesystem::temp_directory_path()
        / ("malformed-legacy-loader-test-" + unique_suffix + ".bin");
    {
        std::ofstream output(malformed, std::ios::binary);
        assert(output);
        output.put('\0');
    }
    const auto malformed_string = malformed.string();
    assert(create_sparse_batch_stream_with_seed("HalfKAv2", 1, malformed_string.c_str(), 1, false, false, 0, 0) == nullptr);
    assert(std::filesystem::remove(malformed));
    assert(fetch_next_sparse_batch(nullptr) == nullptr);
}

static void test_atomic_training_data_schema_handshake()
{
    const char* schema_json = get_atomic_training_data_schema_json();
    assert(schema_json != nullptr);
    assert(std::string(schema_json) ==
        "{\"schema_sha256\":\"acca0f551f1c012c31a6c727dedccaebb7b5ebbc46810edb87e31bb208d5abe1\","
        "\"formats\":{\"legacy-atomic-v1\":{\"read\":true,\"write\":false,"
        "\"record_size\":72}}}");

    const char* schemas_json = get_atomic_training_data_schemas_json();
    assert(schemas_json != nullptr);
    assert(std::string(schemas_json) ==
        "{\"capability_version\":2,\"formats\":{\"legacy-atomic-v1\":{"
        "\"schema_sha256\":\"acca0f551f1c012c31a6c727dedccaebb7b5ebbc46810edb87e31bb208d5abe1\","
        "\"read\":true,\"write\":false,\"header_size\":0,\"record_size\":72},"
        "\"atomic-bin-v2\":{\"read\":true,\"write\":false,"
        "\"entrypoint\":\"manifest\",\"header_size\":96,\"record_size\":64,"
        "\"schema_sha256\":\"0352b036f2a140c609e3eb9c9d635dc553e8d77253d8faa92437390f5cf93cb6\","
        "\"manifest_schema_sha256\":\"83d63922df3ac4a0c81a21ec9d9fd9e180efe50f26efee62fe01710e09da5b42\"}}}");
}

static void test_atomic_bin_v2_manifest_only_roundtrip()
{
    Stockfish::initialize_atomic_core();
    TemporaryDirectory temporary("roundtrip");
    const auto sample = ordinary_v2_sample();
    const auto dataset = create_v2_dataset(temporary.path, "dataset", sample);
    const auto manifest = utf8_path(dataset.manifest);

    training_data::AtomicBinV2InputStream input(manifest, false, nullptr);
    const auto decoded = input.next();
    assert(decoded.has_value());
    assert(decoded->score == std::numeric_limits<std::int32_t>::max());
    assert(decoded->ply == std::numeric_limits<std::uint32_t>::max());
    assert(decoded->result == -1);
    assert(decoded->flags == bin::NoTrainingDataFlags);
    assert(decoded->pos.sideToMove() == Color::White);
    assert(decoded->pos.rule50Counter() == 32767);
    assert(decoded->pos.fullMove() == 100000);
    assert(!input.next().has_value());
    assert(input.eof());

    training_data::AtomicBinV2InputStream cyclic(manifest, true, nullptr);
    assert(cyclic.next()->ply == std::numeric_limits<std::uint32_t>::max());
    assert(cyclic.next()->ply == std::numeric_limits<std::uint32_t>::max());

    const auto raw_shard = utf8_path(dataset.shard);
    auto* raw = create_sparse_batch_stream_with_seed(
        "HalfKAv2", 1, raw_shard.c_str(), 1, false, false, 0, 0);
    assert(raw == nullptr);
    assert(std::string(get_sparse_batch_stream_creation_error()).find("raw shards")
        != std::string::npos);

    auto experimental = create_v2_dataset(temporary.path, "experimental", sample);
    experimental.metadata.options.evalLimit = 3000;
    experimental.metadata.options.filterCaptures = false;
    experimental.metadata.options.filterPromotions = false;
    experimental.metadata.options.filterChecks = true;
    assert(write_v2_manifest(experimental.metadata));
    training_data::AtomicBinV2InputStream experimental_input(
        utf8_path(experimental.manifest),
        false,
        nullptr);
    assert(experimental_input.next().has_value());
}

static void test_widened_internal_position_boundaries()
{
    chess::Position position;
    assert(position.kingSquare(Color::White) == chess::Square::NB);
    assert(position.kingSquare(Color::Black) == chess::Square::NB);
    position.setRule50Counter(std::numeric_limits<std::uint16_t>::max());
    position.setFullMove(std::numeric_limits<std::uint32_t>::max());
    position.setSideToMove(Color::Black);

    assert(position.rule50Counter() == std::numeric_limits<std::uint16_t>::max());
    assert(position.fullMove() == std::numeric_limits<std::uint32_t>::max());
    assert(position.ply() == 2ULL * std::numeric_limits<std::uint32_t>::max());
}

static void test_v2_adapter_defensively_preserves_missing_king_sentinel()
{
    // The authenticated V2 schema currently requires both kings. This direct
    // adapter test protects the internal Position invariant if a future schema
    // permits terminal records; it does not weaken the current reader.
    Stockfish::Data::AtomicBinV2DecodedRecord decoded{};
    auto& fields = decoded.fields;
    fields.position.board.fill(Stockfish::Data::ATOMIC_BIN_V2_EMPTY);
    fields.position.board[static_cast<std::size_t>(Stockfish::SQ_E8)] =
        Stockfish::Data::ATOMIC_BIN_V2_BLACK_KING;
    fields.position.board[static_cast<std::size_t>(Stockfish::SQ_A2)] =
        Stockfish::Data::ATOMIC_BIN_V2_WHITE_PAWN;
    fields.position.sideToMove = Stockfish::Data::ATOMIC_BIN_V2_WHITE_TO_MOVE;
    fields.position.enPassantSquare = Stockfish::Data::AtomicBinV2NoSquare;
    fields.position.castlingRookOrigins.fill(Stockfish::Data::AtomicBinV2NoSquare);
    fields.position.fullmove = 1;
    fields.move.from = static_cast<Stockfish::u8>(Stockfish::SQ_A2);
    fields.move.to = static_cast<Stockfish::u8>(Stockfish::SQ_A3);
    fields.move.type = Stockfish::Data::ATOMIC_BIN_V2_NORMAL;
    fields.move.promotion = Stockfish::Data::ATOMIC_BIN_V2_NO_PROMOTION;

    const auto entry = training_data::atomic_bin_v2::to_training_data_entry(decoded);
    assert(entry.pos.kingSquare(Color::White) == chess::Square::NB);
    assert(entry.pos.kingSquare(Color::Black) == sq(4, 7));
}

static void test_atomic_bin_v2_atomic960_and_stm_semantics()
{
    TemporaryDirectory temporary("atomic960");
    AtomicData::TrainingDataSample sample;
    sample.fen = "7k/8/8/8/8/8/8/1RK5 w B - 0 1";
    sample.move = Stockfish::Move::make<Stockfish::CASTLING>(
        Stockfish::SQ_C1,
        Stockfish::SQ_B1);
    sample.score = -321;
    sample.ply = 27;
    sample.result = 1;
    sample.flags = AtomicData::TRAINING_DATA_CHESS960;
    const auto dataset = create_v2_dataset(temporary.path, "atomic960", sample);

    training_data::AtomicBinV2InputStream input(utf8_path(dataset.manifest), false, nullptr);
    const auto decoded = input.next();
    assert(decoded.has_value());
    assert(decoded->score == -321);
    assert(decoded->result == 1);
    assert(decoded->ply == 27);
    assert(decoded->flags == bin::TrainingDataAtomic960);
    assert(decoded->move.type == MoveType::Castle);
    assert(decoded->move.from == sq(2, 0));
    assert(decoded->move.to == sq(1, 0));
    assert(contains(decoded->pos.castlingRights(), CastlingRights::WhiteQueenSide));
    assert(decoded->pos.castlingRookOrigin(1) == sq(1, 0));

    AtomicData::TrainingDataSample black_sample;
    black_sample.fen = "7k/4p3/8/8/8/8/8/K7 b - - 0 1";
    black_sample.move = Stockfish::Move(Stockfish::SQ_E7, Stockfish::SQ_E6);
    black_sample.score = -444;
    black_sample.ply = 9;
    black_sample.result = 1;
    const auto black_dataset = create_v2_dataset(
        temporary.path,
        "black-stm",
        black_sample);
    training_data::AtomicBinV2InputStream black_input(
        utf8_path(black_dataset.manifest),
        false,
        nullptr);
    const auto black_decoded = black_input.next();
    assert(black_decoded.has_value());
    assert(black_decoded->pos.sideToMove() == Color::Black);
    assert(black_decoded->score == -444);
    assert(black_decoded->result == 1);
}

struct BatchSequence
{
    std::vector<int> batch_sizes;
    std::vector<float> scores;

    friend bool operator==(const BatchSequence& left, const BatchSequence& right)
    {
        return left.batch_sizes == right.batch_sizes && left.scores == right.scores;
    }
};

static BatchSequence collect_v2_batches(
    const std::filesystem::path& manifest,
    int workers,
    int batch_size)
{
    const auto manifest_path = utf8_path(manifest);
    auto* stream = create_sparse_batch_stream_with_seed(
        "HalfKAv2", workers, manifest_path.c_str(), batch_size, false, false, 0, 17);
    assert(stream != nullptr);

    BatchSequence sequence;
    while (SparseBatch* batch = fetch_next_sparse_batch(stream))
    {
        sequence.batch_sizes.push_back(batch->size);
        sequence.scores.insert(sequence.scores.end(), batch->score, batch->score + batch->size);
        destroy_sparse_batch(batch);
    }
    assert(get_sparse_batch_stream_error(stream)[0] == '\0');
    destroy_sparse_batch_stream(stream);
    return sequence;
}

static void test_atomic_bin_v2_multishard_eof_partial_and_determinism()
{
    TemporaryDirectory temporary("multishard");
    std::vector<AtomicData::TrainingDataSample> samples;
    for (int index = 0; index < 5; ++index)
    {
        auto sample = ordinary_v2_sample();
        sample.score = 100 + index;
        sample.ply = 200 + index;
        samples.push_back(std::move(sample));
    }
    const auto dataset = create_v2_multishard_dataset(
        temporary.path,
        "multi",
        {{samples.begin(), samples.begin() + 3}, {samples.begin() + 3, samples.end()}});

    const auto single_worker = collect_v2_batches(dataset.manifest, 1, 3);
    const auto four_workers = collect_v2_batches(dataset.manifest, 4, 3);
    assert(single_worker == four_workers);
    assert(single_worker.batch_sizes == std::vector<int>({3, 2}));
    assert(single_worker.scores == std::vector<float>({100, 101, 102, 103, 104}));

    training_data::AtomicBinV2InputStream cyclic(
        utf8_path(dataset.manifest),
        true,
        nullptr);
    for (int index = 0; index < 5; ++index)
        assert(cyclic.next()->score == 100 + index);
    assert(cyclic.next()->score == 100);
}

static void assert_sparse_batches_equal(const SparseBatch& left, const SparseBatch& right)
{
    assert(left.num_inputs == right.num_inputs);
    assert(left.size == right.size);
    assert(left.num_active_white_features == right.num_active_white_features);
    assert(left.num_active_black_features == right.num_active_black_features);
    assert(left.max_active_features == right.max_active_features);

    for (int index = 0; index < left.size; ++index)
    {
        assert(left.is_white[index] == right.is_white[index]);
        assert(left.outcome[index] == right.outcome[index]);
        assert(left.score[index] == right.score[index]);
        assert(left.psqt_indices[index] == right.psqt_indices[index]);
        assert(left.layer_stack_indices[index] == right.layer_stack_indices[index]);
    }
    for (int index = 0; index < left.size * left.max_active_features; ++index)
    {
        assert(left.white[index] == right.white[index]);
        assert(left.black[index] == right.black[index]);
        assert(left.white_values[index] == right.white_values[index]);
        assert(left.black_values[index] == right.black_values[index]);
    }
}

static void test_atomic_bin_v2_halfkav2_sparse_parity_with_legacy72()
{
    TemporaryDirectory temporary("sparse-parity");
    auto legacy_record = valid_legacy_record();
    legacy_record.score = -77;
    legacy_record.gamePly = 12;
    legacy_record.game_result = 1;
    const auto legacy_path = temporary.path / "same-position.bin";
    {
        std::ofstream output(legacy_path, std::ios::binary | std::ios::trunc);
        output.write(
            reinterpret_cast<const char*>(&legacy_record),
            static_cast<std::streamsize>(sizeof(legacy_record)));
        assert(output);
    }

    AtomicData::TrainingDataSample v2_sample;
    v2_sample.fen = "4k3/8/8/8/8/8/P7/4K3 w - - 0 1";
    v2_sample.move = Stockfish::Move(Stockfish::SQ_A2, Stockfish::SQ_A3);
    v2_sample.score = -77;
    v2_sample.ply = 12;
    v2_sample.result = 1;
    const auto v2_dataset = create_v2_dataset(temporary.path, "same-position", v2_sample);

    const auto legacy_filename = utf8_path(legacy_path);
    const auto v2_filename = utf8_path(v2_dataset.manifest);
    auto* legacy_stream = create_sparse_batch_stream_with_seed(
        "HalfKAv2", 1, legacy_filename.c_str(), 1, false, false, 0, 0);
    auto* v2_stream = create_sparse_batch_stream_with_seed(
        "HalfKAv2", 1, v2_filename.c_str(), 1, false, false, 0, 0);
    assert(legacy_stream != nullptr);
    assert(v2_stream != nullptr);

    SparseBatch* legacy_batch = fetch_next_sparse_batch(legacy_stream);
    SparseBatch* v2_batch = fetch_next_sparse_batch(v2_stream);
    assert(legacy_batch != nullptr);
    assert(v2_batch != nullptr);
    assert_sparse_batches_equal(*legacy_batch, *v2_batch);
    destroy_sparse_batch(legacy_batch);
    destroy_sparse_batch(v2_batch);

    assert(fetch_next_sparse_batch(legacy_stream) == nullptr);
    assert(fetch_next_sparse_batch(v2_stream) == nullptr);
    assert(get_sparse_batch_stream_error(legacy_stream)[0] == '\0');
    assert(get_sparse_batch_stream_error(v2_stream)[0] == '\0');
    destroy_sparse_batch_stream(legacy_stream);
    destroy_sparse_batch_stream(v2_stream);
}

static void test_atomic_bin_v2_bypasses_legacy_smart_filter()
{
    TemporaryDirectory temporary("filter-policy");
    AtomicData::TrainingDataSample sample;
    sample.fen = "7k/8/8/8/8/3p4/4P3/K7 w - - 0 1";
    sample.move = Stockfish::Move(Stockfish::SQ_E2, Stockfish::SQ_D3);
    sample.score = 17;
    sample.ply = 3;
    sample.result = 1;
    const auto dataset = create_v2_dataset(temporary.path, "capture", sample);
    const auto manifest = utf8_path(dataset.manifest);

    auto* stream = create_sparse_batch_stream_with_seed(
        "HalfKAv2", 2, manifest.c_str(), 1, false, true, 0, 7);
    assert(stream != nullptr);
    SparseBatch* batch = fetch_next_sparse_batch(stream);
    assert(batch != nullptr);
    assert(batch->size == 1);
    assert(batch->score[0] == 17.0f);
    destroy_sparse_batch(batch);
    assert(fetch_next_sparse_batch(stream) == nullptr);
    assert(get_sparse_batch_stream_error(stream)[0] == '\0');
    destroy_sparse_batch_stream(stream);
}

static void test_atomic_bin_v2_corruption_and_schema_fail_closed()
{
    TemporaryDirectory temporary("fail-closed");
    const auto sample = ordinary_v2_sample();

    auto corrupt = create_v2_dataset(temporary.path, "corrupt", sample);
    {
        std::fstream shard(corrupt.shard, std::ios::in | std::ios::out | std::ios::binary);
        assert(shard);
        shard.seekp(-1, std::ios::end);
        const char changed = '\x01';
        shard.write(&changed, 1);
        assert(shard);
    }
    const auto corrupt_manifest = utf8_path(corrupt.manifest);
    auto* stream = create_sparse_batch_stream_with_seed(
        "HalfKAv2", 1, corrupt_manifest.c_str(), 1, false, false, 0, 0);
    if (stream != nullptr)
    {
        assert(fetch_next_sparse_batch(stream) == nullptr);
        assert(std::string(get_sparse_batch_stream_error(stream)).find("SHA-256")
            != std::string::npos);
        destroy_sparse_batch_stream(stream);
    }
    else
        assert(std::string(get_sparse_batch_stream_creation_error()).find("SHA-256")
            != std::string::npos);

    auto missing_king = create_v2_dataset(temporary.path, "missing-king", sample);
    {
        std::fstream shard(
            missing_king.shard,
            std::ios::in | std::ios::out | std::ios::binary);
        assert(shard);
        shard.seekg(static_cast<std::streamoff>(AtomicData::AtomicBinV2HeaderSize));
        char first_board_byte = 0;
        shard.read(&first_board_byte, 1);
        assert(shard);
        first_board_byte = static_cast<char>(
            static_cast<unsigned char>(first_board_byte) & 0xF0U);
        shard.seekp(static_cast<std::streamoff>(AtomicData::AtomicBinV2HeaderSize));
        shard.write(&first_board_byte, 1);
        assert(shard);
    }
    std::string missing_king_sha256;
    Stockfish::u64 missing_king_size = 0;
    assert(AtomicData::sha256_file(
        missing_king.shard,
        missing_king_sha256,
        missing_king_size));
    missing_king.metadata.shards.front().sha256 = std::move(missing_king_sha256);
    assert(write_v2_manifest(missing_king.metadata));
    const auto missing_king_manifest = utf8_path(missing_king.manifest);
    stream = create_sparse_batch_stream_with_seed(
        "HalfKAv2", 1, missing_king_manifest.c_str(), 1, false, false, 0, 0);
    assert(stream != nullptr);
    assert(fetch_next_sparse_batch(stream) == nullptr);
    assert(std::string(get_sparse_batch_stream_error(stream)).find("king")
        != std::string::npos);
    destroy_sparse_batch_stream(stream);

    auto schema = create_v2_dataset(temporary.path, "schema", sample);
    std::ifstream input(schema.manifest, std::ios::binary);
    std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    input.close();
    const auto marker = json.find(AtomicData::AtomicBinV2SchemaSha256Hex);
    assert(marker != std::string::npos);
    json[marker] = json[marker] == '0' ? '1' : '0';
    {
        std::ofstream output(schema.manifest, std::ios::binary | std::ios::trunc);
        output.write(json.data(), static_cast<std::streamsize>(json.size()));
        assert(output);
    }
    const auto schema_manifest = utf8_path(schema.manifest);
    assert(create_sparse_batch_stream_with_seed(
        "HalfKAv2", 1, schema_manifest.c_str(), 1, false, false, 0, 0) == nullptr);
    assert(std::string(get_sparse_batch_stream_creation_error()).find("schema")
        != std::string::npos);
}

static void test_atomic_bin_v2_train_validation_overlap()
{
    TemporaryDirectory temporary("overlap");
    const auto sample = ordinary_v2_sample();
    const auto train = create_v2_dataset(temporary.path, "train", sample);
    const auto copied = create_v2_dataset(temporary.path, "copied", sample);

    const auto train_manifest = utf8_path(train.manifest);
    const auto copied_manifest = utf8_path(copied.manifest);
    const char* overlap = validate_training_validation_data_paths(
        train_manifest.c_str(),
        copied_manifest.c_str());
    assert(overlap != nullptr);
    assert(std::string(overlap).find("overlap") != std::string::npos);

    auto different_sample = sample;
    different_sample.score -= 1;
    const auto distinct = create_v2_dataset(temporary.path, "distinct", different_sample);
    const auto distinct_manifest = utf8_path(distinct.manifest);
    assert(validate_training_validation_data_paths(
        train_manifest.c_str(),
        distinct_manifest.c_str()) == nullptr);

    const auto hardlink_shard = temporary.path / "hardlink.atbin";
    std::filesystem::create_hard_link(train.shard, hardlink_shard);
    auto hardlink_metadata = v2_metadata(
        temporary.path / "hardlink.atbin.manifest.json",
        hardlink_shard,
        std::string(64, '2'),
        sample);
    assert(write_v2_manifest(hardlink_metadata));
    const auto hardlink_manifest = utf8_path(hardlink_metadata.manifestPath);
    overlap = validate_training_validation_data_paths(
        train_manifest.c_str(),
        hardlink_manifest.c_str());
    assert(overlap != nullptr);
    assert(std::string(overlap).find("overlap") != std::string::npos);
}

static void test_corrupt_full_size_records_fail_closed()
{
    const auto valid = valid_legacy_record();
    const auto decoded = bin::packedSfenValueToTrainingDataEntry(valid);
    assert(decoded.move.from == sq(0, 1));
    assert(decoded.move.to == sq(0, 2));

    std::vector<std::pair<std::string, bin::nodchip::PackedSfenValue>> corrupt;

    auto duplicate_king = valid;
    set_bits(duplicate_king.sfen, 8, 4, 7);
    corrupt.emplace_back("same square", duplicate_king);

    auto king_out_of_range = valid;
    set_bits(king_out_of_range.sfen, 1, 65, 7);
    corrupt.emplace_back("outside the board", king_out_of_range);

    auto invalid_huffman = valid;
    set_bits(invalid_huffman.sfen, 15, 0b01011, 5);
    corrupt.emplace_back("Huffman", invalid_huffman);

    auto invalid_move = valid;
    set_raw_move(invalid_move, static_cast<std::uint16_t>(8 | (8 << 6)));
    corrupt.emplace_back("move squares", invalid_move);

    auto invalid_result = valid;
    invalid_result.game_result = 2;
    corrupt.emplace_back("result", invalid_result);

    const auto unique_suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
    );

    for (std::size_t index = 0; index < corrupt.size(); ++index)
    {
        const auto& [message, record] = corrupt[index];
        expect_invalid_record(record, message.c_str());

        const auto path = std::filesystem::temp_directory_path()
            / ("corrupt-legacy72-" + unique_suffix + "-" + std::to_string(index) + ".bin");
        {
            std::ofstream output(path, std::ios::binary);
            assert(output);
            output.write(reinterpret_cast<const char*>(&record), sizeof(record));
            assert(output);
        }

        const auto path_string = path.string();
        auto stream = create_sparse_batch_stream_with_seed(
            "HalfKAv2", 2, path_string.c_str(), 1, false, false, 0, 0
        );
        assert(stream != nullptr);
        assert(fetch_next_sparse_batch(stream) == nullptr);
        const char* native_error = get_sparse_batch_stream_error(stream);
        assert(native_error != nullptr && native_error[0] != '\0');
        assert(std::string(native_error).find(message) != std::string::npos);
        destroy_sparse_batch_stream(stream);
        assert(std::filesystem::remove(path));
    }
}

static AtomicData::TrainingDataSample v3_sample(
    const std::string& fen,
    Stockfish::Move move,
    std::int32_t score,
    int result)
{
    AtomicData::TrainingDataSample sample;
    sample.fen = fen;
    sample.move = move;
    sample.score = score;
    sample.ply = 17;
    sample.result = result;
    return sample;
}

static std::string read_binary_file(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    assert(input);
    const auto end = input.tellg();
    assert(end > 0);
    input.seekg(0);
    std::string bytes(static_cast<std::size_t>(end), '\0');
    input.read(bytes.data(), end);
    assert(input);
    return bytes;
}

struct V3ProviderInputs
{
    std::vector<std::string> paths;
    std::vector<std::string> payloads;
    std::vector<std::string> hashes;
    std::vector<std::uint64_t> records;
    std::vector<AtomicV3ManifestInputV1> descriptors;

    explicit V3ProviderInputs(const std::vector<const V2Dataset*>& datasets)
    {
        paths.reserve(datasets.size());
        payloads.reserve(datasets.size());
        hashes.reserve(datasets.size());
        records.reserve(datasets.size());
        descriptors.reserve(datasets.size());
        for (const auto* dataset : datasets)
        {
            paths.push_back(utf8_path(dataset->manifest));
            payloads.push_back(read_binary_file(dataset->manifest));
            AtomicData::Sha256 digest;
            digest.update(payloads.back());
            hashes.push_back(digest.hex_digest());
            records.push_back(dataset->metadata.records);
        }
        for (std::size_t index = 0; index < datasets.size(); ++index)
            descriptors.push_back({
                paths[index].data(), paths[index].size(),
                reinterpret_cast<const std::uint8_t*>(payloads[index].data()),
                payloads[index].size(), hashes[index].data(), hashes[index].size(), records[index]});
    }

    AtomicV3ProviderConfigV1 config(
        std::uint32_t batchSize,
        std::uint32_t skipping,
        bool cyclic,
        const AtomicV3ProviderCursorV1* resume = nullptr) const
    {
        AtomicV3ProviderConfigV1 value{};
        value.abiVersion = AtomicV3ProviderAbiVersion;
        value.structSize = sizeof(value);
        value.manifests = descriptors.data();
        value.manifestCount = static_cast<std::uint32_t>(descriptors.size());
        value.batchSize = batchSize;
        value.randomFenSkipping = skipping;
        value.nativeWorkers = 1;
        value.seed = 20260716;
        value.cyclic = cyclic ? 1 : 0;
        value.resumeCursor = resume;
        return value;
    }
};

static AtomicV3ProviderStreamV1* create_v3_provider(const AtomicV3ProviderConfigV1& config)
{
    AtomicV3ProviderStreamV1* stream = nullptr;
    assert(atomic_v3_provider_create(&config, &stream) == ATOMIC_V3_PROVIDER_OK);
    assert(stream != nullptr);
    return stream;
}

static AtomicV3BatchViewV1 fetch_v3_batch(
    AtomicV3ProviderStreamV1* stream,
    AtomicV3ProviderBatchV1*& batch)
{
    batch = nullptr;
    assert(atomic_v3_provider_fetch(stream, &batch) == ATOMIC_V3_PROVIDER_OK);
    assert(batch != nullptr);
    AtomicV3BatchViewV1 view{};
    assert(atomic_v3_provider_batch_view(batch, &view) == ATOMIC_V3_PROVIDER_OK);
    assert(view.abiVersion == AtomicV3ProviderAbiVersion);
    assert(view.structSize == sizeof(view));
    return view;
}

static std::uint64_t v3_splitmix64(std::uint64_t value)
{
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

static bool v3_keep_record(
    std::uint64_t seed,
    std::uint64_t epoch,
    std::uint64_t role_record,
    std::uint32_t random_fen_skipping)
{
    if (random_fen_skipping == 0)
        return true;
    const std::uint64_t bound = std::uint64_t(random_fen_skipping) + 1;
    const std::uint64_t threshold = (0ULL - bound) % bound;
    std::uint64_t lane = 0;
    for (;;)
    {
        const std::uint64_t counter = seed
            ^ (epoch * 0xD1B54A32D192ED03ULL)
            ^ (role_record * 0x94D049BB133111EBULL)
            ^ (lane++ * 0x9E3779B97F4A7C15ULL);
        const std::uint64_t value = v3_splitmix64(counter);
        if (value >= threshold)
            return value % bound == 0;
    }
}

static Stockfish::Piece v3_piece_from_wire(Stockfish::u8 value)
{
    switch (value)
    {
    case AtomicData::ATOMIC_BIN_V2_EMPTY: return Stockfish::NO_PIECE;
    case AtomicData::ATOMIC_BIN_V2_WHITE_PAWN: return Stockfish::W_PAWN;
    case AtomicData::ATOMIC_BIN_V2_WHITE_KNIGHT: return Stockfish::W_KNIGHT;
    case AtomicData::ATOMIC_BIN_V2_WHITE_BISHOP: return Stockfish::W_BISHOP;
    case AtomicData::ATOMIC_BIN_V2_WHITE_ROOK: return Stockfish::W_ROOK;
    case AtomicData::ATOMIC_BIN_V2_WHITE_QUEEN: return Stockfish::W_QUEEN;
    case AtomicData::ATOMIC_BIN_V2_WHITE_KING: return Stockfish::W_KING;
    case AtomicData::ATOMIC_BIN_V2_BLACK_PAWN: return Stockfish::B_PAWN;
    case AtomicData::ATOMIC_BIN_V2_BLACK_KNIGHT: return Stockfish::B_KNIGHT;
    case AtomicData::ATOMIC_BIN_V2_BLACK_BISHOP: return Stockfish::B_BISHOP;
    case AtomicData::ATOMIC_BIN_V2_BLACK_ROOK: return Stockfish::B_ROOK;
    case AtomicData::ATOMIC_BIN_V2_BLACK_QUEEN: return Stockfish::B_QUEEN;
    case AtomicData::ATOMIC_BIN_V2_BLACK_KING: return Stockfish::B_KING;
    default: assert(false); return Stockfish::NO_PIECE;
    }
}

static AtomicV3::CapturePairSnapshot v3_snapshot(
    const AtomicData::AtomicBinV2DecodedRecord& record)
{
    AtomicV3::CapturePairSnapshot snapshot{};
    snapshot.sideToMove = record.fields.position.sideToMove == AtomicData::ATOMIC_BIN_V2_WHITE_TO_MOVE
        ? Stockfish::WHITE : Stockfish::BLACK;
    snapshot.epSquare = record.fields.position.enPassantSquare == AtomicData::AtomicBinV2NoSquare
        ? Stockfish::SQ_NONE : Stockfish::Square(record.fields.position.enPassantSquare);
    for (std::size_t square = 0; square < snapshot.board.size(); ++square)
        snapshot.board[square] = v3_piece_from_wire(record.fields.position.board[square]);
    return snapshot;
}

template<typename ExpectedIndex>
static void assert_v3_sparse_row(
    const AtomicV3SparseSliceViewV1& row,
    const ExpectedIndex& expected,
    std::uint32_t count,
    std::size_t row_index = 0)
{
    assert(row.width >= count);
    const std::size_t base = row_index * row.width;
    for (std::uint32_t index = 0; index < count; ++index)
    {
        assert(row.indices[base + index] == static_cast<std::int32_t>(expected(index)));
        assert(row.values[base + index] == 1.0F);
    }
    for (std::uint32_t index = count; index < row.width; ++index)
    {
        assert(row.indices[base + index] == -1);
        assert(row.values[base + index] == 0.0F);
    }
}

static void assert_v3_oracle_parity(
    const AtomicV3PerspectiveViewV1& view,
    const AtomicV3::FullRefreshEmission& emission,
    std::size_t row_index = 0)
{
    assert(view.ownKingSquares[row_index]
        == static_cast<std::int64_t>(emission.hm.orientation.ownKing));
    assert_v3_sparse_row(view.hm,
        [&](std::uint32_t index) { return emission.hm.features[index].trainingIndex; },
        emission.hm.size, row_index);
    assert_v3_sparse_row(view.capturePair,
        [&](std::uint32_t index) { return emission.capturePairs.features[index].localIndex; },
        emission.capturePairs.size, row_index);
    assert_v3_sparse_row(view.kingBlastEp,
        [&](std::uint32_t index) { return emission.kingBlastEp.features[index].localIndex; },
        emission.kingBlastEp.size, row_index);
    assert_v3_sparse_row(view.blastRing,
        [&](std::uint32_t index) { return emission.blastRing.features[index].localIndex; },
        emission.blastRing.size, row_index);
}

static std::set<std::filesystem::path> v3_snapshot_names()
{
    std::set<std::filesystem::path> names;
    std::error_code error;
    const auto directory = std::filesystem::temp_directory_path(error);
    if (error)
        return names;
    for (std::filesystem::directory_iterator iterator(directory, error), end;
         !error && iterator != end; iterator.increment(error))
    {
        const auto name = iterator->path().filename().string();
        if (name.rfind("atomic-bin-v2-reader-", 0) == 0 && iterator->path().extension() == ".tmp")
            names.insert(iterator->path());
    }
    return names;
}

static std::size_t v3_open_handle_count()
{
#ifdef _WIN32
    unsigned long count = 0;
    assert(GetProcessHandleCount(GetCurrentProcess(), &count));
    return count;
#else
    std::filesystem::path directory = "/proc/self/fd";
    if (!std::filesystem::exists(directory))
        directory = "/dev/fd";
    std::error_code error;
    std::size_t count = 0;
    for (std::filesystem::directory_iterator iterator(directory, error), end;
         !error && iterator != end; iterator.increment(error))
        ++count;
    assert(!error);
    return count;
#endif
}

#ifdef __linux__
static bool v3_process_has_snapshot_descriptor()
{
    std::error_code error;
    for (std::filesystem::directory_iterator iterator("/proc/self/fd", error), end;
         !error && iterator != end; iterator.increment(error))
    {
        std::error_code link_error;
        const auto target = std::filesystem::read_symlink(iterator->path(), link_error);
        if (!link_error
            && target.string().find("atomic-bin-v2-reader-") != std::string::npos)
            return true;
    }
    assert(!error);
    return false;
}

static void v3_assert_snapshot_descriptor_is_cloexec()
{
    assert(v3_process_has_snapshot_descriptor());
    const pid_t child = ::fork();
    assert(child >= 0);
    if (child == 0)
    {
        ::execl("/proc/self/exe", "training_data_loader_tests",
                "--assert-no-v3-snapshot-descriptor", static_cast<char*>(nullptr));
        ::_exit(127);
    }
    int status = 0;
    pid_t waited = -1;
    do
    {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    assert(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
#endif

static void test_atomic_v3_provider_validation_boundaries_and_oracle()
{
    Stockfish::initialize_atomic_core();
    TemporaryDirectory temporary("v3-validation");
    const auto firstSample = v3_sample(
        "7k/3p4/8/2N1p3/3P4/8/8/K7 w - - 0 1",
        Stockfish::Move(Stockfish::SQ_C5, Stockfish::SQ_D7), 123, -1);
    const auto secondSample = v3_sample(
        "7k/3p4/8/2N1p3/3P4/8/8/K7 b - - 0 1",
        Stockfish::Move(Stockfish::SQ_E5, Stockfish::SQ_D4), -456, 1);
    const auto first = create_v2_dataset(temporary.path, "v3-first", firstSample);
    const auto second = create_v2_dataset(temporary.path, "v3-second", secondSample);
    const V3ProviderInputs inputs({&first, &second});
    const auto config = inputs.config(3, 0, false);
    auto* stream = create_v3_provider(config);

    AtomicV3ProviderCursorV1 initial{};
    assert(atomic_v3_provider_committed_cursor(stream, &initial) == ATOMIC_V3_PROVIDER_OK);
    assert(initial.epoch == 0 && initial.manifestIndex == 0 && initial.recordIndex == 0);

    AtomicV3ProviderBatchV1* batch = nullptr;
    const auto view = fetch_v3_batch(stream, batch);
    assert(view.size == 2);
    assert(view.score[0] == 123.0F && view.score[1] == -456.0F);
    assert(view.outcome[0] == 0.0F && view.outcome[1] == 1.0F);
    assert(view.sideToMoveWhite[0] == 1.0F && view.sideToMoveWhite[1] == 0.0F);
    assert(view.cursorAfter.eof == 1 && view.cursorAfter.manifestIndex == 2);

    std::unique_ptr<AtomicData::AtomicBinV2DatasetReader> reader;
    assert(AtomicData::AtomicBinV2DatasetReader::open(first.manifest, reader));
    AtomicData::AtomicBinV2DecodedRecord decoded{};
    bool hasRecord = false;
    assert(reader->next(decoded, hasRecord) && hasRecord);
    const auto snapshot = v3_snapshot(decoded);
    AtomicV3::FullRefreshEmission white{};
    AtomicV3::FullRefreshEmission black{};
    assert(AtomicV3::emit_full_refresh(snapshot, Stockfish::WHITE, white)
        == AtomicV3::CapturePairError::None);
    assert(AtomicV3::emit_full_refresh(snapshot, Stockfish::BLACK, black)
        == AtomicV3::CapturePairError::None);
    assert_v3_oracle_parity(view.white, white);
    assert_v3_oracle_parity(view.black, black);
    assert(view.pieceCounts[0] == white.hm.size);
    assert(view.bucketIndices[0] == white.hm.networkBucket);

    AtomicV3ProviderCursorV1 beforeCommit{};
    assert(atomic_v3_provider_committed_cursor(stream, &beforeCommit) == ATOMIC_V3_PROVIDER_OK);
    assert(beforeCommit.manifestIndex == 0 && beforeCommit.eof == 0);
    assert(atomic_v3_provider_commit(stream) == ATOMIC_V3_PROVIDER_OK);
    AtomicV3ProviderCursorV1 committed{};
    assert(atomic_v3_provider_committed_cursor(stream, &committed) == ATOMIC_V3_PROVIDER_OK);
    assert(committed.eof == 1 && committed.acceptedSamples == 2
        && committed.nextBatchSequence == 1);
    atomic_v3_provider_destroy_batch(batch);
    batch = nullptr;
    assert(atomic_v3_provider_fetch(stream, &batch) == ATOMIC_V3_PROVIDER_EOF);
    atomic_v3_provider_destroy(stream);

    const auto resumeConfig = inputs.config(3, 0, false, &committed);
    auto* resumed = create_v3_provider(resumeConfig);
    assert(atomic_v3_provider_fetch(resumed, &batch) == ATOMIC_V3_PROVIDER_EOF);
    atomic_v3_provider_destroy(resumed);
}

static void test_atomic_v3_provider_exact_multirow_arena_parity()
{
    TemporaryDirectory temporary("v3-arena-parity");
    std::vector<AtomicData::TrainingDataSample> samples = {
        v3_sample("7k/8/8/8/8/8/4P3/K7 w - - 0 1",
            Stockfish::Move(Stockfish::SQ_E2, Stockfish::SQ_E3), 11, -1),
        v3_sample("7k/4p3/8/8/8/8/8/K7 b - - 0 1",
            Stockfish::Move(Stockfish::SQ_E7, Stockfish::SQ_E6), -22, 1),
        v3_sample("7k/3p4/8/2N1p3/3P4/8/8/K7 w - - 0 1",
            Stockfish::Move(Stockfish::SQ_C5, Stockfish::SQ_D7), 33, 0),
        v3_sample("7k/3p4/8/2N1p3/3P4/8/8/K7 b - - 0 1",
            Stockfish::Move(Stockfish::SQ_E5, Stockfish::SQ_D4), -44, -1),
        v3_sample("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            Stockfish::Move(Stockfish::SQ_E2, Stockfish::SQ_E4), 55, 1),
        v3_sample("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            Stockfish::Move(Stockfish::SQ_E7, Stockfish::SQ_E5), -66, 0),
        v3_sample("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
            Stockfish::Move::make<Stockfish::EN_PASSANT>(Stockfish::SQ_E5, Stockfish::SQ_D6),
            77, -1),
        v3_sample("7k/8/8/8/3Q4/8/8/K7 w - - 0 1",
            Stockfish::Move(Stockfish::SQ_D4, Stockfish::SQ_D5), -88, 1),
    };
    const auto data = create_v2_multishard_dataset(
        temporary.path, "v3-arena-parity",
        {{samples.begin(), samples.begin() + 4}, {samples.begin() + 4, samples.end()}});
    V2Dataset adapter;
    adapter.manifest = data.manifest;
    adapter.shard = data.shards.front();
    adapter.metadata = data.metadata;
    const V3ProviderInputs inputs({&adapter});
    auto* stream = create_v3_provider(inputs.config(8, 0, false));
    AtomicV3ProviderBatchV1* batch = nullptr;
    const auto view = fetch_v3_batch(stream, batch);
    assert(view.size == samples.size());

    std::array<std::uint32_t, 4> white_widths{};
    std::array<std::uint32_t, 4> black_widths{};
    for (std::size_t row = 0; row < samples.size(); ++row)
    {
        AtomicData::AtomicBinV2Record wire{};
        assert(AtomicData::encode_atomic_bin_v2(samples[row], wire));
        AtomicData::AtomicBinV2DecodedRecord decoded{};
        assert(AtomicData::decode_atomic_bin_v2_record_structural(wire, decoded.fields));
        const auto snapshot = v3_snapshot(decoded);
        AtomicV3::FullRefreshEmission white{};
        AtomicV3::FullRefreshEmission black{};
        assert(AtomicV3::emit_full_refresh(snapshot, Stockfish::WHITE, white)
            == AtomicV3::CapturePairError::None);
        assert(AtomicV3::emit_full_refresh(snapshot, Stockfish::BLACK, black)
            == AtomicV3::CapturePairError::None);
        assert_v3_oracle_parity(view.white, white, row);
        assert_v3_oracle_parity(view.black, black, row);
        white_widths[0] = std::max(white_widths[0], white.hm.size);
        white_widths[1] = std::max(white_widths[1], white.capturePairs.size);
        white_widths[2] = std::max(white_widths[2], white.kingBlastEp.size);
        white_widths[3] = std::max(white_widths[3], white.blastRing.size);
        black_widths[0] = std::max(black_widths[0], black.hm.size);
        black_widths[1] = std::max(black_widths[1], black.capturePairs.size);
        black_widths[2] = std::max(black_widths[2], black.kingBlastEp.size);
        black_widths[3] = std::max(black_widths[3], black.blastRing.size);
        assert(view.sideToMoveWhite[row]
            == (snapshot.sideToMove == Stockfish::WHITE ? 1.0F : 0.0F));
        assert(view.pieceCounts[row] == static_cast<std::int64_t>(white.hm.size));
        assert(view.outcome[row] == (float(samples[row].result) + 1.0F) / 2.0F);
        assert(view.score[row] == static_cast<float>(samples[row].score));
        assert(view.bucketIndices[row] == static_cast<std::int64_t>(white.hm.networkBucket));
    }
    assert(view.white.hm.width == white_widths[0]
        && view.white.capturePair.width == white_widths[1]
        && view.white.kingBlastEp.width == white_widths[2]
        && view.white.blastRing.width == white_widths[3]);
    assert(view.black.hm.width == black_widths[0]
        && view.black.capturePair.width == black_widths[1]
        && view.black.kingBlastEp.width == black_widths[2]
        && view.black.blastRing.width == black_widths[3]);
    assert(view.cursorAfter.eof == 1 && view.cursorAfter.epoch == 0
        && view.cursorAfter.manifestIndex == 1 && view.cursorAfter.recordIndex == 0
        && view.cursorAfter.acceptedSamples == samples.size()
        && view.cursorAfter.nextBatchSequence == 1);
    assert(atomic_v3_provider_commit(stream) == ATOMIC_V3_PROVIDER_OK);
    AtomicV3ProviderCursorV1 committed{};
    assert(atomic_v3_provider_committed_cursor(stream, &committed) == ATOMIC_V3_PROVIDER_OK);
    assert(std::memcmp(&committed, &view.cursorAfter, sizeof(committed)) == 0);
    atomic_v3_provider_destroy_batch(batch);
    atomic_v3_provider_destroy(stream);
}

static void test_atomic_v3_provider_cyclic_cursor_and_exact_resume()
{
    TemporaryDirectory temporary("v3-resume");
    const auto first = create_v2_dataset(temporary.path, "v3-a", v3_sample(
        "7k/8/8/8/8/8/4P3/K7 w - - 0 1",
        Stockfish::Move(Stockfish::SQ_E2, Stockfish::SQ_E3), 101, -1));
    const auto second = create_v2_dataset(temporary.path, "v3-b", v3_sample(
        "7k/4p3/8/8/8/8/8/K7 b - - 0 1",
        Stockfish::Move(Stockfish::SQ_E7, Stockfish::SQ_E6), 202, 1));
    const V3ProviderInputs inputs({&first, &second});
    auto* stream = create_v3_provider(inputs.config(3, 0, true));
    AtomicV3ProviderBatchV1* firstBatch = nullptr;
    auto firstView = fetch_v3_batch(stream, firstBatch);
    assert(firstView.size == 3);
    assert(firstView.score[0] == 101.0F && firstView.score[1] == 202.0F
        && firstView.score[2] == 101.0F);
    assert(firstView.cursorAfter.epoch == 1 && firstView.cursorAfter.manifestIndex == 1);
    assert(atomic_v3_provider_commit(stream) == ATOMIC_V3_PROVIDER_OK);
    AtomicV3ProviderCursorV1 committed{};
    assert(atomic_v3_provider_committed_cursor(stream, &committed) == ATOMIC_V3_PROVIDER_OK);
    atomic_v3_provider_destroy_batch(firstBatch);

    AtomicV3ProviderBatchV1* uncommittedBatch = nullptr;
    auto uncommittedView = fetch_v3_batch(stream, uncommittedBatch);
    const std::vector<float> expected(
        uncommittedView.score, uncommittedView.score + uncommittedView.size);
    AtomicV3ProviderCursorV1 stillCommitted{};
    assert(atomic_v3_provider_committed_cursor(stream, &stillCommitted) == ATOMIC_V3_PROVIDER_OK);
    assert(stillCommitted.epoch == committed.epoch
        && stillCommitted.manifestIndex == committed.manifestIndex
        && stillCommitted.recordIndex == committed.recordIndex
        && stillCommitted.acceptedSamples == committed.acceptedSamples);
    atomic_v3_provider_destroy_batch(uncommittedBatch);
    atomic_v3_provider_destroy(stream);

    auto* resumed = create_v3_provider(inputs.config(3, 0, true, &committed));
    AtomicV3ProviderBatchV1* resumedBatch = nullptr;
    const auto resumedView = fetch_v3_batch(resumed, resumedBatch);
    assert(std::vector<float>(resumedView.score, resumedView.score + resumedView.size) == expected);
    atomic_v3_provider_destroy_batch(resumedBatch);
    atomic_v3_provider_destroy(resumed);
}

static void test_atomic_v3_provider_cursor_and_counter_boundaries()
{
    TemporaryDirectory temporary("v3-cursor-bounds");
    const auto dataset = create_v2_dataset(temporary.path, "v3-cursor-bounds", v3_sample(
        "7k/8/8/8/8/8/4P3/K7 w - - 0 1",
        Stockfish::Move(Stockfish::SQ_E2, Stockfish::SQ_E3), 707, 0));
    const V3ProviderInputs inputs({&dataset});
    const auto base_config = inputs.config(1, 0, false);
    auto* binding_stream = create_v3_provider(base_config);
    AtomicV3ProviderCursorV1 initial{};
    assert(atomic_v3_provider_committed_cursor(binding_stream, &initial)
        == ATOMIC_V3_PROVIDER_OK);
    atomic_v3_provider_destroy(binding_stream);

    AtomicV3ProviderCursorV1 invalid_eof = initial;
    invalid_eof.eof = 2;
    auto invalid_config = inputs.config(1, 0, false, &invalid_eof);
    AtomicV3ProviderStreamV1* rejected = nullptr;
    assert(atomic_v3_provider_create(&invalid_config, &rejected)
        == ATOMIC_V3_PROVIDER_ERROR);
    assert(rejected == nullptr);
    assert(std::string(atomic_v3_provider_creation_error()).find("cursor")
        != std::string::npos);

    auto assert_overflow_fails_without_committing = [&](AtomicV3ProviderCursorV1 cursor,
                                                         const char* expected_error) {
        const std::size_t handles_before = v3_open_handle_count();
        auto config = inputs.config(1, 0, false, &cursor);
        auto* stream = create_v3_provider(config);
        AtomicV3ProviderBatchV1* batch = nullptr;
        assert(atomic_v3_provider_fetch(stream, &batch) == ATOMIC_V3_PROVIDER_ERROR);
        assert(batch == nullptr);
        assert(std::string(atomic_v3_provider_error(stream)).find(expected_error)
            != std::string::npos);
        AtomicV3ProviderCursorV1 committed{};
        assert(atomic_v3_provider_committed_cursor(stream, &committed)
            == ATOMIC_V3_PROVIDER_OK);
        assert(std::memcmp(&committed, &cursor, sizeof(cursor)) == 0);
        assert(v3_open_handle_count() == handles_before);
        atomic_v3_provider_destroy(stream);
        assert(v3_open_handle_count() == handles_before);
    };

    AtomicV3ProviderCursorV1 accepted_overflow = initial;
    accepted_overflow.acceptedSamples = std::numeric_limits<std::uint64_t>::max();
    assert_overflow_fails_without_committing(accepted_overflow, "accepted-sample counter overflow");

    AtomicV3ProviderCursorV1 sequence_overflow = initial;
    sequence_overflow.nextBatchSequence = std::numeric_limits<std::uint64_t>::max();
    assert_overflow_fails_without_committing(sequence_overflow, "batch-sequence counter overflow");
}

static void test_atomic_v3_provider_skip_precedes_decode_and_resume_seeks_directly()
{
    TemporaryDirectory temporary("v3-fast-reader");
    std::vector<AtomicData::TrainingDataSample> samples;
    for (int index = 0; index < 32; ++index)
        samples.push_back(v3_sample(
            "7k/8/8/8/8/8/4P3/K7 w - - 0 1",
            Stockfish::Move(Stockfish::SQ_E2, Stockfish::SQ_E3),
            1000 + index,
            index % 3 - 1));

    const auto resume_data = create_v2_multishard_dataset(
        temporary.path,
        "v3-direct-resume",
        {{samples.begin(), samples.begin() + 24}, {samples.begin() + 24, samples.end()}});
    V2Dataset resume_adapter;
    resume_adapter.manifest = resume_data.manifest;
    resume_adapter.shard = resume_data.shards.front();
    resume_adapter.metadata = resume_data.metadata;
    const V3ProviderInputs resume_inputs({&resume_adapter});

    auto* uninterrupted = create_v3_provider(resume_inputs.config(2, 3, true));
    AtomicV3ProviderBatchV1* first_batch = nullptr;
    (void) fetch_v3_batch(uninterrupted, first_batch);
    assert(atomic_v3_provider_commit(uninterrupted) == ATOMIC_V3_PROVIDER_OK);
    AtomicV3ProviderCursorV1 committed{};
    assert(atomic_v3_provider_committed_cursor(uninterrupted, &committed)
        == ATOMIC_V3_PROVIDER_OK);
    assert(committed.manifestIndex == 0 && committed.recordIndex > 0
        && committed.recordIndex < resume_data.metadata.records);
    atomic_v3_provider_destroy_batch(first_batch);

    AtomicV3ProviderBatchV1* expected_batch = nullptr;
    const auto expected_view = fetch_v3_batch(uninterrupted, expected_batch);
    const std::vector<float> expected_scores(
        expected_view.score,
        expected_view.score + expected_view.size);
    atomic_v3_provider_destroy_batch(expected_batch);
    atomic_v3_provider_destroy(uninterrupted);

    auto* resumed = create_v3_provider(resume_inputs.config(2, 3, true, &committed));
    AtomicV3ProviderBatchV1* actual_batch = nullptr;
    const auto actual_view = fetch_v3_batch(resumed, actual_batch);
    assert(std::vector<float>(actual_view.score, actual_view.score + actual_view.size)
        == expected_scores);
    atomic_v3_provider_destroy_batch(actual_batch);
    atomic_v3_provider_destroy(resumed);

    // Corrupt a record which the deterministic selector rejects, then bind the
    // new bytes into a canonical manifest. The skip-3 stream must never decode
    // or full-refresh that raw record; the no-skip control must reject it.
    std::vector<AtomicData::TrainingDataSample> corrupt_samples(
        samples.begin(), samples.begin() + 16);
    auto corrupt_data = create_v2_multishard_dataset(
        temporary.path,
        "v3-skip-before-decode",
        {{corrupt_samples.begin(), corrupt_samples.begin() + 12},
         {corrupt_samples.begin() + 12, corrupt_samples.end()}});
    std::size_t corrupt_index = 0;
    while (corrupt_index < 12
        && v3_keep_record(20260716, 0, corrupt_index, 3))
        ++corrupt_index;
    assert(corrupt_index < 12);
    {
        std::fstream shard(
            corrupt_data.shards.front(),
            std::ios::binary | std::ios::in | std::ios::out);
        assert(shard);
        const std::uint64_t offset = AtomicData::AtomicBinV2HeaderSize
            + corrupt_index * AtomicData::AtomicBinV2RecordSize;
        shard.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
        AtomicData::AtomicBinV2Record invalid{};
        shard.write(
            reinterpret_cast<const char*>(invalid.data()),
            static_cast<std::streamsize>(invalid.size()));
        assert(shard);
    }
    std::string rebound_sha256;
    Stockfish::u64 rebound_size = 0;
    assert(AtomicData::sha256_file(
        corrupt_data.shards.front(), rebound_sha256, rebound_size));
    assert(rebound_size == corrupt_data.metadata.shards.front().bytes);
    corrupt_data.metadata.shards.front().sha256 = std::move(rebound_sha256);
    assert(write_v2_manifest(corrupt_data.metadata));

    V2Dataset corrupt_adapter;
    corrupt_adapter.manifest = corrupt_data.manifest;
    corrupt_adapter.shard = corrupt_data.shards.front();
    corrupt_adapter.metadata = corrupt_data.metadata;
    const V3ProviderInputs corrupt_inputs({&corrupt_adapter});
    std::vector<float> retained_scores;
    for (std::size_t index = 0; index < corrupt_samples.size(); ++index)
        if (v3_keep_record(20260716, 0, index, 3))
            retained_scores.push_back(static_cast<float>(corrupt_samples[index].score));
    assert(!retained_scores.empty());

    // A separate binding-compatible cursor starts after a structurally invalid
    // record which the selector *keeps*. Direct seek must succeed, while any
    // implementation which replays the prefix would decode that record and
    // fail. The complete cursorAfter value is part of the proof.
    std::vector<AtomicData::TrainingDataSample> direct_samples(
        samples.begin(), samples.begin() + 24);
    auto direct_data = create_v2_multishard_dataset(
        temporary.path, "v3-kept-invalid-prefix",
        {{direct_samples.begin(), direct_samples.begin() + 16},
         {direct_samples.begin() + 16, direct_samples.end()}});
    std::size_t kept_invalid_index = 0;
    while (kept_invalid_index < 8
        && !v3_keep_record(20260716, 0, kept_invalid_index, 3))
        ++kept_invalid_index;
    assert(kept_invalid_index < 8);
    {
        std::fstream shard(direct_data.shards.front(),
            std::ios::binary | std::ios::in | std::ios::out);
        assert(shard);
        const std::uint64_t offset = AtomicData::AtomicBinV2HeaderSize
            + kept_invalid_index * AtomicData::AtomicBinV2RecordSize;
        shard.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
        AtomicData::AtomicBinV2Record invalid{};
        shard.write(reinterpret_cast<const char*>(invalid.data()),
            static_cast<std::streamsize>(invalid.size()));
        assert(shard);
    }
    std::string direct_sha256;
    Stockfish::u64 direct_size = 0;
    assert(AtomicData::sha256_file(
        direct_data.shards.front(), direct_sha256, direct_size));
    assert(direct_size == direct_data.metadata.shards.front().bytes);
    direct_data.metadata.shards.front().sha256 = std::move(direct_sha256);
    assert(write_v2_manifest(direct_data.metadata));

    V2Dataset direct_adapter;
    direct_adapter.manifest = direct_data.manifest;
    direct_adapter.shard = direct_data.shards.front();
    direct_adapter.metadata = direct_data.metadata;
    const V3ProviderInputs direct_inputs({&direct_adapter});
    auto* binding_stream = create_v3_provider(direct_inputs.config(2, 3, false));
    AtomicV3ProviderCursorV1 direct_cursor{};
    assert(atomic_v3_provider_committed_cursor(binding_stream, &direct_cursor)
        == ATOMIC_V3_PROVIDER_OK);
    atomic_v3_provider_destroy(binding_stream);

    // Control: from record zero the kept invalid prefix is actually rejected.
    auto* replay_control = create_v3_provider(direct_inputs.config(2, 3, false));
    AtomicV3ProviderBatchV1* replay_batch = nullptr;
    assert(atomic_v3_provider_fetch(replay_control, &replay_batch)
        == ATOMIC_V3_PROVIDER_ERROR);
    assert(replay_batch == nullptr);
    atomic_v3_provider_destroy(replay_control);

    direct_cursor.recordIndex = kept_invalid_index + 1;
    direct_cursor.acceptedSamples = 0;
    for (std::size_t index = 0; index <= kept_invalid_index; ++index)
        direct_cursor.acceptedSamples +=
            v3_keep_record(20260716, 0, index, 3) ? 1U : 0U;
    direct_cursor.nextBatchSequence = 7;
    std::vector<float> direct_expected_scores;
    std::size_t record_after_batch = direct_cursor.recordIndex;
    while (record_after_batch < direct_samples.size()
        && direct_expected_scores.size() != 2)
    {
        if (v3_keep_record(20260716, 0, record_after_batch, 3))
            direct_expected_scores.push_back(
                static_cast<float>(direct_samples[record_after_batch].score));
        ++record_after_batch;
    }
    assert(direct_expected_scores.size() == 2
        && record_after_batch < direct_samples.size());

    auto* direct_resumed = create_v3_provider(
        direct_inputs.config(2, 3, false, &direct_cursor));
    AtomicV3ProviderBatchV1* direct_resumed_batch = nullptr;
    const auto direct_resumed_view = fetch_v3_batch(direct_resumed, direct_resumed_batch);
    assert(std::vector<float>(direct_resumed_view.score,
        direct_resumed_view.score + direct_resumed_view.size) == direct_expected_scores);
    AtomicV3ProviderCursorV1 expected_cursor = direct_cursor;
    expected_cursor.recordIndex = record_after_batch;
    expected_cursor.acceptedSamples += 2;
    ++expected_cursor.nextBatchSequence;
    assert(std::memcmp(&direct_resumed_view.cursorAfter, &expected_cursor,
        sizeof(expected_cursor)) == 0);
    atomic_v3_provider_destroy_batch(direct_resumed_batch);
    atomic_v3_provider_destroy(direct_resumed);

    auto* skipped = create_v3_provider(
        corrupt_inputs.config(static_cast<std::uint32_t>(retained_scores.size()), 3, false));
    AtomicV3ProviderBatchV1* skipped_batch = nullptr;
    const auto skipped_view = fetch_v3_batch(skipped, skipped_batch);
    assert(std::vector<float>(skipped_view.score, skipped_view.score + skipped_view.size)
        == retained_scores);
    atomic_v3_provider_destroy_batch(skipped_batch);
    atomic_v3_provider_destroy(skipped);

    const std::size_t decode_error_handles = v3_open_handle_count();
    const auto decode_error_snapshots = v3_snapshot_names();
    auto* no_skip = create_v3_provider(corrupt_inputs.config(16, 0, false));
    AtomicV3ProviderBatchV1* rejected_batch = nullptr;
    assert(atomic_v3_provider_fetch(no_skip, &rejected_batch) == ATOMIC_V3_PROVIDER_ERROR);
    assert(rejected_batch == nullptr);
    const std::string rejection = atomic_v3_provider_error(no_skip);
    assert(rejection.find("retained Atomic V3 record") != std::string::npos
        || rejection.find("full refresh rejected") != std::string::npos);
    assert(v3_open_handle_count() == decode_error_handles);
    assert(v3_snapshot_names() == decode_error_snapshots);
    atomic_v3_provider_destroy(no_skip);
    assert(v3_open_handle_count() == decode_error_handles);
    assert(v3_snapshot_names() == decode_error_snapshots);
}

static void test_atomic_v3_provider_snapshot_freezes_unread_source_bytes()
{
    TemporaryDirectory temporary("v3-snapshot-freeze");
    std::vector<AtomicData::TrainingDataSample> first_shard;
    first_shard.reserve(4097);
    for (int index = 0; index < 4097; ++index)
        first_shard.push_back(v3_sample(
            "7k/8/8/8/8/8/4P3/K7 w - - 0 1",
            Stockfish::Move(Stockfish::SQ_E2, Stockfish::SQ_E3),
            2000 + index,
            index % 3 - 1));
    std::vector<AtomicData::TrainingDataSample> second_shard = {v3_sample(
        "7k/4p3/8/8/8/8/8/K7 b - - 0 1",
        Stockfish::Move(Stockfish::SQ_E7, Stockfish::SQ_E6), 9000, 1)};
    const auto data = create_v2_multishard_dataset(
        temporary.path, "v3-snapshot-freeze", {first_shard, second_shard});
    V2Dataset adapter;
    adapter.manifest = data.manifest;
    adapter.shard = data.shards.front();
    adapter.metadata = data.metadata;
    const V3ProviderInputs inputs({&adapter});
    const auto snapshots_before = v3_snapshot_names();

    auto* stream = create_v3_provider(inputs.config(4096, 0, false));
    AtomicV3ProviderBatchV1* first_batch = nullptr;
    const auto first_view = fetch_v3_batch(stream, first_batch);
    assert(first_view.size == 4096 && first_view.score[4095] == 6095.0F);
    atomic_v3_provider_destroy_batch(first_batch);

    // This record has not entered the 4,096-record snapshot buffer yet.  A
    // source-backed reader would observe the replacement at its next refill;
    // the private authenticated snapshot must preserve the original bytes.
    auto replacement = first_shard.back();
    replacement.score = -7777;
    AtomicData::AtomicBinV2Record replacement_wire{};
    assert(AtomicData::encode_atomic_bin_v2(replacement, replacement_wire));
    {
        std::fstream shard(data.shards.front(),
            std::ios::binary | std::ios::in | std::ios::out);
        assert(shard);
        const std::uint64_t offset = AtomicData::AtomicBinV2HeaderSize
            + 4096ULL * AtomicData::AtomicBinV2RecordSize;
        shard.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
        shard.write(reinterpret_cast<const char*>(replacement_wire.data()),
            static_cast<std::streamsize>(replacement_wire.size()));
        assert(shard);
    }

    AtomicV3ProviderBatchV1* remainder_batch = nullptr;
    const auto remainder_view = fetch_v3_batch(stream, remainder_batch);
    assert(remainder_view.size == 2);
    assert(remainder_view.score[0] == 6096.0F);
    assert(remainder_view.score[1] == 9000.0F);
    atomic_v3_provider_destroy_batch(remainder_batch);
    atomic_v3_provider_destroy(stream);
    assert(v3_snapshot_names() == snapshots_before);

    // A fresh provider has no authority to reuse the old snapshot and must
    // reject the now-mutated source against the manifest SHA-256.
    auto* fresh = create_v3_provider(inputs.config(1, 0, false));
    AtomicV3ProviderBatchV1* rejected = nullptr;
    assert(atomic_v3_provider_fetch(fresh, &rejected) == ATOMIC_V3_PROVIDER_ERROR);
    assert(rejected == nullptr);
    assert(std::string(atomic_v3_provider_error(fresh)).find("SHA-256")
        != std::string::npos);
    atomic_v3_provider_destroy(fresh);
    assert(v3_snapshot_names() == snapshots_before);
}

static void test_atomic_v3_provider_rejects_replaced_and_duplicate_shards()
{
    TemporaryDirectory temporary("v3-identities");
    const auto sample = v3_sample(
        "7k/8/8/8/8/8/4P3/K7 w - - 0 1",
        Stockfish::Move(Stockfish::SQ_E2, Stockfish::SQ_E3), 321, 0);

    auto repeated_path = create_v2_multishard_dataset(
        temporary.path, "v3-repeat-path", {{sample}, {sample}});
    repeated_path.metadata.shards[1].path = repeated_path.metadata.shards[0].path;
    repeated_path.metadata.shards[1].sha256 = repeated_path.metadata.shards[0].sha256;
    assert(write_v2_manifest(repeated_path.metadata));
    V2Dataset repeated_path_adapter;
    repeated_path_adapter.manifest = repeated_path.manifest;
    repeated_path_adapter.shard = repeated_path.shards.front();
    repeated_path_adapter.metadata = repeated_path.metadata;
    const V3ProviderInputs repeated_path_inputs({&repeated_path_adapter});
    AtomicV3ProviderStreamV1* rejected = nullptr;
    auto config = repeated_path_inputs.config(1, 0, false);
    assert(atomic_v3_provider_create(&config, &rejected) == ATOMIC_V3_PROVIDER_ERROR);
    assert(rejected == nullptr);
    assert(std::string(atomic_v3_provider_creation_error()).find("pathname")
        != std::string::npos);

    auto repeated_identity = create_v2_multishard_dataset(
        temporary.path, "v3-repeat-identity", {{sample}, {sample}});
    assert(std::filesystem::remove(repeated_identity.shards[1]));
    std::filesystem::create_hard_link(
        repeated_identity.shards[0], repeated_identity.shards[1]);
    assert(write_v2_manifest(repeated_identity.metadata));
    V2Dataset repeated_identity_adapter;
    repeated_identity_adapter.manifest = repeated_identity.manifest;
    repeated_identity_adapter.shard = repeated_identity.shards.front();
    repeated_identity_adapter.metadata = repeated_identity.metadata;
    const V3ProviderInputs repeated_identity_inputs({&repeated_identity_adapter});
    config = repeated_identity_inputs.config(1, 0, false);
    assert(atomic_v3_provider_create(&config, &rejected) == ATOMIC_V3_PROVIDER_ERROR);
    assert(rejected == nullptr);
    assert(std::string(atomic_v3_provider_creation_error()).find("filesystem identity")
        != std::string::npos);

    // Reopening a cyclic stream must preserve the source identity established
    // at provider creation, even when replacement bytes have the same hash.
    auto reopen = create_v2_multishard_dataset(
        temporary.path, "v3-reopen", {{sample, sample}, {sample, sample}});
    V2Dataset reopen_adapter;
    reopen_adapter.manifest = reopen.manifest;
    reopen_adapter.shard = reopen.shards.front();
    reopen_adapter.metadata = reopen.metadata;
    const V3ProviderInputs reopen_inputs({&reopen_adapter});
    auto* cyclic = create_v3_provider(reopen_inputs.config(2, 0, true));
    AtomicV3ProviderBatchV1* batch = nullptr;
    (void) fetch_v3_batch(cyclic, batch); // Consumes and closes shard zero.
    atomic_v3_provider_destroy_batch(batch);
    const auto replacement = temporary.path / "v3-reopen-replacement.atbin";
    assert(std::filesystem::copy_file(
        reopen.shards.front(), replacement, std::filesystem::copy_options::none));
    assert(std::filesystem::remove(reopen.shards.front()));
    std::filesystem::rename(replacement, reopen.shards.front());
    (void) fetch_v3_batch(cyclic, batch); // Completes shard one and epoch zero.
    atomic_v3_provider_destroy_batch(batch);
    batch = nullptr;
    assert(atomic_v3_provider_fetch(cyclic, &batch) == ATOMIC_V3_PROVIDER_ERROR);
    assert(batch == nullptr);
    assert(std::string(atomic_v3_provider_error(cyclic)).find("identity")
        != std::string::npos);
    atomic_v3_provider_destroy(cyclic);
}

static void test_atomic_v3_provider_many_shards_keep_resources_bounded()
{
    TemporaryDirectory temporary("v3-many-shards");
    const auto sample = v3_sample(
        "7k/8/8/8/8/8/4P3/K7 w - - 0 1",
        Stockfish::Move(Stockfish::SQ_E2, Stockfish::SQ_E3), 654, 0);
    std::vector<std::vector<AtomicData::TrainingDataSample>> shards(
        512, std::vector<AtomicData::TrainingDataSample>{sample, sample});
    const auto data = create_v2_multishard_dataset(
        temporary.path, "v3-many-shards", shards);
    V2Dataset adapter;
    adapter.manifest = data.manifest;
    adapter.shard = data.shards.front();
    adapter.metadata = data.metadata;
    const V3ProviderInputs inputs({&adapter});
    const auto snapshots_before = v3_snapshot_names();
    const std::size_t handles_before = v3_open_handle_count();

    auto* stream = create_v3_provider(inputs.config(3, 0, false));
    const std::size_t handles_after_create = v3_open_handle_count();
    assert(handles_after_create <= handles_before + 2);
    AtomicV3ProviderBatchV1* batch = nullptr;
    const auto view = fetch_v3_batch(stream, batch);
    assert(view.size == 3 && view.score[0] == 654.0F);
    const std::size_t handles_with_snapshot = v3_open_handle_count();
    assert(handles_with_snapshot <= handles_before + 4);
    const auto snapshots_during = v3_snapshot_names();
    std::size_t new_snapshots = 0;
    for (const auto& path : snapshots_during)
        new_snapshots += snapshots_before.count(path) == 0 ? 1U : 0U;
    assert(new_snapshots <= 1);
#ifdef __linux__
    v3_assert_snapshot_descriptor_is_cloexec();
#endif

    std::uint64_t delivered = view.size;
    atomic_v3_provider_destroy_batch(batch);
    for (;;)
    {
        batch = nullptr;
        const auto status = atomic_v3_provider_fetch(stream, &batch);
        if (status == ATOMIC_V3_PROVIDER_EOF)
            break;
        assert(status == ATOMIC_V3_PROVIDER_OK && batch != nullptr);
        AtomicV3BatchViewV1 next_view{};
        assert(atomic_v3_provider_batch_view(batch, &next_view) == ATOMIC_V3_PROVIDER_OK);
        delivered += next_view.size;
        atomic_v3_provider_destroy_batch(batch);
    }
    assert(delivered == 1024);
    assert(v3_open_handle_count() == handles_before);
    assert(v3_snapshot_names() == snapshots_before);
    atomic_v3_provider_destroy(stream);
    assert(v3_open_handle_count() == handles_before);
    assert(v3_snapshot_names() == snapshots_before);
}

static void test_atomic_v3_provider_manifest_tamper_and_snapshot_cleanup()
{
    const auto before = v3_snapshot_names();
    const std::size_t handles_before = v3_open_handle_count();
    TemporaryDirectory temporary("v3-cleanup");
    std::vector<AtomicData::TrainingDataSample> samples;
    for (int index = 0; index < 4; ++index)
        samples.push_back(v3_sample(
            "7k/8/8/8/8/8/4P3/K7 w - - 0 1",
            Stockfish::Move(Stockfish::SQ_E2, Stockfish::SQ_E3), 10 + index, index % 3 - 1));
    const auto multi = create_v2_multishard_dataset(
        temporary.path, "v3-multi", {{samples.begin(), samples.begin() + 3}, {samples.back()}});

    // Use a one-record descriptor helper for creation-time manifest tamper.
    const auto one = create_v2_dataset(temporary.path, "v3-one", samples.front());
    V3ProviderInputs tamperedInputs({&one});
    {
        std::ofstream output(one.manifest, std::ios::binary | std::ios::app);
        output.put(' ');
    }
    AtomicV3ProviderStreamV1* rejected = nullptr;
    const auto tamperedConfig = tamperedInputs.config(1, 0, false);
    assert(atomic_v3_provider_create(&tamperedConfig, &rejected) == ATOMIC_V3_PROVIDER_ERROR);
    assert(rejected == nullptr);
    assert(std::string(atomic_v3_provider_creation_error()).find("changed") != std::string::npos);

    // Build the equivalent descriptor directly for the multi-shard manifest.
    V2Dataset multiAdapter;
    multiAdapter.manifest = multi.manifest;
    multiAdapter.shard = multi.shards.front();
    multiAdapter.metadata = multi.metadata;
    const V3ProviderInputs validInputs({&multiAdapter});
    auto* stream = create_v3_provider(validInputs.config(1, 0, false));
    AtomicV3ProviderBatchV1* batch = nullptr;
    (void) fetch_v3_batch(stream, batch); // Leaves the first authenticated shard snapshot live.
    atomic_v3_provider_destroy_batch(batch);
    atomic_v3_provider_destroy(stream);
    assert(v3_snapshot_names() == before);

    auto* corruptStream = create_v3_provider(validInputs.config(1, 0, false));
    {
        std::fstream shard(multi.shards.front(), std::ios::binary | std::ios::in | std::ios::out);
        assert(shard);
        shard.seekg(-1, std::ios::end);
        char value = 0;
        shard.read(&value, 1);
        shard.clear();
        shard.seekp(-1, std::ios::end);
        shard.put(static_cast<char>(value ^ 1));
    }
    batch = nullptr;
    assert(atomic_v3_provider_fetch(corruptStream, &batch) == ATOMIC_V3_PROVIDER_ERROR);
    assert(batch == nullptr);
    assert(std::string(atomic_v3_provider_error(corruptStream)).find("SHA-256")
        != std::string::npos);
    atomic_v3_provider_destroy(corruptStream);
    assert(v3_snapshot_names() == before);
    assert(v3_open_handle_count() == handles_before);
}

int main(int argc, char** argv)
{
#ifdef __linux__
    if (argc == 2 && std::strcmp(argv[1], "--assert-no-v3-snapshot-descriptor") == 0)
        return v3_process_has_snapshot_descriptor() ? 1 : 0;
#else
    (void) argc;
    (void) argv;
#endif
    static_assert(sizeof(bin::nodchip::PackedSfenValue) == 72, "legacy ABI must stay 72 bytes");
    test_smart_filter();
    test_seeded_random_skipping();
    test_c_api_validation();
    test_atomic_training_data_schema_handshake();
    test_atomic_bin_v2_manifest_only_roundtrip();
    test_widened_internal_position_boundaries();
    test_v2_adapter_defensively_preserves_missing_king_sentinel();
    test_atomic_bin_v2_atomic960_and_stm_semantics();
    test_atomic_bin_v2_multishard_eof_partial_and_determinism();
    test_atomic_bin_v2_halfkav2_sparse_parity_with_legacy72();
    test_atomic_bin_v2_bypasses_legacy_smart_filter();
    test_atomic_bin_v2_corruption_and_schema_fail_closed();
    test_atomic_bin_v2_train_validation_overlap();
    test_corrupt_full_size_records_fail_closed();
    test_atomic_v3_provider_validation_boundaries_and_oracle();
    test_atomic_v3_provider_exact_multirow_arena_parity();
    test_atomic_v3_provider_cyclic_cursor_and_exact_resume();
    test_atomic_v3_provider_cursor_and_counter_boundaries();
    test_atomic_v3_provider_skip_precedes_decode_and_resume_seeks_directly();
    test_atomic_v3_provider_snapshot_freezes_unread_source_bytes();
    test_atomic_v3_provider_rejects_replaced_and_duplicate_shards();
    test_atomic_v3_provider_many_shards_keep_resources_bounded();
    test_atomic_v3_provider_manifest_tamper_and_snapshot_cleanup();
    return 0;
}
