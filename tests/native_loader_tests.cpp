#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "../training_data_loader.cpp"
#include "../external/Atomic-Stockfish/src/atomic_init.h"
#include "../external/Atomic-Stockfish/src/data/sha256.h"

using bin::TrainingDataEntry;
using namespace chess;
namespace AtomicData = Stockfish::Data;

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
    position.setRule50Counter(std::numeric_limits<std::uint16_t>::max());
    position.setFullMove(std::numeric_limits<std::uint32_t>::max());
    position.setSideToMove(Color::Black);

    assert(position.rule50Counter() == std::numeric_limits<std::uint16_t>::max());
    assert(position.fullMove() == std::numeric_limits<std::uint32_t>::max());
    assert(position.ply() == 2ULL * std::numeric_limits<std::uint32_t>::max());
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

int main()
{
    static_assert(sizeof(bin::nodchip::PackedSfenValue) == 72, "legacy ABI must stay 72 bytes");
    test_smart_filter();
    test_seeded_random_skipping();
    test_c_api_validation();
    test_atomic_training_data_schema_handshake();
    test_atomic_bin_v2_manifest_only_roundtrip();
    test_widened_internal_position_boundaries();
    test_atomic_bin_v2_atomic960_and_stm_semantics();
    test_atomic_bin_v2_multishard_eof_partial_and_determinism();
    test_atomic_bin_v2_halfkav2_sparse_parity_with_legacy72();
    test_atomic_bin_v2_bypasses_legacy_smart_filter();
    test_atomic_bin_v2_corruption_and_schema_fail_closed();
    test_atomic_bin_v2_train_validation_overlap();
    test_corrupt_full_size_records_fail_closed();
    return 0;
}
