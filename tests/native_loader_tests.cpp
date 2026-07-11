#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "../training_data_loader.cpp"

using bin::TrainingDataEntry;
using namespace chess;

static Square sq(int file, int rank)
{
    return static_cast<Square>(rank * static_cast<int>(File::FILE_NB) + file);
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

int main()
{
    static_assert(sizeof(bin::nodchip::PackedSfenValue) == 72, "legacy ABI must stay 72 bytes");
    test_smart_filter();
    test_seeded_random_skipping();
    test_c_api_validation();
    return 0;
}
