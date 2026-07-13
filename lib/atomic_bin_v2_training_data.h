#ifndef ATOMIC_BIN_V2_TRAINING_DATA_H_
#define ATOMIC_BIN_V2_TRAINING_DATA_H_

#include "nnue_training_data_formats.h"

#include "../external/Atomic-Stockfish/src/data/atomic_bin_v2_reader.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace training_data
{
    namespace atomic_bin_v2
    {
        inline chess::Piece piece_from_wire(Stockfish::u8 value)
        {
            using namespace Stockfish::Data;
            switch (value)
            {
            case ATOMIC_BIN_V2_EMPTY:
                return chess::Piece::None;
            case ATOMIC_BIN_V2_WHITE_PAWN:
                return chess::make_piece(chess::PieceType::Pawn, chess::Color::White);
            case ATOMIC_BIN_V2_WHITE_KNIGHT:
                return chess::make_piece(chess::PieceType::Knight, chess::Color::White);
            case ATOMIC_BIN_V2_WHITE_BISHOP:
                return chess::make_piece(chess::PieceType::Bishop, chess::Color::White);
            case ATOMIC_BIN_V2_WHITE_ROOK:
                return chess::make_piece(chess::PieceType::Rook, chess::Color::White);
            case ATOMIC_BIN_V2_WHITE_QUEEN:
                return chess::make_piece(chess::PieceType::Queen, chess::Color::White);
            case ATOMIC_BIN_V2_WHITE_KING:
                return chess::make_piece(chess::PieceType::King, chess::Color::White);
            case ATOMIC_BIN_V2_BLACK_PAWN:
                return chess::make_piece(chess::PieceType::Pawn, chess::Color::Black);
            case ATOMIC_BIN_V2_BLACK_KNIGHT:
                return chess::make_piece(chess::PieceType::Knight, chess::Color::Black);
            case ATOMIC_BIN_V2_BLACK_BISHOP:
                return chess::make_piece(chess::PieceType::Bishop, chess::Color::Black);
            case ATOMIC_BIN_V2_BLACK_ROOK:
                return chess::make_piece(chess::PieceType::Rook, chess::Color::Black);
            case ATOMIC_BIN_V2_BLACK_QUEEN:
                return chess::make_piece(chess::PieceType::Queen, chess::Color::Black);
            case ATOMIC_BIN_V2_BLACK_KING:
                return chess::make_piece(chess::PieceType::King, chess::Color::Black);
            default:
                throw std::invalid_argument("Atomic BIN V2 reader returned an unknown piece code");
            }
        }

        inline chess::PieceType promotion_from_wire(Stockfish::u8 value)
        {
            using namespace Stockfish::Data;
            switch (value)
            {
            case ATOMIC_BIN_V2_PROMOTE_KNIGHT:
                return chess::PieceType::Knight;
            case ATOMIC_BIN_V2_PROMOTE_BISHOP:
                return chess::PieceType::Bishop;
            case ATOMIC_BIN_V2_PROMOTE_ROOK:
                return chess::PieceType::Rook;
            case ATOMIC_BIN_V2_PROMOTE_QUEEN:
                return chess::PieceType::Queen;
            default:
                throw std::invalid_argument("Atomic BIN V2 reader returned an invalid promotion code");
            }
        }

        inline chess::Move move_from_fields(
            const Stockfish::Data::AtomicBinV2MoveFields& fields,
            chess::Color side_to_move)
        {
            using namespace Stockfish::Data;
            const auto from = static_cast<chess::Square>(fields.from);
            const auto to = static_cast<chess::Square>(fields.to);
            switch (fields.type)
            {
            case ATOMIC_BIN_V2_NORMAL:
                return {from, to, chess::MoveType::Normal};
            case ATOMIC_BIN_V2_EN_PASSANT:
                return {from, to, chess::MoveType::EnPassant};
            case ATOMIC_BIN_V2_CASTLING:
                return {from, to, chess::MoveType::Castle};
            case ATOMIC_BIN_V2_PROMOTION:
                return {
                    from,
                    to,
                    chess::MoveType::Promotion,
                    chess::make_piece(promotion_from_wire(fields.promotion), side_to_move)};
            default:
                throw std::invalid_argument("Atomic BIN V2 reader returned an unknown move type");
            }
        }

        inline bin::TrainingDataEntry to_training_data_entry(
            const Stockfish::Data::AtomicBinV2DecodedRecord& decoded)
        {
            using namespace Stockfish::Data;
            const auto& fields = decoded.fields;

            bin::TrainingDataEntry entry{};
            entry.pos.setCastlingRights(
                static_cast<chess::CastlingRights>(fields.position.castlingRights));
            entry.pos.setSideToMove(
                fields.position.sideToMove == ATOMIC_BIN_V2_WHITE_TO_MOVE
                    ? chess::Color::White
                    : chess::Color::Black);
            entry.pos.setEpSquare(
                fields.position.enPassantSquare == AtomicBinV2NoSquare
                    ? chess::Square::NB
                    : static_cast<chess::Square>(fields.position.enPassantSquare));
            entry.pos.setRule50Counter(fields.position.rule50);
            entry.pos.setFullMove(fields.position.fullmove);

            for (std::size_t index = 0; index < fields.position.castlingRookOrigins.size(); ++index)
            {
                const auto origin = fields.position.castlingRookOrigins[index];
                entry.pos.setCastlingRookOrigin(
                    index,
                    origin == AtomicBinV2NoSquare
                        ? chess::Square::NB
                        : static_cast<chess::Square>(origin));
            }

            for (std::size_t square = 0; square < fields.position.board.size(); ++square)
            {
                const chess::Piece piece = piece_from_wire(fields.position.board[square]);
                if (piece != chess::Piece::None)
                    entry.pos.place(piece, static_cast<chess::Square>(square));
            }

            entry.move = move_from_fields(fields.move, entry.pos.sideToMove());
            entry.score = fields.score;
            entry.ply = fields.ply;
            entry.result = fields.result;
            entry.flags = (fields.flags & ATOMIC_BIN_V2_ATOMIC960)
                ? bin::TrainingDataAtomic960
                : bin::NoTrainingDataFlags;
            return entry;
        }
    }
}

#endif
