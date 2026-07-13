#include "../external/Atomic-Stockfish/src/tt.h"

namespace Stockfish
{
    // The authenticated dataset reader never supplies a transposition table
    // to Position::do_move(). Keep the loader isolated from the engine's
    // threaded TT subsystem, matching Atomic-Stockfish's standalone reader.
    TTEntry* TranspositionTable::first_entry(Key) const
    {
        return nullptr;
    }
}
