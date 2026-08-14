#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace alpha_go {

// Board representation: flat array for cache efficiency
// Values: 0=EMPTY, 1=BLACK, 2=WHITE
class GoBoard {
public:
    static constexpr int8_t EMPTY = 0;
    static constexpr int8_t BLACK = 1;
    static constexpr int8_t WHITE = 2;
    // 7.5 is recommended for high level / AI (black's first-move advantage is maximized)
    // 5.5 is recommended for casual games (to prevent game from being too hard for black)
    static constexpr float KOMI = 7.5f;

    explicit GoBoard(int size = 9, float komi = KOMI);
    GoBoard(const GoBoard&) = default;
    GoBoard& operator=(const GoBoard&) = default;
    GoBoard(GoBoard&&) = default;
    GoBoard& operator=(GoBoard&&) = default;

    // Core operations
    bool play(int row, int col);      // Returns false if illegal
    bool play_flat(int index);        // Flat index version
    bool pass();                      // Pass move

    // Queries
    bool is_legal(int row, int col) const;
    bool is_legal_flat(int index) const;
    std::vector<int> get_legal_moves_flat() const;  // Returns flat indices
    // Game ends after two CONSECUTIVE passes. consecutive_passes_ is reset to
    // 0 inside play() on any non-pass move, so this is a strict consecutive
    // check, not a cumulative one.
    bool is_game_over() const { return consecutive_passes_ >= 2; }
    float score() const;              // Tromp-Taylor area scoring, returns black - white (with komi)
    int8_t get_winner() const;        // BLACK, WHITE, or 0 for draw

    // State access
    int size() const { return size_; }
    int8_t at(int row, int col) const { return board_[row * size_ + col]; }
    int8_t at_flat(int index) const { return board_[index]; }
    int8_t to_play() const { return to_play_; }
    const int8_t* data() const { return board_.data(); }
    int consecutive_passes() const { return consecutive_passes_; }
    int move_count() const { return move_count_; }
    float komi() const { return komi_; }
    std::optional<int> ko_point() const { return ko_point_; }

    // Utility
    int flat_index(int row, int col) const { return row * size_ + col; }
    std::pair<int, int> row_col(int flat) const { return {flat / size_, flat % size_}; }

    // Set board state from array (for Python interop)
    // board_data: flat array of size*size int8_t values (EMPTY/BLACK/WHITE)
    // to_play: current player (BLACK or WHITE)
    void set_from_array(const int8_t* board_data, int8_t to_play);

private:
    void init_neighbors();
    void init_zobrist();              // Fill zobrist_ with deterministic values

    // Flood-fill to find connected group and its liberties
    // Returns (group_indices, liberty_indices)
    std::pair<std::vector<int>, std::vector<int>> get_group_and_liberties(int index) const;

    // Remove all stones in a group
    int remove_group(const std::vector<int>& group);

    // Apply a move without re-checking legality. Used by play_flat (after
    // is_legal_flat returns true) and by is_legal_flat itself for the
    // positional-superko simulation (on a temporary copy). Mutates board_,
    // ko_point_, consecutive_passes_, move_count_, to_play_, current_hash_,
    // and ALSO inserts the pre-move hash into seen_hashes_.
    void play_flat_unchecked(int index);

    int size_;
    float komi_;
    std::vector<int8_t> board_;       // size_ * size_
    int8_t to_play_ = BLACK;
    std::optional<int> ko_point_;     // Flat index of ko point
    int consecutive_passes_ = 0;
    int move_count_ = 0;

    // Pre-computed neighbors (4 neighbors per cell max)
    // neighbor_indices_[i] contains the flat indices of neighbors of cell i
    // neighbor_counts_[i] is the number of valid neighbors (1-4)
    std::vector<std::array<int, 4>> neighbor_indices_;
    std::vector<int> neighbor_counts_;

    // Positional superko (PSK) state.
    //
    // - zobrist_[color][index] is a per-(color, position) random 64-bit
    //   value, fixed at construction (deterministic seed). The current
    //   board hash is the XOR of zobrist_[board_[i]][i] over all
    //   non-empty positions (color codes 1=BLACK, 2=WHITE; we ignore
    //   slot 0).
    // - current_hash_ is incrementally maintained: every stone placement
    //   XORs in zobrist_[player][p]; every capture XORs out
    //   zobrist_[stone][p].
    // - seen_hashes_ is the set of hashes for board states reached
    //   *before* the current state (positions B0, B1, ..., B_{n-1} when
    //   the current state is Bn). is_legal_flat rejects any move whose
    //   resulting hash is in seen_hashes_ — that's positional superko.
    //   A pass leaves the board unchanged so the resulting hash equals
    //   current_hash_ (which is NOT in seen_hashes_ until the NEXT move
    //   adds it), so passing is always legal under PSK as expected.
    std::vector<std::array<uint64_t, 3>> zobrist_;
    uint64_t current_hash_ = 0;
    std::unordered_set<uint64_t> seen_hashes_;
};

}  // namespace alpha_go
