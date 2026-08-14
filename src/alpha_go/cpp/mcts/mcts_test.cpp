#include "mcts.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <unordered_map>

using namespace alpha_go;

// Helper: Uniform policy evaluator
std::pair<std::unordered_map<int, float>, float> uniform_evaluator(const GoBoard& state) {
    auto moves = state.get_legal_moves_flat();
    // Add pass action
    moves.push_back(PASS_ACTION);

    std::unordered_map<int, float> policy;
    float uniform_prob = 1.0f / moves.size();
    for (int action : moves) {
        policy[action] = uniform_prob;
    }

    return {policy, 0.5f};  // Neutral value
}

// Helper: Biased policy evaluator (prefers lower indices)
std::pair<std::unordered_map<int, float>, float> biased_evaluator(const GoBoard& state) {
    auto moves = state.get_legal_moves_flat();
    moves.push_back(PASS_ACTION);

    std::unordered_map<int, float> policy;
    float total = 0.0f;

    for (int action : moves) {
        float weight = 1.0f / (action + 2);  // Higher weight for lower indices
        policy[action] = weight;
        total += weight;
    }

    for (auto& [action, prob] : policy) {
        prob /= total;
    }

    return {policy, 0.6f};
}

TEST_CASE("MCTSTree construction", "[mcts]") {
    GoBoard board(9);
    MCTSConfig config;
    MCTSTree tree(board, config);

    REQUIRE(tree.tree_size() == 1);
    REQUIRE(tree.get_root_visit_count() == 0);
}

TEST_CASE("MCTSTree single simulation", "[mcts]") {
    GoBoard board(9);
    MCTSConfig config;
    MCTSTree tree(board, config);

    tree.run_simulations(1, uniform_evaluator);

    REQUIRE(tree.get_root_visit_count() == 1);
    REQUIRE(tree.tree_size() >= 1);
}

TEST_CASE("MCTSTree multiple simulations", "[mcts]") {
    GoBoard board(9);
    MCTSConfig config;
    config.c_puct = 1.0f;

    MCTSTree tree(board, config);
    tree.run_simulations(100, uniform_evaluator);

    REQUIRE(tree.get_root_visit_count() == 100);
    REQUIRE(tree.tree_size() > 1);  // Should have expanded some nodes

    // Should have expanded some children
    auto child_visits = tree.get_child_visit_counts();
    REQUIRE_FALSE(child_visits.empty());

    // Total child visits should equal root visits minus 1
    // (first simulation evaluates root, doesn't expand children)
    int total_child_visits = 0;
    for (const auto& [action, visits] : child_visits) {
        total_child_visits += visits;
    }
    REQUIRE(total_child_visits == 99);
}

TEST_CASE("MCTSTree action probabilities temperature 1", "[mcts]") {
    GoBoard board(9);
    MCTSConfig config;
    MCTSTree tree(board, config);

    tree.run_simulations(100, uniform_evaluator);

    auto probs = tree.get_action_probabilities(1.0f);

    // Probabilities should sum to 1
    float sum = 0.0f;
    for (const auto& [action, prob] : probs) {
        sum += prob;
        REQUIRE(prob >= 0.0f);
        REQUIRE(prob <= 1.0f);
    }
    REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(1.0f, 0.01f));
}

TEST_CASE("MCTSTree action probabilities temperature 0", "[mcts]") {
    GoBoard board(9);
    MCTSConfig config;
    MCTSTree tree(board, config);

    tree.run_simulations(100, biased_evaluator);

    auto probs = tree.get_action_probabilities(0.0f);

    // Should be deterministic - one action has probability 1
    int ones = 0;
    for (const auto& [action, prob] : probs) {
        if (prob == 1.0f) ones++;
    }
    REQUIRE(ones == 1);
}

TEST_CASE("MCTSTree select action", "[mcts]") {
    GoBoard board(9);
    MCTSConfig config;
    MCTSTree tree(board, config);

    tree.run_simulations(100, uniform_evaluator);

    // Select action with temperature 1
    int action = tree.select_action(1.0f);

    // Action should be valid (either a legal move or pass)
    auto legal = board.get_legal_moves_flat();
    bool is_valid = (action == PASS_ACTION) ||
                    (std::find(legal.begin(), legal.end(), action) != legal.end());
    REQUIRE(is_valid);
}

TEST_CASE("MCTSTree deterministic selection with temperature 0", "[mcts]") {
    GoBoard board(9);
    MCTSConfig config;
    MCTSTree tree(board, config);

    tree.run_simulations(100, biased_evaluator);

    // With temperature 0, should always select same action
    int first_action = tree.select_action(0.0f);
    for (int i = 0; i < 10; ++i) {
        REQUIRE(tree.select_action(0.0f) == first_action);
    }
}

TEST_CASE("MCTSTree respects high prior", "[mcts]") {
    GoBoard board(9);
    MCTSConfig config;
    config.c_puct = 1.0f;

    MCTSTree tree(board, config);
    tree.run_simulations(200, biased_evaluator);

    auto visits = tree.get_child_visit_counts();

    // With biased policy, lower indices should generally have more visits
    // (though not guaranteed due to MCTS exploration)
    if (visits.size() > 1) {
        // Find the action with most visits
        int max_action = -1;
        int max_visits = 0;
        for (const auto& [action, v] : visits) {
            if (v > max_visits) {
                max_visits = v;
                max_action = action;
            }
        }
        // The most visited action should be a low-index one (or pass)
        REQUIRE((max_action < 10 || max_action == PASS_ACTION));
    }
}

TEST_CASE("MCTSTree with Dirichlet noise", "[mcts]") {
    GoBoard board(9);
    MCTSConfig config;
    config.dirichlet_alpha = 0.3f;
    config.dirichlet_weight = 0.25f;

    MCTSTree tree(board, config);
    tree.run_simulations(50, uniform_evaluator);

    // Should still work with noise
    REQUIRE(tree.get_root_visit_count() == 50);
}

TEST_CASE("MCTSTree PUCT scores", "[mcts]") {
    GoBoard board(9);
    MCTSConfig config;
    config.c_puct = 1.0f;

    MCTSTree tree(board, config);
    tree.run_simulations(10, uniform_evaluator);

    // Just check it doesn't crash and returns reasonable values
    auto child_visits = tree.get_child_visit_counts();
    auto child_q = tree.get_child_q_values();

    for (const auto& [action, visits] : child_visits) {
        REQUIRE(visits >= 0);
    }

    for (const auto& [action, q] : child_q) {
        REQUIRE(q >= 0.0f);
        REQUIRE(q <= 1.0f);
    }
}

TEST_CASE("MCTSTree handles terminal states", "[mcts]") {
    GoBoard board(9);
    // End the game immediately
    board.pass();  // Black
    board.pass();  // White
    REQUIRE(board.is_game_over());

    MCTSConfig config;
    MCTSTree tree(board, config);

    tree.run_simulations(10, uniform_evaluator);

    // Should handle terminal state gracefully
    REQUIRE(tree.get_root_visit_count() == 10);
}

TEST_CASE("MCTSTree value perspective flipping", "[mcts]") {
    // Create a simple game position and verify Q values are stored correctly
    GoBoard board(9);
    MCTSConfig config;
    config.c_puct = 1.0f;

    MCTSTree tree(board, config);
    tree.run_simulations(100, uniform_evaluator);

    // Root Q should be between 0 and 1
    float root_q = tree.get_root_q_value();
    REQUIRE(root_q >= 0.0f);
    REQUIRE(root_q <= 1.0f);

    // Child Q values should also be between 0 and 1
    auto child_q = tree.get_child_q_values();
    for (const auto& [action, q] : child_q) {
        REQUIRE(q >= 0.0f);
        REQUIRE(q <= 1.0f);
    }
}
