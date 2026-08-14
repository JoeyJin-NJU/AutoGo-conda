#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>

#include "go/go_game.h"

namespace alpha_go {

// Action type for Go: int (flat index, -1 for pass)
using GoAction = int;
constexpr GoAction PASS_ACTION = -1;

struct MCTSConfig {
    float c_puct = 1.0f;
    float lambda_ = 0.0f;          // 0 = AlphaZero style (pure value), 1 = pure rollout
    float dirichlet_alpha = 0.0f;  // 0 = no noise
    float dirichlet_weight = 0.25f;
    float temperature = 1.0f;
    int max_depth = 100;           // Max total depth from game start (tree + rollout combined)
    float rollout_temperature = 1.0f;  // Temperature for sampling during rollouts

    // Playout Cap Randomization (PCR): if non-empty, each call to run_simulations
    // samples num_simulations from this categorical distribution, overriding the
    // argument passed in. pcr_probs must sum to 1 and have the same length as pcr_sims.
    std::vector<int> pcr_sims;
    std::vector<float> pcr_probs;
};

// Node in the MCTS tree
// Stored in a flat vector with indices instead of pointers
struct MCTSNode {
    int N = 0;                     // Visit count (real backups)
    int N_virt = 0;                // Virtual-loss pending visits (leaves in-flight through this node)
    float Q = 0.0f;                // Action value, stored from player_at_parent perspective.
                                   // At the root, that is the OPPONENT of the player to move.
    // Raw v_theta from the single NN evaluation done at node expansion, stored
    // from player_at_parent perspective (same convention as Q). Unlike Q this
    // value is never re-averaged across simulations, so it reflects the pure
    // policy/value network estimate for this position. Meaningful only after
    // the node has been expanded at least once (has_eval == true).
    float first_eval_value = 0.0f;
    bool has_eval = false;
    int parent_idx = -1;           // -1 for root
    int8_t player_at_parent = 0;   // Which player made the move to reach this node
    int depth = 0;                 // Depth from root (root has depth 0)

    // Children stored as sparse map: action -> node_index
    std::unordered_map<int, int> children;

    // Policy priors (log-probabilities)
    std::unordered_map<int, float> logP_A;

    // Game state at this node
    GoBoard state;

    explicit MCTSNode(const GoBoard& s) : state(s) {}
    MCTSNode() : state(9) {}
};

class MCTSTree {
public:
    // Policy/value callback type
    // Returns (policy_dict: action -> probability, value: float)
    using EvaluatorFn = std::function<std::pair<std::unordered_map<int, float>, float>(const GoBoard&)>;

    // Batched evaluator: takes a vector of boards, returns per-board (policy, value).
    using BatchedEvaluatorFn = std::function<
        std::vector<std::pair<std::unordered_map<int, float>, float>>
        (const std::vector<GoBoard>&)>;

    MCTSTree(const GoBoard& root_state, const MCTSConfig& config);

    // Run simulations
    void run_simulations(int num_simulations, EvaluatorFn evaluator);

    // Leaf-parallel MCTS with virtual loss. Collects up to leaf_batch_size leaves
    // per iteration via PUCT selection (with virtual loss applied along each path),
    // evaluates them in one batched call, then expands + backs up. Terminal leaves
    // are backed up without going through the evaluator.
    void run_simulations_batched(
        int num_simulations, int leaf_batch_size, BatchedEvaluatorFn evaluator);

    // Action selection
    std::unordered_map<int, float> get_action_probabilities(float temperature = 1.0f) const;
    int select_action(float temperature = 1.0f) const;

    // Access
    const MCTSNode& root() const { return nodes_[0]; }
    size_t tree_size() const { return nodes_.size(); }
    int get_root_visit_count() const { return nodes_[0].N; }
    float get_root_q_value() const { return nodes_[0].Q; }

    // Get child statistics for debugging
    std::unordered_map<int, int> get_child_visit_counts() const;
    std::unordered_map<int, float> get_child_q_values() const;
    // Raw NN value stored at each root-child at expansion time, from the
    // root's to_play-player perspective (same convention as Q). Only children
    // that were actually evaluated appear in the map.
    std::unordered_map<int, float> get_child_first_eval_values() const;
    // For each root child, the maximum depth reached anywhere in the subtree
    // rooted at that child (in the same units as MCTSNode::depth). Useful for
    // teacher-mode "how deeply did MCTS explore this move" visualization.
    std::unordered_map<int, int> get_child_max_subtree_depths() const;
    // Root policy priors (exp of logP_A), as recorded from the evaluator at
    // the start of run_simulations. Includes Dirichlet noise if it was applied.
    std::unordered_map<int, float> get_root_policy_priors() const;

private:
    // Core MCTS operations
    int create_node(const GoBoard& state, int parent_idx, int8_t player_at_parent);

    // Single playout (recursive)
    float perform_playout(int node_idx, EvaluatorFn& evaluator);

    // PUCT selection
    int select_action_puct(int node_idx) const;
    std::unordered_map<int, float> compute_puct_scores(int node_idx) const;

    // Dirichlet noise
    void add_dirichlet_noise(float alpha, float weight);

    // Fast rollout from a state using policy sampling
    // Returns rollout_value from perspective of player_perspective (0=BLACK, 1=WHITE)
    // remaining_depth: how many more moves can be made (tree + rollout combined limit)
    float fast_rollout(
        const GoBoard& start_state,
        int8_t player_perspective,
        int remaining_depth,
        EvaluatorFn& evaluator);

    // Sample action from policy with temperature
    int sample_action_from_policy(
        const std::unordered_map<int, float>& policy,
        float temperature);

    std::vector<MCTSNode> nodes_;
    MCTSConfig config_;

    // Random number generator for action selection
    mutable std::mt19937 rng_;
};

}  // namespace alpha_go
