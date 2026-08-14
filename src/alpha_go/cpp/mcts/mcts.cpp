#include "mcts.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace alpha_go {

MCTSTree::MCTSTree(const GoBoard& root_state, const MCTSConfig& config)
    : config_(config), rng_(std::random_device{}()) {
    // Create root node
    nodes_.emplace_back(root_state);
    // Root's player_at_parent is the opponent (who made the "last" move)
    nodes_[0].player_at_parent = (root_state.to_play() == GoBoard::BLACK) ? 1 : 0;
}

int MCTSTree::create_node(const GoBoard& state, int parent_idx, int8_t player_at_parent) {
    int idx = static_cast<int>(nodes_.size());
    nodes_.emplace_back(state);
    nodes_[idx].parent_idx = parent_idx;
    nodes_[idx].player_at_parent = player_at_parent;
    // Set depth based on parent
    nodes_[idx].depth = (parent_idx >= 0) ? nodes_[parent_idx].depth + 1 : 0;
    return idx;
}

std::unordered_map<int, float> MCTSTree::compute_puct_scores(int node_idx) const {
    const MCTSNode& node = nodes_[node_idx];

    // Total visits to children
    int total_visits = 0;
    for (const auto& [action, log_prior] : node.logP_A) {
        auto it = node.children.find(action);
        if (it != node.children.end()) {
            total_visits += nodes_[it->second].N;
        }
    }
    float sqrt_total = std::sqrt(static_cast<float>(total_visits) + 1.0f);

    std::unordered_map<int, float> scores;
    for (const auto& [action, log_prior] : node.logP_A) {
        float prior = std::exp(log_prior);

        float q_value = 0.0f;
        int n_visits = 0;

        auto it = node.children.find(action);
        if (it != node.children.end()) {
            q_value = nodes_[it->second].Q;
            n_visits = nodes_[it->second].N;
        }

        // PUCT formula: Q + c_puct * P * sqrt(N) / (1 + n)
        float u_value = config_.c_puct * prior * sqrt_total / (1.0f + n_visits);
        scores[action] = q_value + u_value;
    }

    return scores;
}

int MCTSTree::select_action_puct(int node_idx) const {
    auto scores = compute_puct_scores(node_idx);

    int best_action = -1;
    float best_score = -std::numeric_limits<float>::infinity();

    for (const auto& [action, score] : scores) {
        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }

    return best_action;
}

float MCTSTree::perform_playout(int node_idx, EvaluatorFn& evaluator) {
    // IMPORTANT: Don't store references to nodes_ elements - vector may reallocate!
    // Always access via index.

    // Determine which player we're evaluating for (who made the move to reach this node)
    int8_t player_perspective = nodes_[node_idx].player_at_parent;

    float U;

    // Case 1: Terminal node - game is over
    if (nodes_[node_idx].state.is_game_over()) {
        // Get reward from player_perspective
        int8_t winner = nodes_[node_idx].state.get_winner();
        if (winner == 0) {
            U = 0.5f;  // Draw
        } else {
            // player_perspective: 0 = BLACK, 1 = WHITE
            int8_t player_color = (player_perspective == 0) ? GoBoard::BLACK : GoBoard::WHITE;
            U = (winner == player_color) ? 1.0f : 0.0f;
        }
    }
    // Case 2: Leaf node not yet visited - evaluate and expand
    else if (nodes_[node_idx].N == 0) {
        // Get policy and value from evaluator
        auto [policy_dict, v_theta] = evaluator(nodes_[node_idx].state);

        // Store log-priors
        for (const auto& [action, prob] : policy_dict) {
            nodes_[node_idx].logP_A[action] = std::log(prob + 1e-8f);
        }

        // Value network gives value from current player's perspective
        // We need to convert to parent player's perspective
        int8_t current_player = (nodes_[node_idx].state.to_play() == GoBoard::BLACK) ? 0 : 1;
        if (current_player != player_perspective) {
            v_theta = 1.0f - v_theta;
        }

        // Snapshot the raw NN value (pre-rollout mix) for teacher-mode display.
        nodes_[node_idx].first_eval_value = v_theta;
        nodes_[node_idx].has_eval = true;

        // Mix with fast rollout if lambda > 0
        if (config_.lambda_ > 0) {
            // Compute remaining depth for rollout
            int current_depth = nodes_[node_idx].depth;
            int remaining_depth = config_.max_depth - current_depth;

            if (remaining_depth > 0) {
                float z_L = fast_rollout(
                    nodes_[node_idx].state,
                    player_perspective,
                    remaining_depth,
                    evaluator);

                // U = (1 - lambda) * v_theta + lambda * z_L
                U = (1.0f - config_.lambda_) * v_theta + config_.lambda_ * z_L;
            } else {
                // No depth remaining, just use value network
                U = v_theta;
            }
        } else {
            U = v_theta;
        }
    }
    // Case 3: Internal node - select action and recurse
    else {
        int action = select_action_puct(node_idx);

        // Expand if child doesn't exist
        auto it = nodes_[node_idx].children.find(action);
        if (it == nodes_[node_idx].children.end()) {
            GoBoard new_state = nodes_[node_idx].state;
            if (action == PASS_ACTION) {
                new_state.pass();
            } else {
                auto [row, col] = new_state.row_col(action);
                new_state.play(row, col);
            }

            int8_t current_player = (nodes_[node_idx].state.to_play() == GoBoard::BLACK) ? 0 : 1;
            int child_idx = create_node(new_state, node_idx, current_player);
            // Note: create_node may reallocate, so we need to access nodes_ again
            nodes_[node_idx].children[action] = child_idx;
        }

        int child_idx = nodes_[node_idx].children[action];

        // Recurse - get value from child's perspective
        float child_value = perform_playout(child_idx, evaluator);

        // Child value is from current player's perspective, we need parent's perspective
        U = 1.0f - child_value;
    }

    // Backup: Update visit count and action value
    nodes_[node_idx].N += 1;
    // Incremental mean update: Q = Q + (U - Q) / N
    nodes_[node_idx].Q = nodes_[node_idx].Q + (U - nodes_[node_idx].Q) / nodes_[node_idx].N;

    return U;
}

int MCTSTree::sample_action_from_policy(
    const std::unordered_map<int, float>& policy,
    float temperature) {

    if (policy.empty()) {
        return PASS_ACTION;
    }

    std::vector<int> actions;
    std::vector<float> probs;
    actions.reserve(policy.size());
    probs.reserve(policy.size());

    for (const auto& [action, prob] : policy) {
        actions.push_back(action);
        probs.push_back(prob);
    }

    // Apply temperature
    if (temperature != 1.0f && temperature > 0) {
        float max_logit = -std::numeric_limits<float>::infinity();
        std::vector<float> logits(probs.size());
        for (size_t i = 0; i < probs.size(); ++i) {
            logits[i] = std::log(probs[i] + 1e-8f) / temperature;
            max_logit = std::max(max_logit, logits[i]);
        }
        float sum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i] = std::exp(logits[i] - max_logit);
            sum += probs[i];
        }
        for (float& p : probs) {
            p /= sum;
        }
    }

    // Sample
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return actions[dist(rng_)];
}

float MCTSTree::fast_rollout(
    const GoBoard& start_state,
    int8_t player_perspective,
    int remaining_depth,
    EvaluatorFn& evaluator) {

    GoBoard current = start_state;
    int depth = 0;

    while (!current.is_game_over() && depth < remaining_depth) {
        // Get policy from evaluator
        auto [policy, _] = evaluator(current);

        int action;
        if (policy.empty()) {
            // No legal moves, pass
            current.pass();
            action = PASS_ACTION;
        } else {
            // Sample action from policy
            action = sample_action_from_policy(policy, config_.rollout_temperature);

            if (action == PASS_ACTION) {
                current.pass();
            } else {
                auto [row, col] = current.row_col(action);
                if (!current.play(row, col)) {
                    // Illegal move - shouldn't happen but fallback to pass
                    current.pass();
                    action = PASS_ACTION;
                }
            }
        }
        ++depth;
    }

    // Compute rollout value
    float rollout_value;
    if (current.is_game_over()) {
        int8_t winner = current.get_winner();
        if (winner == 0) {
            rollout_value = 0.5f;
        } else {
            int8_t player_color = (player_perspective == 0) ? GoBoard::BLACK : GoBoard::WHITE;
            rollout_value = (winner == player_color) ? 1.0f : 0.0f;
        }
    } else {
        // Hit depth limit - use value estimate from current position
        auto [_, v] = evaluator(current);
        // v is from current player's perspective
        int8_t current_player = (current.to_play() == GoBoard::BLACK) ? 0 : 1;
        if (current_player != player_perspective) {
            v = 1.0f - v;
        }
        rollout_value = v;
    }

    return rollout_value;
}

void MCTSTree::add_dirichlet_noise(float alpha, float weight) {
    MCTSNode& root = nodes_[0];
    if (root.logP_A.empty()) {
        return;
    }

    std::vector<int> actions;
    actions.reserve(root.logP_A.size());
    for (const auto& [action, _] : root.logP_A) {
        actions.push_back(action);
    }

    // Generate Dirichlet noise
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> noise(actions.size());
    float sum = 0.0f;
    for (size_t i = 0; i < actions.size(); ++i) {
        noise[i] = gamma(rng_);
        sum += noise[i];
    }
    for (float& n : noise) {
        n /= sum;
    }

    // Mix noise with priors
    for (size_t i = 0; i < actions.size(); ++i) {
        int action = actions[i];
        float log_prior = root.logP_A[action];
        float prior = std::exp(log_prior);
        float noisy_prior = (1.0f - weight) * prior + weight * noise[i];
        root.logP_A[action] = std::log(noisy_prior + 1e-8f);
    }
}

void MCTSTree::run_simulations(int num_simulations, EvaluatorFn evaluator) {
    // Playout cap randomization: sample sim count from categorical distribution
    if (!config_.pcr_sims.empty()) {
        std::discrete_distribution<int> dist(config_.pcr_probs.begin(), config_.pcr_probs.end());
        num_simulations = config_.pcr_sims[dist(rng_)];
    }
    // Get initial policy for root
    auto [policy_dict, _] = evaluator(nodes_[0].state);
    for (const auto& [action, prob] : policy_dict) {
        nodes_[0].logP_A[action] = std::log(prob + 1e-8f);
    }

    // Add exploration noise at root
    if (config_.dirichlet_alpha > 0) {
        add_dirichlet_noise(config_.dirichlet_alpha, config_.dirichlet_weight);
    }

    // Run simulations
    for (int i = 0; i < num_simulations; ++i) {
        perform_playout(0, evaluator);
    }
}

// ---------------- Leaf-parallel MCTS with virtual loss ----------------

namespace {
// Per-leaf bookkeeping for one batched iteration.
struct PendingLeaf {
    std::vector<int> path;   // root..leaf node indices
    int leaf_idx;
    bool is_terminal;
    float terminal_U;        // value from leaf.player_at_parent perspective
    int eval_slot;           // index into eval_states/eval_results, or -1 if terminal
};
}  // namespace

void MCTSTree::run_simulations_batched(
    int num_simulations, int leaf_batch_size, BatchedEvaluatorFn evaluator) {

    if (!config_.pcr_sims.empty()) {
        std::discrete_distribution<int> dist(config_.pcr_probs.begin(), config_.pcr_probs.end());
        num_simulations = config_.pcr_sims[dist(rng_)];
    }

    // Evaluate root synchronously (single-item batch) if not already evaluated.
    if (nodes_[0].logP_A.empty()) {
        auto results = evaluator(std::vector<GoBoard>{nodes_[0].state});
        auto& [policy_dict, _v] = results[0];
        for (const auto& [action, prob] : policy_dict) {
            nodes_[0].logP_A[action] = std::log(prob + 1e-8f);
        }
    }
    if (config_.dirichlet_alpha > 0) {
        add_dirichlet_noise(config_.dirichlet_alpha, config_.dirichlet_weight);
    }

    int completed = 0;
    while (completed < num_simulations) {
        int target = std::min(leaf_batch_size, num_simulations - completed);

        std::vector<PendingLeaf> pending;
        std::vector<GoBoard> eval_states;
        pending.reserve(target);
        eval_states.reserve(target);

        for (int i = 0; i < target; ++i) {
            // --- Select a leaf using PUCT with virtual loss, descend & create if needed ---
            std::vector<int> path;
            int node_idx = 0;
            path.push_back(node_idx);

            while (true) {
                // Terminal leaf
                if (nodes_[node_idx].state.is_game_over()) {
                    int8_t winner = nodes_[node_idx].state.get_winner();
                    int8_t persp = nodes_[node_idx].player_at_parent;
                    float U;
                    if (winner == 0) U = 0.5f;
                    else {
                        int8_t pc = (persp == 0) ? GoBoard::BLACK : GoBoard::WHITE;
                        U = (winner == pc) ? 1.0f : 0.0f;
                    }
                    PendingLeaf pl{path, node_idx, true, U, -1};
                    pending.push_back(pl);
                    for (int idx : path) nodes_[idx].N_virt += 1;
                    break;
                }
                // Unevaluated leaf (reached existing node with no priors)
                if (nodes_[node_idx].logP_A.empty()) {
                    PendingLeaf pl{path, node_idx, false, 0.0f, (int)eval_states.size()};
                    eval_states.push_back(nodes_[node_idx].state);
                    pending.push_back(pl);
                    for (int idx : path) nodes_[idx].N_virt += 1;
                    break;
                }
                // PUCT selection with virtual loss
                const auto& node = nodes_[node_idx];
                int total_visits = 0;
                for (const auto& [action, _lp] : node.logP_A) {
                    auto it = node.children.find(action);
                    if (it != node.children.end()) {
                        total_visits += nodes_[it->second].N + nodes_[it->second].N_virt;
                    }
                }
                float sqrt_total = std::sqrt((float)total_visits + 1.0f);
                int best_action = -1;
                float best_score = -std::numeric_limits<float>::infinity();
                for (const auto& [action, log_prior] : node.logP_A) {
                    float prior = std::exp(log_prior);
                    float q = 0.0f;
                    int n = 0, nv = 0;
                    auto it = node.children.find(action);
                    if (it != node.children.end()) {
                        q = nodes_[it->second].Q;
                        n = nodes_[it->second].N;
                        nv = nodes_[it->second].N_virt;
                    }
                    float n_eff = (float)(n + nv);
                    float q_eff = (n_eff > 0) ? (q * (float)n) / n_eff : 0.0f;
                    float u = config_.c_puct * prior * sqrt_total / (1.0f + n_eff);
                    float score = q_eff + u;
                    if (score > best_score) { best_score = score; best_action = action; }
                }

                int action = best_action;
                auto it = nodes_[node_idx].children.find(action);
                int child_idx;
                if (it == nodes_[node_idx].children.end()) {
                    // Create new child — unevaluated leaf
                    GoBoard new_state = nodes_[node_idx].state;
                    if (action == PASS_ACTION) {
                        new_state.pass();
                    } else {
                        auto [row, col] = new_state.row_col(action);
                        new_state.play(row, col);
                    }
                    int8_t current_player =
                        (nodes_[node_idx].state.to_play() == GoBoard::BLACK) ? 0 : 1;
                    child_idx = create_node(new_state, node_idx, current_player);
                    nodes_[node_idx].children[action] = child_idx;
                    path.push_back(child_idx);
                    PendingLeaf pl{path, child_idx, false, 0.0f, (int)eval_states.size()};
                    // Terminal short-circuit if the just-created state is game over.
                    if (nodes_[child_idx].state.is_game_over()) {
                        int8_t winner = nodes_[child_idx].state.get_winner();
                        int8_t persp = nodes_[child_idx].player_at_parent;
                        float U;
                        if (winner == 0) U = 0.5f;
                        else {
                            int8_t pc = (persp == 0) ? GoBoard::BLACK : GoBoard::WHITE;
                            U = (winner == pc) ? 1.0f : 0.0f;
                        }
                        pl.is_terminal = true;
                        pl.terminal_U = U;
                        pl.eval_slot = -1;
                    } else {
                        eval_states.push_back(nodes_[child_idx].state);
                    }
                    pending.push_back(pl);
                    for (int idx : path) nodes_[idx].N_virt += 1;
                    break;
                }
                child_idx = nodes_[node_idx].children[action];
                path.push_back(child_idx);
                node_idx = child_idx;
            }
        }

        // --- Batched GPU eval for all non-terminal leaves in this iteration ---
        std::vector<std::pair<std::unordered_map<int, float>, float>> eval_results;
        if (!eval_states.empty()) {
            eval_results = evaluator(eval_states);
        }

        // --- Expand + backup each pending leaf ---
        for (auto& pl : pending) {
            float U;
            if (pl.is_terminal) {
                U = pl.terminal_U;
            } else {
                auto& [policy_dict, v_theta] = eval_results[pl.eval_slot];
                // Store priors on leaf (skip if somehow already set by duplicate)
                if (nodes_[pl.leaf_idx].logP_A.empty()) {
                    for (const auto& [action, prob] : policy_dict) {
                        nodes_[pl.leaf_idx].logP_A[action] = std::log(prob + 1e-8f);
                    }
                }
                int8_t persp = nodes_[pl.leaf_idx].player_at_parent;
                int8_t curp = (nodes_[pl.leaf_idx].state.to_play() == GoBoard::BLACK) ? 0 : 1;
                U = (curp != persp) ? (1.0f - v_theta) : v_theta;
                // Cache the flipped v_theta on the leaf for teacher-mode. Only
                // set on the first eval to match the non-batched path.
                if (!nodes_[pl.leaf_idx].has_eval) {
                    nodes_[pl.leaf_idx].first_eval_value = U;
                    nodes_[pl.leaf_idx].has_eval = true;
                }
            }
            // Backup along path, flipping perspective at each parent.
            for (int k = (int)pl.path.size() - 1; k >= 0; --k) {
                int idx = pl.path[k];
                nodes_[idx].N_virt -= 1;
                nodes_[idx].N += 1;
                nodes_[idx].Q += (U - nodes_[idx].Q) / (float)nodes_[idx].N;
                U = 1.0f - U;
            }
            completed++;
        }
    }
}

std::unordered_map<int, float> MCTSTree::get_action_probabilities(float temperature) const {
    const MCTSNode& root = nodes_[0];

    if (root.children.empty()) {
        // No children expanded - return uniform over legal actions
        std::unordered_map<int, float> probs;
        float uniform = 1.0f / root.logP_A.size();
        for (const auto& [action, _] : root.logP_A) {
            probs[action] = uniform;
        }
        return probs;
    }

    // Get visit counts
    std::unordered_map<int, int> visit_counts;
    for (const auto& [action, child_idx] : root.children) {
        visit_counts[action] = nodes_[child_idx].N;
    }

    if (temperature == 0) {
        // Deterministic: pick action with most visits
        int best_action = -1;
        int best_visits = -1;
        for (const auto& [action, visits] : visit_counts) {
            if (visits > best_visits) {
                best_visits = visits;
                best_action = action;
            }
        }

        std::unordered_map<int, float> probs;
        for (const auto& [action, _] : visit_counts) {
            probs[action] = (action == best_action) ? 1.0f : 0.0f;
        }
        return probs;
    }

    // Apply temperature: N^(1/tau)
    std::unordered_map<int, float> visits_temp;
    float total = 0.0f;
    for (const auto& [action, visits] : visit_counts) {
        float v = std::pow(static_cast<float>(visits), 1.0f / temperature);
        visits_temp[action] = v;
        total += v;
    }

    if (total == 0) {
        float uniform = 1.0f / visits_temp.size();
        for (auto& [action, v] : visits_temp) {
            v = uniform;
        }
        return visits_temp;
    }

    for (auto& [action, v] : visits_temp) {
        v /= total;
    }
    return visits_temp;
}

int MCTSTree::select_action(float temperature) const {
    auto probs = get_action_probabilities(temperature);

    std::vector<int> actions;
    std::vector<float> probabilities;
    for (const auto& [action, prob] : probs) {
        actions.push_back(action);
        probabilities.push_back(prob);
    }

    if (temperature == 0) {
        // Deterministic
        auto max_it = std::max_element(probabilities.begin(), probabilities.end());
        return actions[std::distance(probabilities.begin(), max_it)];
    }

    // Stochastic selection
    std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
    return actions[dist(rng_)];
}

std::unordered_map<int, int> MCTSTree::get_child_visit_counts() const {
    std::unordered_map<int, int> counts;
    for (const auto& [action, child_idx] : nodes_[0].children) {
        counts[action] = nodes_[child_idx].N;
    }
    return counts;
}

std::unordered_map<int, float> MCTSTree::get_root_policy_priors() const {
    std::unordered_map<int, float> out;
    out.reserve(nodes_[0].logP_A.size());
    for (const auto& [action, log_prior] : nodes_[0].logP_A) {
        out[action] = std::exp(log_prior);
    }
    return out;
}

std::unordered_map<int, float> MCTSTree::get_child_q_values() const {
    std::unordered_map<int, float> values;
    for (const auto& [action, child_idx] : nodes_[0].children) {
        values[action] = nodes_[child_idx].Q;
    }
    return values;
}

std::unordered_map<int, float> MCTSTree::get_child_first_eval_values() const {
    std::unordered_map<int, float> values;
    for (const auto& [action, child_idx] : nodes_[0].children) {
        if (nodes_[child_idx].has_eval) {
            values[action] = nodes_[child_idx].first_eval_value;
        }
    }
    return values;
}

std::unordered_map<int, int> MCTSTree::get_child_max_subtree_depths() const {
    // For each root child, walk its subtree and record the deepest node.
    // Subtrees are disjoint so total work across children is O(tree_size).
    std::unordered_map<int, int> out;
    std::vector<int> stack;
    for (const auto& [action, child_idx] : nodes_[0].children) {
        int max_depth = nodes_[child_idx].depth;
        stack.clear();
        stack.push_back(child_idx);
        while (!stack.empty()) {
            int cur = stack.back();
            stack.pop_back();
            if (nodes_[cur].depth > max_depth) {
                max_depth = nodes_[cur].depth;
            }
            for (const auto& [_a, ci] : nodes_[cur].children) {
                stack.push_back(ci);
            }
        }
        out[action] = max_depth;
    }
    return out;
}

}  // namespace alpha_go
