#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "go/go_game.h"
#include "mcts/mcts.h"

namespace py = pybind11;

PYBIND11_MODULE(alpha_go_cpp, m) {
    m.doc() = "C++ backend for AlphaGo MCTS and Go game";

    // Constants
    m.attr("PASS_ACTION") = alpha_go::PASS_ACTION;

    // GoBoard binding
    py::class_<alpha_go::GoBoard>(m, "GoBoard")
        .def(py::init<int, float>(), py::arg("size") = 9, py::arg("komi") = alpha_go::GoBoard::KOMI)
        .def("play", &alpha_go::GoBoard::play, py::arg("row"), py::arg("col"),
             "Play a stone at (row, col). Returns true if legal.")
        .def("play_flat", &alpha_go::GoBoard::play_flat, py::arg("index"),
             "Play a stone at flat index. Returns true if legal.")
        .def("pass_move", &alpha_go::GoBoard::pass,
             "Pass the turn. Returns true.")
        .def("is_legal", &alpha_go::GoBoard::is_legal, py::arg("row"), py::arg("col"),
             "Check if playing at (row, col) is legal.")
        .def("is_legal_flat", &alpha_go::GoBoard::is_legal_flat, py::arg("index"),
             "Check if playing at flat index is legal.")
        .def("get_legal_moves_flat", &alpha_go::GoBoard::get_legal_moves_flat,
             "Get list of legal move indices (not including pass).")
        .def("is_game_over", &alpha_go::GoBoard::is_game_over,
             "Check if game is over (two consecutive passes).")
        .def("score", &alpha_go::GoBoard::score,
             "Get score (black - white) using Chinese rules.")
        .def("get_winner", &alpha_go::GoBoard::get_winner,
             "Get winner: BLACK (1), WHITE (2), or 0 for draw.")
        .def("size", &alpha_go::GoBoard::size,
             "Get board size.")
        .def("to_play", &alpha_go::GoBoard::to_play,
             "Get current player: BLACK (1) or WHITE (2).")
        .def("move_count", &alpha_go::GoBoard::move_count,
             "Get total number of moves played.")
        .def("komi", &alpha_go::GoBoard::komi,
             "Get komi value for this board.")
        .def("at", &alpha_go::GoBoard::at, py::arg("row"), py::arg("col"),
             "Get stone at (row, col): EMPTY (0), BLACK (1), WHITE (2).")
        .def("row_col", &alpha_go::GoBoard::row_col, py::arg("flat_index"),
             "Convert flat index to (row, col) pair.")
        .def("copy", [](const alpha_go::GoBoard& b) { return alpha_go::GoBoard(b); },
             "Create a copy of the board.")
        .def("to_numpy", [](const alpha_go::GoBoard& b) {
            // Return numpy array copy of board
            auto arr = py::array_t<int8_t>({b.size(), b.size()});
            auto buf = arr.mutable_unchecked<2>();
            for (int i = 0; i < b.size(); i++) {
                for (int j = 0; j < b.size(); j++) {
                    buf(i, j) = b.at(i, j);
                }
            }
            return arr;
        }, "Convert board to numpy array.")
        .def("set_from_numpy", [](alpha_go::GoBoard& b, py::array_t<int8_t> arr, int8_t to_play) {
            // Set board state from numpy array
            auto buf = arr.unchecked<2>();
            if (buf.shape(0) != b.size() || buf.shape(1) != b.size()) {
                throw std::runtime_error("Array shape must match board size");
            }
            // Flatten and copy
            std::vector<int8_t> flat(b.size() * b.size());
            for (int i = 0; i < b.size(); i++) {
                for (int j = 0; j < b.size(); j++) {
                    flat[i * b.size() + j] = buf(i, j);
                }
            }
            b.set_from_array(flat.data(), to_play);
        }, py::arg("board_array"), py::arg("to_play"),
           "Set board state from numpy array and current player.")
        .def("__repr__", [](const alpha_go::GoBoard& b) {
            std::string s = "GoBoard(" + std::to_string(b.size()) + "x" +
                           std::to_string(b.size()) + ", to_play=";
            s += (b.to_play() == alpha_go::GoBoard::BLACK) ? "BLACK" : "WHITE";
            s += ", moves=" + std::to_string(b.move_count()) + ")";
            return s;
        })
        .def_readonly_static("EMPTY", &alpha_go::GoBoard::EMPTY)
        .def_readonly_static("BLACK", &alpha_go::GoBoard::BLACK)
        .def_readonly_static("WHITE", &alpha_go::GoBoard::WHITE)
        .def_readonly_static("KOMI", &alpha_go::GoBoard::KOMI);

    // MCTSConfig binding
    py::class_<alpha_go::MCTSConfig>(m, "MCTSConfig")
        .def(py::init<>())
        .def_readwrite("c_puct", &alpha_go::MCTSConfig::c_puct,
                      "PUCT exploration constant (default: 1.0)")
        .def_readwrite("lambda_", &alpha_go::MCTSConfig::lambda_,
                      "Mix between value and rollout (0 = pure value, default: 0.0)")
        .def_readwrite("dirichlet_alpha", &alpha_go::MCTSConfig::dirichlet_alpha,
                      "Dirichlet noise alpha (0 = no noise, default: 0.0)")
        .def_readwrite("dirichlet_weight", &alpha_go::MCTSConfig::dirichlet_weight,
                      "Weight of Dirichlet noise (default: 0.25)")
        .def_readwrite("temperature", &alpha_go::MCTSConfig::temperature,
                      "Temperature for action selection (default: 1.0)")
        .def_readwrite("max_depth", &alpha_go::MCTSConfig::max_depth,
                      "Maximum total depth from game start (tree + rollout combined, default: 100)")
        .def_readwrite("rollout_temperature", &alpha_go::MCTSConfig::rollout_temperature,
                      "Temperature for sampling during fast rollouts (default: 1.0)")
        .def_readwrite("pcr_sims", &alpha_go::MCTSConfig::pcr_sims,
                      "Playout cap randomization: list of sim counts to sample from.")
        .def_readwrite("pcr_probs", &alpha_go::MCTSConfig::pcr_probs,
                      "Playout cap randomization: categorical probabilities (must match pcr_sims length, sum to 1).")
        .def("__repr__", [](const alpha_go::MCTSConfig& c) {
            return "MCTSConfig(c_puct=" + std::to_string(c.c_puct) +
                   ", lambda=" + std::to_string(c.lambda_) +
                   ", dirichlet_alpha=" + std::to_string(c.dirichlet_alpha) + ")";
        });

    // MCTSTree binding
    py::class_<alpha_go::MCTSTree>(m, "MCTSTree")
        .def(py::init<const alpha_go::GoBoard&, const alpha_go::MCTSConfig&>(),
             py::arg("root_state"), py::arg("config"),
             "Create MCTS tree from root state with given config.")
        .def("run_simulations", &alpha_go::MCTSTree::run_simulations,
             py::arg("num_simulations"), py::arg("evaluator"),
             "Run MCTS simulations using the evaluator function.\n"
             "evaluator: callable(GoBoard) -> (dict[int, float], float)\n"
             "Returns (action -> probability dict, value estimate).")
        .def("get_action_probabilities", &alpha_go::MCTSTree::get_action_probabilities,
             py::arg("temperature") = 1.0f,
             "Get action probabilities based on visit counts.\n"
             "temperature=0 gives deterministic (argmax), temperature=1 proportional.")
        .def("select_action", &alpha_go::MCTSTree::select_action,
             py::arg("temperature") = 1.0f,
             "Select an action based on visit counts and temperature.")
        .def("tree_size", &alpha_go::MCTSTree::tree_size,
             "Get number of nodes in the tree.")
        .def("get_root_visit_count", &alpha_go::MCTSTree::get_root_visit_count,
             "Get visit count of root node.")
        .def("get_root_q_value", &alpha_go::MCTSTree::get_root_q_value,
             "Get Q-value of root node (player_at_parent / opponent perspective).")
        .def("get_root_policy_priors", &alpha_go::MCTSTree::get_root_policy_priors,
             "Get root policy priors as dict[action, probability]. Includes Dirichlet noise if applied.")
        .def("get_child_visit_counts", &alpha_go::MCTSTree::get_child_visit_counts,
             "Get visit counts of root's children as dict[action, count].")
        .def("get_child_q_values", &alpha_go::MCTSTree::get_child_q_values,
             "Get Q-values of root's children as dict[action, q_value].")
        .def("get_child_first_eval_values", &alpha_go::MCTSTree::get_child_first_eval_values,
             "Get the raw NN v_theta recorded at each root-child at expansion "
             "time as dict[action, value]. Same perspective as Q (root player).")
        .def("get_child_max_subtree_depths", &alpha_go::MCTSTree::get_child_max_subtree_depths,
             "Get max subtree depth under each root child as dict[action, depth].")
        .def("run_simulations_batched", &alpha_go::MCTSTree::run_simulations_batched,
             py::arg("num_simulations"), py::arg("leaf_batch_size"), py::arg("batched_evaluator"),
             "Leaf-parallel MCTS with virtual loss.\n"
             "batched_evaluator: callable(list[GoBoard]) -> list[(dict[int,float], float)]");

    // Convenience function for running MCTS with a Python evaluator
    m.def("run_mcts", [](
        const alpha_go::GoBoard& state,
        int num_simulations,
        const alpha_go::MCTSConfig& config,
        py::function evaluator,
        float temperature
    ) {
        alpha_go::MCTSTree tree(state, config);

        // Wrap Python evaluator
        auto cpp_evaluator = [&evaluator](const alpha_go::GoBoard& s)
            -> std::pair<std::unordered_map<int, float>, float> {
            py::object result = evaluator(s);
            auto policy = result.attr("__getitem__")(0).cast<std::unordered_map<int, float>>();
            auto value = result.attr("__getitem__")(1).cast<float>();
            return std::make_pair(policy, value);
        };

        tree.run_simulations(num_simulations, cpp_evaluator);
        return tree.get_action_probabilities(temperature);
    },
    py::arg("state"),
    py::arg("num_simulations"),
    py::arg("config"),
    py::arg("evaluator"),
    py::arg("temperature") = 1.0f,
    "Run MCTS search and return action probabilities.\n"
    "state: GoBoard root state\n"
    "num_simulations: number of MCTS simulations\n"
    "config: MCTSConfig\n"
    "evaluator: callable(GoBoard) -> (dict[int, float], float)\n"
    "temperature: temperature for action selection\n"
    "Returns: dict[action, probability]");

    // Version info
    m.attr("__version__") = "0.1.0";
}
