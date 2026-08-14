"""Tests for C++ MCTS implementation parity with Python."""

import sys
from pathlib import Path

import numpy as np
import pytest

import alpha_go_cpp
CPP_AVAILABLE = True

# Check for neural network dependencies
try:
    import torch
    from alpha_go.inference_server import create_evaluator_from_checkpoint
    NN_AVAILABLE = True
except ImportError:
    NN_AVAILABLE = False

# Default checkpoint path for NN tests
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "experiments/2025-12-30_21-45-retrain-10k-mcts-analysis/checkpoints/step_10000.pt"


@pytest.fixture
def cpp_board():
    """Create a C++ GoBoard."""
    if not CPP_AVAILABLE:
        pytest.skip("C++ extension not available")
    return alpha_go_cpp.GoBoard(9)


@pytest.fixture
def mcts_config():
    """Create MCTS config."""
    if not CPP_AVAILABLE:
        pytest.skip("C++ extension not available")
    config = alpha_go_cpp.MCTSConfig()
    config.c_puct = 1.0
    config.dirichlet_alpha = 0.0  # No noise for deterministic tests
    return config


def uniform_evaluator(state):
    """Uniform policy evaluator."""
    moves = state.get_legal_moves_flat()
    moves.append(alpha_go_cpp.PASS_ACTION)
    prob = 1.0 / len(moves)
    policy = {m: prob for m in moves}
    return (policy, 0.5)


def biased_evaluator(state):
    """Biased policy evaluator (prefers lower indices)."""
    moves = state.get_legal_moves_flat()
    moves.append(alpha_go_cpp.PASS_ACTION)

    total = 0.0
    policy = {}
    for action in moves:
        weight = 1.0 / (action + 2)  # Higher weight for lower indices
        policy[action] = weight
        total += weight

    for action in policy:
        policy[action] /= total

    return (policy, 0.6)


class TestMCTSTreeBasics:
    """Test basic MCTS tree functionality."""

    def test_construction(self, cpp_board, mcts_config):
        """Test tree construction."""
        tree = alpha_go_cpp.MCTSTree(cpp_board, mcts_config)
        assert tree.tree_size() == 1
        assert tree.get_root_visit_count() == 0

    def test_single_simulation(self, cpp_board, mcts_config):
        """Test running a single simulation."""
        tree = alpha_go_cpp.MCTSTree(cpp_board, mcts_config)
        tree.run_simulations(1, uniform_evaluator)

        assert tree.get_root_visit_count() == 1
        assert tree.tree_size() >= 1

    def test_multiple_simulations(self, cpp_board, mcts_config):
        """Test running multiple simulations."""
        tree = alpha_go_cpp.MCTSTree(cpp_board, mcts_config)
        tree.run_simulations(100, uniform_evaluator)

        assert tree.get_root_visit_count() == 100
        assert tree.tree_size() > 1

        # Check child visit counts
        child_visits = tree.get_child_visit_counts()
        assert len(child_visits) > 0

        # Total child visits should be 99 (first sim evaluates root)
        total_child_visits = sum(child_visits.values())
        assert total_child_visits == 99


class TestActionProbabilities:
    """Test action probability computation."""

    def test_temperature_1(self, cpp_board, mcts_config):
        """Test action probabilities with temperature 1."""
        tree = alpha_go_cpp.MCTSTree(cpp_board, mcts_config)
        tree.run_simulations(100, uniform_evaluator)

        probs = tree.get_action_probabilities(1.0)

        # Probabilities should sum to 1
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

        # All probabilities should be in [0, 1]
        for prob in probs.values():
            assert 0.0 <= prob <= 1.0

    def test_temperature_0(self, cpp_board, mcts_config):
        """Test action probabilities with temperature 0 (deterministic)."""
        tree = alpha_go_cpp.MCTSTree(cpp_board, mcts_config)
        tree.run_simulations(100, biased_evaluator)

        probs = tree.get_action_probabilities(0.0)

        # Should be deterministic - one action has probability 1
        ones = sum(1 for p in probs.values() if p == 1.0)
        assert ones == 1


class TestActionSelection:
    """Test action selection."""

    def test_select_action(self, cpp_board, mcts_config):
        """Test selecting an action."""
        tree = alpha_go_cpp.MCTSTree(cpp_board, mcts_config)
        tree.run_simulations(100, uniform_evaluator)

        action = tree.select_action(1.0)

        # Action should be valid
        legal_moves = cpp_board.get_legal_moves_flat()
        assert action in legal_moves or action == alpha_go_cpp.PASS_ACTION

    def test_deterministic_selection(self, cpp_board, mcts_config):
        """Test deterministic selection with temperature 0."""
        tree = alpha_go_cpp.MCTSTree(cpp_board, mcts_config)
        tree.run_simulations(100, biased_evaluator)

        first_action = tree.select_action(0.0)

        # Should always select same action
        for _ in range(10):
            assert tree.select_action(0.0) == first_action


class TestQValues:
    """Test Q-value computation."""

    def test_q_values_in_range(self, cpp_board, mcts_config):
        """Test that Q-values are in valid range."""
        tree = alpha_go_cpp.MCTSTree(cpp_board, mcts_config)
        tree.run_simulations(100, uniform_evaluator)

        # Root Q should be in [0, 1]
        root_q = tree.get_root_q_value()
        assert 0.0 <= root_q <= 1.0

        # Child Q values should be in [0, 1]
        child_q = tree.get_child_q_values()
        for q in child_q.values():
            assert 0.0 <= q <= 1.0

    def test_first_eval_values_and_max_depths(self, cpp_board, mcts_config):
        """Per-child v_theta and max subtree depth accessors exposed for
        teacher mode on the web demo."""
        tree = alpha_go_cpp.MCTSTree(cpp_board, mcts_config)
        tree.run_simulations(64, uniform_evaluator)

        visits = tree.get_child_visit_counts()
        fe = tree.get_child_first_eval_values()
        md = tree.get_child_max_subtree_depths()

        # Every visited root child should have recorded its first NN eval and
        # a max subtree depth ≥ 1 (children sit at depth 1 themselves).
        assert set(fe.keys()).issubset(set(visits.keys()))
        assert set(md.keys()) == set(visits.keys())
        for action, v in fe.items():
            assert 0.0 <= v <= 1.0
        for action, d in md.items():
            assert d >= 1
            assert d <= mcts_config.max_depth


class TestDirichletNoise:
    """Test Dirichlet noise at root."""

    def test_with_noise(self, cpp_board):
        """Test MCTS with Dirichlet noise."""
        config = alpha_go_cpp.MCTSConfig()
        config.c_puct = 1.0
        config.dirichlet_alpha = 0.3
        config.dirichlet_weight = 0.25

        tree = alpha_go_cpp.MCTSTree(cpp_board, config)
        tree.run_simulations(50, uniform_evaluator)

        # Should work without errors
        assert tree.get_root_visit_count() == 50


class TestRunMCTSConvenience:
    """Test the run_mcts convenience function."""

    def test_run_mcts(self, cpp_board, mcts_config):
        """Test run_mcts function."""
        probs = alpha_go_cpp.run_mcts(
            cpp_board,
            num_simulations=50,
            config=mcts_config,
            evaluator=uniform_evaluator,
            temperature=1.0
        )

        assert len(probs) > 0
        assert abs(sum(probs.values()) - 1.0) < 0.01


class TestTerminalStates:
    """Test handling of terminal states."""

    def test_terminal_state(self, mcts_config):
        """Test MCTS on terminal state."""
        if not CPP_AVAILABLE:
            pytest.skip("C++ extension not available")

        board = alpha_go_cpp.GoBoard(9)
        board.pass_move()
        board.pass_move()
        assert board.is_game_over()

        tree = alpha_go_cpp.MCTSTree(board, mcts_config)
        tree.run_simulations(10, uniform_evaluator)

        # Should handle terminal state gracefully
        assert tree.get_root_visit_count() == 10


class TestBiasedPolicy:
    """Test MCTS respects policy priors."""

    def test_respects_high_prior(self, cpp_board, mcts_config):
        """Test that MCTS explores high-prior actions more."""
        tree = alpha_go_cpp.MCTSTree(cpp_board, mcts_config)
        tree.run_simulations(200, biased_evaluator)

        visits = tree.get_child_visit_counts()

        if len(visits) > 1:
            # Find action with most visits
            max_action = max(visits, key=visits.get)
            # Should be a low-index action (or pass)
            assert max_action < 10 or max_action == alpha_go_cpp.PASS_ACTION


def create_nn_evaluator(engine):
    """Create an evaluator function for C++ MCTS from the inference engine."""
    def evaluator(cpp_board):
        board_np = cpp_board.to_numpy().astype(np.float32)
        policy_dict, value = engine.evaluate(board_np)

        # Filter to legal moves + pass
        legal_moves = cpp_board.get_legal_moves_flat()
        filtered_policy = {}
        total = 0.0
        for move in legal_moves:
            if move in policy_dict:
                filtered_policy[move] = policy_dict[move]
                total += policy_dict[move]

        # Add pass action
        pass_prob = 0.01
        filtered_policy[alpha_go_cpp.PASS_ACTION] = pass_prob
        total += pass_prob

        for move in filtered_policy:
            filtered_policy[move] /= total

        return (filtered_policy, value)
    return evaluator


# Try to import agents for parity tests
try:
    from alpha_go.agents.nn_mcts import NNMCTSAgent, CppMCTSAgent, CppSearchResult, LocalNNEvaluator
    from alpha_go.go import GoState, FastGoBoard
    from alpha_go.mcts import get_action_probabilities
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False


@pytest.mark.skipif(not NN_AVAILABLE, reason="Neural network dependencies not available")
@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="Checkpoint not found")
class TestMCTSWithNeuralNetwork:
    """Test C++ MCTS with neural network evaluation."""

    @pytest.fixture
    def nn_evaluator(self):
        """Create NN evaluator from checkpoint."""
        engine = create_evaluator_from_checkpoint(str(DEFAULT_CHECKPOINT), device="cpu")
        return create_nn_evaluator(engine)

    def test_mcts_with_nn_single_move(self, cpp_board, mcts_config, nn_evaluator):
        """Test MCTS with NN for a single move."""
        tree = alpha_go_cpp.MCTSTree(cpp_board, mcts_config)
        tree.run_simulations(50, nn_evaluator)

        assert tree.get_root_visit_count() == 50
        assert tree.tree_size() > 1

        # Q-value should be in valid range
        root_q = tree.get_root_q_value()
        assert 0.0 <= root_q <= 1.0

        # Should produce valid action probabilities
        probs = tree.get_action_probabilities(1.0)
        assert len(probs) > 0
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_mcts_with_nn_action_selection(self, cpp_board, mcts_config, nn_evaluator):
        """Test action selection with NN evaluation."""
        tree = alpha_go_cpp.MCTSTree(cpp_board, mcts_config)
        tree.run_simulations(100, nn_evaluator)

        action = tree.select_action(temperature=0.0)
        legal_moves = cpp_board.get_legal_moves_flat()
        assert action in legal_moves or action == alpha_go_cpp.PASS_ACTION

    def test_mcts_with_nn_dirichlet_noise(self, cpp_board, nn_evaluator):
        """Test MCTS with NN and Dirichlet noise."""
        config = alpha_go_cpp.MCTSConfig()
        config.c_puct = 1.5
        config.dirichlet_alpha = 0.03
        config.dirichlet_weight = 0.25

        tree = alpha_go_cpp.MCTSTree(cpp_board, config)
        tree.run_simulations(50, nn_evaluator)

        assert tree.get_root_visit_count() == 50

    def test_play_multiple_moves(self, mcts_config, nn_evaluator):
        """Test playing multiple moves in sequence."""
        board = alpha_go_cpp.GoBoard(9)
        moves_played = 0

        for _ in range(5):
            tree = alpha_go_cpp.MCTSTree(board, mcts_config)
            tree.run_simulations(30, nn_evaluator)

            action = tree.select_action(temperature=0.5)

            if action == alpha_go_cpp.PASS_ACTION:
                board.pass_move()
            else:
                if board.is_legal_flat(action):
                    board.play_flat(action)
                    moves_played += 1

        assert moves_played > 0

@pytest.mark.skipif(not NN_AVAILABLE, reason="Neural network dependencies not available")
@pytest.mark.skipif(not AGENTS_AVAILABLE, reason="Agent classes not available")
@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="Checkpoint not found")
class TestPythonCppMCTSParity:
    """Test that Python and C++ MCTS produce equivalent results."""

    @pytest.fixture
    def py_agent(self):
        """Create Python NNMCTSAgent."""
        return NNMCTSAgent(
            checkpoint_path=str(DEFAULT_CHECKPOINT),
            num_simulations=50,
            c_puct=1.0,
            add_noise=False,  # Disable noise for deterministic comparison
            device="cpu",
        )

    @pytest.fixture
    def cpp_agent(self):
        """Create C++ CppMCTSAgent."""
        evaluator = LocalNNEvaluator(
            checkpoint_path=str(DEFAULT_CHECKPOINT),
            board_size=9,
            device="cpu",
        )
        return CppMCTSAgent(
            evaluator=evaluator,
            num_simulations=50,
            c_puct=1.0,
            add_noise=False,  # Disable noise for deterministic comparison
        )

    def test_visit_counts_similar(self, py_agent, cpp_agent):
        """Test that visit counts are similar between Python and C++ MCTS."""
        # Create empty board state
        board = FastGoBoard(9)
        py_state = GoState(board)
        cpp_board = alpha_go_cpp.GoBoard(9)

        # Set same seed for both
        np.random.seed(42)
        import torch
        torch.manual_seed(42)

        # Run Python MCTS
        py_root = py_agent.search(py_state)
        py_visits = {a: c.N for a, c in py_root.children.items()}

        # Reset seed
        np.random.seed(42)
        torch.manual_seed(42)

        # Run C++ MCTS
        cpp_result = cpp_agent.search_from_cpp_board(cpp_board)
        cpp_visits = cpp_result.get_child_visits()

        # Both should have explored similar actions
        assert len(py_visits) > 0
        assert len(cpp_visits) > 0

        # Total visits should be close (both do 50 simulations)
        py_total = sum(py_visits.values())
        cpp_total = sum(cpp_visits.values())
        # First simulation evaluates root, so children get 49 visits
        assert py_total == 49
        assert cpp_total == 49

    def test_q_values_in_valid_range(self, py_agent, cpp_agent):
        """Test that Q-values are in valid range for both implementations."""
        board = FastGoBoard(9)
        py_state = GoState(board)
        cpp_board = alpha_go_cpp.GoBoard(9)

        # Run both searches
        np.random.seed(123)
        import torch
        torch.manual_seed(123)
        py_root = py_agent.search(py_state)

        np.random.seed(123)
        torch.manual_seed(123)
        cpp_result = cpp_agent.search_from_cpp_board(cpp_board)

        # Python Q-values
        assert 0.0 <= py_root.Q <= 1.0
        for child in py_root.children.values():
            assert 0.0 <= child.Q <= 1.0

        # C++ Q-values
        assert 0.0 <= cpp_result.Q <= 1.0
        for q in cpp_result.get_child_q_values().values():
            assert 0.0 <= q <= 1.0

    def test_action_probabilities_sum_to_one(self, py_agent, cpp_agent):
        """Test that action probabilities sum to 1 for both implementations."""
        board = FastGoBoard(9)
        py_state = GoState(board)
        cpp_board = alpha_go_cpp.GoBoard(9)

        # Run searches
        py_root = py_agent.search(py_state)
        cpp_result = cpp_agent.search_from_cpp_board(cpp_board)

        # Get probabilities
        py_probs = get_action_probabilities(py_root, temperature=1.0)
        cpp_probs = cpp_result.get_action_probabilities(temperature=1.0)

        # Both should sum to 1
        assert abs(sum(py_probs.values()) - 1.0) < 0.01
        assert abs(sum(cpp_probs.values()) - 1.0) < 0.01

    def test_top_actions_overlap(self, py_agent, cpp_agent):
        """Test that top actions are similar between Python and C++ MCTS."""
        board = FastGoBoard(9)
        py_state = GoState(board)
        cpp_board = alpha_go_cpp.GoBoard(9)

        # Run with same seed
        np.random.seed(42)
        import torch
        torch.manual_seed(42)
        py_root = py_agent.search(py_state)

        np.random.seed(42)
        torch.manual_seed(42)
        cpp_result = cpp_agent.search_from_cpp_board(cpp_board)

        # Get top 5 actions by visits
        py_probs = get_action_probabilities(py_root, temperature=1.0)
        cpp_probs = cpp_result.get_action_probabilities(temperature=1.0)

        py_top5 = sorted(py_probs.keys(), key=lambda a: py_probs[a], reverse=True)[:5]
        cpp_top5 = sorted(cpp_probs.keys(), key=lambda a: cpp_probs[a], reverse=True)[:5]

        # Top actions should have significant overlap
        # If both only have 1 action, they should agree on it
        overlap = set(py_top5) & set(cpp_top5)
        min_expected = min(len(py_top5), len(cpp_top5), 2)
        assert len(overlap) >= min(1, min_expected), f"Python top5: {py_top5}, C++ top5: {cpp_top5}"

        # The #1 action should be the same (most important check)
        assert py_top5[0] == cpp_top5[0], f"Top action mismatch: Python={py_top5[0]}, C++={cpp_top5[0]}"

    def test_deterministic_selection_valid(self, py_agent, cpp_agent):
        """Test that deterministic selection produces valid moves."""
        board = FastGoBoard(9)
        py_state = GoState(board)
        cpp_board = alpha_go_cpp.GoBoard(9)

        # Run searches
        py_root = py_agent.search(py_state)
        cpp_result = cpp_agent.search_from_cpp_board(cpp_board)

        # Get deterministic selections (temperature=0)
        py_probs = get_action_probabilities(py_root, temperature=0.0)
        cpp_probs = cpp_result.get_action_probabilities(temperature=0.0)

        # Both should have exactly one action with probability 1
        py_ones = sum(1 for p in py_probs.values() if p == 1.0)
        cpp_ones = sum(1 for p in cpp_probs.values() if p == 1.0)

        assert py_ones == 1
        assert cpp_ones == 1

    def test_cpp_search_result_interface(self, cpp_agent):
        """Test that CppSearchResult provides correct interface."""
        cpp_board = alpha_go_cpp.GoBoard(9)
        result = cpp_agent.search_from_cpp_board(cpp_board)

        # Check interface
        assert isinstance(result, CppSearchResult)
        assert result.N == 50  # Should match num_simulations
        assert 0.0 <= result.Q <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
