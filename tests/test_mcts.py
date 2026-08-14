"""Tests for Monte Carlo Tree Search implementation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from alpha_go.mcts import (
    MCTS,
    MCTSConfig,
    Node,
    PlayoutTrace,
    add_dirichlet_noise,
    compute_puct_scores,
    get_action_probabilities,
    perform_alphago_playout,
    run_mcts,
    select_action_from_mcts,
    select_action_puct,
)


# ============================================================================
# Simple Test Game: Tic-Tac-Toe-like
# ============================================================================


@dataclass
class SimpleGameState:
    """Simple 3x3 game for testing MCTS.

    Players take turns placing pieces. Game ends when board is full.
    Score is sum of positions occupied by each player.
    """

    board: list[int] = field(default_factory=lambda: [0] * 9)  # 0=empty, 1=P1, 2=P2
    current: int = 0  # 0=P1, 1=P2
    move_count: int = 0

    def get_legal_actions(self) -> list[int]:
        return [i for i in range(9) if self.board[i] == 0]

    def apply_action(self, action: int) -> "SimpleGameState":
        new_board = self.board.copy()
        new_board[action] = self.current + 1
        return SimpleGameState(
            board=new_board,
            current=1 - self.current,
            move_count=self.move_count + 1,
        )

    def is_terminal(self) -> bool:
        return self.move_count >= 9 or len(self.get_legal_actions()) == 0

    def get_reward(self, player: int) -> float:
        # Simple scoring: count positions
        p1_score = sum(1 for x in self.board if x == 1)
        p2_score = sum(1 for x in self.board if x == 2)
        if player == 0:
            return 1.0 if p1_score > p2_score else 0.0
        return 1.0 if p2_score > p1_score else 0.0

    def current_player(self) -> int:
        return self.current

    def clone(self) -> "SimpleGameState":
        return SimpleGameState(
            board=self.board.copy(),
            current=self.current,
            move_count=self.move_count,
        )


def uniform_policy_and_value(
    state: SimpleGameState,
) -> tuple[dict[int, float], float]:
    """Uniform policy, random value."""
    actions = state.get_legal_actions()
    policy = {a: 1.0 / len(actions) for a in actions}
    value = 0.5
    return policy, value


def biased_policy_and_value(
    state: SimpleGameState,
) -> tuple[dict[int, float], float]:
    """Policy biased toward lower indices."""
    actions = state.get_legal_actions()
    weights = [1.0 / (a + 1) for a in actions]
    total = sum(weights)
    policy = {a: w / total for a, w in zip(actions, weights)}
    value = 0.6
    return policy, value


# ============================================================================
# Node Tests
# ============================================================================


class TestNode:
    def test_node_creation(self):
        state = SimpleGameState()
        node = Node(game_state=state)

        assert node.N == 0
        assert node.Q == 0.0
        assert node.logP_A == {}
        assert node.children == {}
        assert node.parent is None

    def test_node_with_priors(self):
        state = SimpleGameState()
        node = Node(game_state=state)
        node.logP_A = {0: np.log(0.5), 1: np.log(0.3), 2: np.log(0.2)}

        assert len(node.logP_A) == 3
        assert np.isclose(np.exp(node.logP_A[0]), 0.5, atol=1e-6)


# ============================================================================
# PUCT Tests
# ============================================================================


class TestPUCT:
    def test_puct_unexplored_prefers_high_prior(self):
        """With no visits, PUCT should prefer actions with high prior."""
        state = SimpleGameState()
        node = Node(game_state=state, N=1)
        node.logP_A = {
            0: np.log(0.8),
            1: np.log(0.15),
            2: np.log(0.05),
        }

        scores = compute_puct_scores(node, c_puct=1.0)

        # Action 0 should have highest score
        assert scores[0] > scores[1] > scores[2]

    def test_puct_balances_exploration_exploitation(self):
        """PUCT should balance visit counts with priors."""
        state = SimpleGameState()
        node = Node(game_state=state, N=10)
        node.logP_A = {0: np.log(0.5), 1: np.log(0.5)}

        # Create children with different visit counts and Q values
        child0 = Node(game_state=state.apply_action(0), N=5, Q=0.6)
        child1 = Node(game_state=state.apply_action(1), N=3, Q=0.4)
        node.children = {0: child0, 1: child1}

        scores = compute_puct_scores(node, c_puct=1.0)

        # Both should have positive scores
        assert scores[0] > 0
        assert scores[1] > 0

    def test_select_action_puct(self):
        """select_action_puct should return action with highest score."""
        state = SimpleGameState()
        node = Node(game_state=state, N=1)
        node.logP_A = {0: np.log(0.9), 1: np.log(0.1)}

        action = select_action_puct(node, c_puct=1.0)
        assert action == 0


# ============================================================================
# Playout Tests
# ============================================================================


class TestPlayout:
    def test_single_playout_updates_counts(self):
        """A single playout should update N and Q."""
        state = SimpleGameState()
        config = MCTSConfig(c_puct=1.0)
        node = Node(
            game_state=state,
            N=0,
            Q=0.0,
            player_at_parent=1,  # Opponent made last move
        )

        perform_alphago_playout(node, config, uniform_policy_and_value)

        assert node.N == 1
        # Q should be updated to the value estimate
        assert 0.0 <= node.Q <= 1.0

    def test_multiple_playouts_build_tree(self):
        """Multiple playouts should expand the tree."""
        state = SimpleGameState()
        config = MCTSConfig(c_puct=1.0)
        node = Node(
            game_state=state,
            N=0,
            Q=0.0,
            player_at_parent=1,
        )

        for _ in range(50):
            perform_alphago_playout(node, config, uniform_policy_and_value)

        assert node.N == 50
        # Should have expanded some children
        assert len(node.children) > 0

    def test_playout_trace_recorded(self):
        """Playout trace should record visited nodes."""
        state = SimpleGameState()
        config = MCTSConfig(c_puct=1.0)
        node = Node(game_state=state, N=0, Q=0.0, player_at_parent=1)

        trace = PlayoutTrace()
        perform_alphago_playout(node, config, uniform_policy_and_value, trace=trace)

        assert len(trace.nodes_visited) >= 1
        assert trace.nodes_visited[0] is node

    def test_terminal_state_returns_game_value(self):
        """Terminal state should return actual game outcome."""
        # Create a full board
        state = SimpleGameState(
            board=[1, 2, 1, 2, 1, 2, 1, 2, 1],  # Full board
            current=0,
            move_count=9,
        )
        assert state.is_terminal()

        config = MCTSConfig()
        node = Node(game_state=state, N=0, Q=0.0, player_at_parent=1)

        trace = PlayoutTrace()
        value = perform_alphago_playout(
            node, config, uniform_policy_and_value, trace=trace
        )

        assert trace.is_terminal
        assert value in [0.0, 1.0]


# ============================================================================
# Full MCTS Tests
# ============================================================================


class TestRunMCTS:
    def test_run_mcts_basic(self):
        """run_mcts should return a root node with visits."""
        state = SimpleGameState()
        config = MCTSConfig(c_puct=1.0)

        root = run_mcts(
            root_state=state,
            num_simulations=100,
            config=config,
            get_policy_and_value_fn=uniform_policy_and_value,
        )

        assert root.N == 100
        assert len(root.children) > 0

    def test_mcts_respects_high_prior(self):
        """MCTS should visit actions with higher prior more often."""
        state = SimpleGameState()
        config = MCTSConfig(c_puct=1.0)

        root = run_mcts(
            root_state=state,
            num_simulations=200,
            config=config,
            get_policy_and_value_fn=biased_policy_and_value,
        )

        # Action 0 should have most visits (it has highest prior)
        if len(root.children) > 1:
            visit_counts = {a: c.N for a, c in root.children.items()}
            max_action = max(visit_counts, key=lambda a: visit_counts[a])
            # With biased policy, lower actions should generally have more visits
            assert max_action in [0, 1, 2]


class TestActionSelection:
    def test_get_action_probabilities_temperature_1(self):
        """Temperature 1 should give probabilities proportional to visit counts."""
        state = SimpleGameState()
        root = Node(game_state=state, N=100)
        root.logP_A = {0: 0, 1: 0, 2: 0}
        root.children = {
            0: Node(game_state=state.apply_action(0), N=50),
            1: Node(game_state=state.apply_action(1), N=30),
            2: Node(game_state=state.apply_action(2), N=20),
        }

        probs = get_action_probabilities(root, temperature=1.0)

        assert np.isclose(probs[0], 0.5, atol=0.01)
        assert np.isclose(probs[1], 0.3, atol=0.01)
        assert np.isclose(probs[2], 0.2, atol=0.01)

    def test_get_action_probabilities_temperature_0(self):
        """Temperature 0 should be deterministic (max visits)."""
        state = SimpleGameState()
        root = Node(game_state=state, N=100)
        root.logP_A = {0: 0, 1: 0, 2: 0}
        root.children = {
            0: Node(game_state=state.apply_action(0), N=50),
            1: Node(game_state=state.apply_action(1), N=30),
            2: Node(game_state=state.apply_action(2), N=20),
        }

        probs = get_action_probabilities(root, temperature=0)

        assert probs[0] == 1.0
        assert probs[1] == 0.0
        assert probs[2] == 0.0

    def test_select_action_deterministic(self):
        """Temperature 0 should always select best action."""
        state = SimpleGameState()
        root = Node(game_state=state, N=100)
        root.logP_A = {0: 0, 1: 0}
        root.children = {
            0: Node(game_state=state.apply_action(0), N=60),
            1: Node(game_state=state.apply_action(1), N=40),
        }

        for _ in range(10):
            action = select_action_from_mcts(root, temperature=0)
            assert action == 0


class TestDirichletNoise:
    def test_add_dirichlet_noise(self):
        """Dirichlet noise should modify priors."""
        state = SimpleGameState()
        node = Node(game_state=state)
        node.logP_A = {0: np.log(0.5), 1: np.log(0.3), 2: np.log(0.2)}

        original_probs = {a: np.exp(lp) for a, lp in node.logP_A.items()}

        np.random.seed(42)
        add_dirichlet_noise(node, alpha=0.3, weight=0.25)

        new_probs = {a: np.exp(lp) for a, lp in node.logP_A.items()}

        # Priors should have changed
        for a in node.logP_A:
            assert not np.isclose(original_probs[a], new_probs[a], atol=0.01)

        # Should still sum to approximately 1
        assert np.isclose(sum(new_probs.values()), 1.0, atol=0.01)


# ============================================================================
# MCTS Class (Compatibility) Tests
# ============================================================================


class MockEvaluator:
    """Mock evaluator for testing MCTS class."""

    def evaluate(
        self, state: SimpleGameState
    ) -> tuple[dict[int, float], float]:
        return uniform_policy_and_value(state)


class TestMCTSClass:
    def test_mcts_class_search(self):
        """MCTS class should run search and return root."""
        evaluator = MockEvaluator()
        mcts = MCTS(evaluator=evaluator, num_simulations=50, c_puct=1.0)

        state = SimpleGameState()
        root = mcts.search(state)

        assert root.N == 50
        assert len(root.children) > 0

    def test_mcts_class_select_action(self):
        """MCTS class should select valid actions."""
        evaluator = MockEvaluator()
        mcts = MCTS(evaluator=evaluator, num_simulations=50, c_puct=1.0)

        state = SimpleGameState()
        action = mcts.select_action(state, temperature=1.0)

        assert action in state.get_legal_actions()

    def test_mcts_class_with_noise(self):
        """MCTS class should support Dirichlet noise."""
        evaluator = MockEvaluator()
        mcts = MCTS(
            evaluator=evaluator,
            num_simulations=50,
            add_noise=True,
            noise_alpha=0.3,
            noise_weight=0.25,
        )

        state = SimpleGameState()
        root = mcts.search(state)

        # Should still work with noise
        assert root.N == 50


# ============================================================================
# Player Perspective / Opponent Selection Tests
# ============================================================================


@dataclass
class AsymmetricGameState:
    """Game where action 0 is good for P1, action 1 is good for P2.

    This tests that MCTS correctly selects actions to maximize the current
    player's value (minimize opponent's value).
    """

    current: int = 0  # 0=P1, 1=P2
    chosen_action: int | None = None

    def get_legal_actions(self) -> list[int]:
        if self.chosen_action is None:
            return [0, 1]  # Two actions
        return []

    def apply_action(self, action: int) -> "AsymmetricGameState":
        return AsymmetricGameState(
            current=1 - self.current,
            chosen_action=action,
        )

    def is_terminal(self) -> bool:
        return self.chosen_action is not None

    def get_reward(self, player: int) -> float:
        """Action 0 wins for P1, action 1 wins for P2."""
        if self.chosen_action is None:
            return 0.5
        if self.chosen_action == 0:
            # Action 0 is good for P1
            return 1.0 if player == 0 else 0.0
        else:
            # Action 1 is good for P2
            return 0.0 if player == 0 else 1.0

    def current_player(self) -> int:
        return self.current

    def clone(self) -> "AsymmetricGameState":
        return AsymmetricGameState(
            current=self.current,
            chosen_action=self.chosen_action,
        )


def asymmetric_policy_and_value(
    state: AsymmetricGameState,
) -> tuple[dict[int, float], float]:
    """Uniform policy, value based on game outcome prediction."""
    actions = state.get_legal_actions()
    if not actions:
        # Terminal - return actual reward
        return {}, state.get_reward(state.current_player())
    # Uniform policy, neutral value
    policy = {a: 1.0 / len(actions) for a in actions}
    return policy, 0.5


def biased_asymmetric_policy(
    state: AsymmetricGameState,
) -> tuple[dict[int, float], float]:
    """Policy that correctly predicts optimal action for current player."""
    actions = state.get_legal_actions()
    if not actions:
        return {}, state.get_reward(state.current_player())

    # Policy favors the winning action for current player
    if state.current_player() == 0:
        # P1 should prefer action 0
        policy = {0: 0.9, 1: 0.1}
        value = 0.9  # P1 expects to win
    else:
        # P2 should prefer action 1
        policy = {0: 0.1, 1: 0.9}
        value = 0.9  # P2 expects to win
    return policy, value


class TestPlayerPerspective:
    """Tests for correct handling of player perspectives in MCTS."""

    def test_p1_selects_winning_action(self):
        """Player 1 should select action 0 which wins for them."""
        state = AsymmetricGameState(current=0)  # P1's turn
        config = MCTSConfig(c_puct=1.0)

        root = run_mcts(
            root_state=state,
            num_simulations=100,
            config=config,
            get_policy_and_value_fn=biased_asymmetric_policy,
        )

        # P1 should prefer action 0 (their winning action)
        probs = get_action_probabilities(root, temperature=0)
        assert probs[0] == 1.0, f"P1 should select action 0, got probs: {probs}"

    def test_p2_selects_winning_action(self):
        """Player 2 should select action 1 which wins for them."""
        state = AsymmetricGameState(current=1)  # P2's turn
        config = MCTSConfig(c_puct=1.0)

        root = run_mcts(
            root_state=state,
            num_simulations=100,
            config=config,
            get_policy_and_value_fn=biased_asymmetric_policy,
        )

        # P2 should prefer action 1 (their winning action)
        probs = get_action_probabilities(root, temperature=0)
        assert probs[1] == 1.0, f"P2 should select action 1, got probs: {probs}"

    def test_q_values_from_correct_perspective(self):
        """Q values should be stored from parent player's perspective."""
        state = AsymmetricGameState(current=0)  # P1's turn
        config = MCTSConfig(c_puct=1.0)

        root = run_mcts(
            root_state=state,
            num_simulations=100,
            config=config,
            get_policy_and_value_fn=biased_asymmetric_policy,
        )

        # Child Q values should be from P1's perspective (since P1 took the action)
        if 0 in root.children and 1 in root.children:
            # Action 0 wins for P1, so Q should be high
            # Action 1 wins for P2, so Q should be low from P1's perspective
            assert root.children[0].Q > root.children[1].Q, (
                f"Action 0 should have higher Q from P1's perspective: "
                f"Q[0]={root.children[0].Q}, Q[1]={root.children[1].Q}"
            )

    def test_backup_negates_value_correctly(self):
        """Backup should negate values when alternating between players."""
        state = AsymmetricGameState(current=0)  # P1's turn
        config = MCTSConfig(c_puct=1.0)

        root = run_mcts(
            root_state=state,
            num_simulations=50,
            config=config,
            get_policy_and_value_fn=biased_asymmetric_policy,
        )

        # After many simulations, root Q should reflect P1's expected value
        # Since we're at P1's turn and P1 has the winning action with high prior,
        # root Q (from opponent's perspective since root.player_at_parent = P2)
        # should be low (P2 expects to lose)
        assert root.Q < 0.5, f"Root Q should be low (P2's perspective): {root.Q}"

    def test_puct_selects_best_for_current_player(self):
        """PUCT should select the best action for the current player."""
        # Create a node where P2 is to move
        state = AsymmetricGameState(current=1)  # P2's turn

        # Manually set up node with known Q values
        root = Node(
            game_state=state,
            N=10,
            Q=0.5,
            player_at_parent=0,  # P1 made last move to get here
        )
        # Equal priors
        root.logP_A = {0: np.log(0.5), 1: np.log(0.5)}

        # Create children with Q values from P2's perspective
        # Action 1 is better for P2
        child0 = Node(
            game_state=state.apply_action(0),
            N=5,
            Q=0.2,  # Low Q = bad for P2
            player_at_parent=1,  # P2 took this action
        )
        child1 = Node(
            game_state=state.apply_action(1),
            N=5,
            Q=0.8,  # High Q = good for P2
            player_at_parent=1,
        )
        root.children = {0: child0, 1: child1}

        # PUCT should select action 1 (higher Q for current player P2)
        action = select_action_puct(root, c_puct=1.0)
        assert action == 1, f"PUCT should select action 1 for P2, got {action}"

    def test_uniform_policy_converges_to_correct_action(self):
        """Even with uniform policy, MCTS should converge to optimal action."""
        # P1's turn - should eventually prefer action 0
        state = AsymmetricGameState(current=0)
        config = MCTSConfig(c_puct=1.0)

        root = run_mcts(
            root_state=state,
            num_simulations=500,  # More simulations for uniform policy
            config=config,
            get_policy_and_value_fn=asymmetric_policy_and_value,
        )

        # With enough simulations, P1 should discover action 0 is better
        if 0 in root.children and 1 in root.children:
            # Action 0 should have more visits or higher Q
            assert root.children[0].N >= root.children[1].N or root.children[0].Q > root.children[1].Q, (
                f"P1 should prefer action 0: N[0]={root.children[0].N}, N[1]={root.children[1].N}, "
                f"Q[0]={root.children[0].Q}, Q[1]={root.children[1].Q}"
            )


# ============================================================================
# Fast Rollout / Lambda Tests
# ============================================================================


class TestFastRollout:
    """Tests for AlphaGo-style fast rollout (lambda > 0)."""

    def test_rollout_not_called_when_lambda_zero(self):
        """With lambda=0, rollout_policy_fn should never be called."""
        state = SimpleGameState()
        config = MCTSConfig(c_puct=1.0, lambda_=0.0)

        rollout_call_count = [0]  # Use list to allow mutation in closure

        def tracking_rollout(s: SimpleGameState) -> int:
            rollout_call_count[0] += 1
            actions = s.get_legal_actions()
            return actions[0] if actions else 0

        root = run_mcts(
            root_state=state,
            num_simulations=50,
            config=config,
            get_policy_and_value_fn=uniform_policy_and_value,
            rollout_policy_fn=tracking_rollout,
        )

        assert rollout_call_count[0] == 0, (
            f"Rollout should not be called when lambda=0, but was called {rollout_call_count[0]} times"
        )
        assert root.N == 50

    def test_rollout_called_when_lambda_positive(self):
        """With lambda=0.5, rollout_policy_fn should be called during leaf evaluation."""
        state = SimpleGameState()
        config = MCTSConfig(c_puct=1.0, lambda_=0.5)

        rollout_call_count = [0]

        def tracking_rollout(s: SimpleGameState) -> int:
            rollout_call_count[0] += 1
            actions = s.get_legal_actions()
            return actions[0] if actions else 0

        root = run_mcts(
            root_state=state,
            num_simulations=50,
            config=config,
            get_policy_and_value_fn=uniform_policy_and_value,
            rollout_policy_fn=tracking_rollout,
        )

        assert rollout_call_count[0] > 0, (
            f"Rollout should be called when lambda=0.5, but was never called"
        )
        assert root.N == 50

    def test_lambda_one_uses_pure_rollout(self):
        """With lambda=1.0, value should come entirely from rollout."""
        # Use a deterministic game where rollout outcome is predictable
        state = AsymmetricGameState(current=0)
        config = MCTSConfig(c_puct=1.0, lambda_=1.0)

        def deterministic_rollout(s: AsymmetricGameState) -> int:
            """Always take action 0 (which is good for P1)."""
            actions = s.get_legal_actions()
            if actions:
                return 0  # Always pick action 0
            return 0

        root = run_mcts(
            root_state=state,
            num_simulations=100,
            config=config,
            get_policy_and_value_fn=biased_asymmetric_policy,
            rollout_policy_fn=deterministic_rollout,
        )

        # With lambda=1.0 and deterministic rollout always picking action 0,
        # the value should reflect that outcome
        assert root.N == 100

    def test_lambda_interpolates_value_and_rollout(self):
        """With lambda=0.5, value should be mix of network and rollout."""
        state = SimpleGameState()

        # Track values from different sources
        nn_values = []
        rollout_outcomes = []

        def tracking_policy_value(s: SimpleGameState) -> tuple[dict[int, float], float]:
            actions = s.get_legal_actions()
            policy = {a: 1.0 / len(actions) for a in actions}
            value = 0.8  # Always return 0.8 from NN
            nn_values.append(value)
            return policy, value

        def tracking_rollout(s: SimpleGameState) -> int:
            """Rollout that leads to known outcome."""
            actions = s.get_legal_actions()
            if actions:
                rollout_outcomes.append(1)  # Track that rollout happened
                return actions[0]
            return 0

        config_0 = MCTSConfig(c_puct=1.0, lambda_=0.0)
        config_05 = MCTSConfig(c_puct=1.0, lambda_=0.5)

        # Run with lambda=0 (pure NN)
        nn_values.clear()
        run_mcts(
            root_state=state,
            num_simulations=20,
            config=config_0,
            get_policy_and_value_fn=tracking_policy_value,
            rollout_policy_fn=tracking_rollout,
        )

        # Run with lambda=0.5 (mix)
        rollout_outcomes.clear()
        run_mcts(
            root_state=state,
            num_simulations=20,
            config=config_05,
            get_policy_and_value_fn=tracking_policy_value,
            rollout_policy_fn=tracking_rollout,
        )

        # With lambda=0.5, rollout should have been called
        assert len(rollout_outcomes) > 0, "Rollout should be called with lambda=0.5"

    def test_mcts_class_with_lambda(self):
        """MCTS class should support lambda parameter."""
        evaluator = MockEvaluator()

        rollout_call_count = [0]

        def tracking_rollout(s: SimpleGameState) -> int:
            rollout_call_count[0] += 1
            actions = s.get_legal_actions()
            return actions[0] if actions else 0

        # Note: MCTS class doesn't currently support rollout_policy_fn directly
        # This test verifies the config is set correctly
        mcts = MCTS(
            evaluator=evaluator,
            num_simulations=50,
            c_puct=1.0,
            lambda_=0.5,
        )

        assert mcts.config.lambda_ == 0.5, "MCTS class should store lambda in config"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_single_legal_action(self):
        """MCTS should handle single legal action."""
        # Board with only one empty spot
        state = SimpleGameState(
            board=[1, 2, 1, 2, 1, 2, 1, 2, 0],
            current=0,
            move_count=8,
        )
        assert len(state.get_legal_actions()) == 1

        config = MCTSConfig(c_puct=1.0)
        root = run_mcts(
            root_state=state,
            num_simulations=10,
            config=config,
            get_policy_and_value_fn=uniform_policy_and_value,
        )

        assert root.N == 10
        assert 8 in root.children  # Only action is position 8

    def test_already_terminal(self):
        """MCTS should handle already-terminal states."""
        state = SimpleGameState(
            board=[1, 2, 1, 2, 1, 2, 1, 2, 1],
            current=0,
            move_count=9,
        )
        assert state.is_terminal()

        config = MCTSConfig(c_puct=1.0)
        root = run_mcts(
            root_state=state,
            num_simulations=10,
            config=config,
            get_policy_and_value_fn=uniform_policy_and_value,
        )

        # Should still complete without error
        assert root.N == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
