"""Monte Carlo Tree Search implementation following AlphaGo paper nomenclature.

This module implements MCTS with the following key concepts from the AlphaGo paper:
- N(s,a): Visit count for state-action pair
- Q(s,a): Mean action value (expected reward from taking action a in state s)
- P(s,a): Prior probability from policy network
- PUCT: Predicted Upper Confidence Trees for action selection

The implementation supports both:
- AlphaGo style: v_theta + lambda * rollout value
- AlphaZero style: pure v_theta (lambda=0, no rollout)

Dimension key:
    A: number of actions (board_size * board_size + 1 for pass)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, TypeVar
import numpy as np

# Type variable for actions
Action = TypeVar("Action")


class GameState(Protocol[Action]):
    """Protocol for game states compatible with MCTS."""

    def get_legal_actions(self) -> list[Action]: ...
    def apply_action(self, action: Action) -> "GameState[Action]": ...
    def is_terminal(self) -> bool: ...
    def get_reward(self, player: int) -> float: ...
    def current_player(self) -> int: ...
    def clone(self) -> "GameState[Action]": ...


@dataclass
class Node(Generic[Action]):
    """Node in the MCTS tree.

    This is a simple tree data structure that acts as a container for data only.
    All MCTS methods are defined functionally outside this class.

    Following AlphaGo paper notation:
        - N: Visit count through this node
        - Q: Action value (mean value of taking this node as an action)
        - logP_A: Log-probabilities of each possible action/child node from here

    Attributes:
        game_state: The game state at this node
        N: Total visit count through this node
        Q: Action value (expected reward from perspective of parent's player)
        logP_A: Dict mapping actions to log-prior probabilities from policy network
        children: Dict mapping actions to child nodes
        parent: Parent node (None for root)
        action_from_parent: Action that led to this node from parent
        player_at_parent: Which player made the move to reach this node
    """

    game_state: GameState[Action]
    N: int = 0
    Q: float = 0.0
    logP_A: dict[Action, float] = field(default_factory=dict)
    children: dict[Action, "Node[Action]"] = field(default_factory=dict)
    parent: "Node[Action] | None" = None
    action_from_parent: Action | None = None
    player_at_parent: int = 0


@dataclass
class MCTSConfig:
    """Configuration for MCTS.

    Attributes:
        c_puct: Exploration constant for PUCT formula (default 1.0)
        lambda_: Mixing parameter between value network and rollout (0=pure value, 1=pure rollout)
        dirichlet_alpha: Dirichlet noise alpha for root exploration (0 to disable)
        dirichlet_weight: Weight of Dirichlet noise at root (epsilon in paper)
        temperature: Temperature for final action selection
        max_rollout_depth: Maximum depth for fast rollout (if lambda > 0)
    """

    c_puct: float = 1.0
    lambda_: float = 0.0  # AlphaZero style: no rollout
    dirichlet_alpha: float = 0.0  # 0 means no noise
    dirichlet_weight: float = 0.25
    temperature: float = 1.0
    max_rollout_depth: int = 100


@dataclass
class RolloutStep(Generic[Action]):
    """Single step in a fast rollout."""
    action: Action
    state_before: Any = None  # Optional: game state before action (expensive to store)


@dataclass
class PlayoutTrace(Generic[Action]):
    """Trace of a single MCTS playout for debugging/visualization.

    Records the sequence of nodes visited and actions taken during one playout,
    including the fast rollout if lambda > 0.
    """

    nodes_visited: list[Node[Action]] = field(default_factory=list)
    actions_taken: list[Action] = field(default_factory=list)
    leaf_value: float = 0.0
    nn_value: float = 0.0  # Value from neural network (before mixing with rollout)
    rollout_value: float = 0.0  # Value from fast rollout
    is_terminal: bool = False
    expansion_occurred: bool = False
    rollout_steps: list[RolloutStep[Action]] = field(default_factory=list)
    leaf_state: Any = None  # Game state at leaf (for analysis)


def compute_puct_scores(
    node: Node[Action],
    c_puct: float,
) -> dict[Action, float]:
    """Compute PUCT scores for all legal actions.

    PUCT formula from AlphaGo paper:
        a = argmax_a (Q(s,a) + U(s,a))
        U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

    Args:
        node: Current node to select action from
        c_puct: Exploration constant

    Returns:
        Dict mapping each legal action to its PUCT score
    """
    total_visits = sum(
        node.children[a].N if a in node.children else 0
        for a in node.logP_A
    )
    sqrt_total = np.sqrt(total_visits + 1)  # +1 to handle N=0 case

    scores: dict[Action, float] = {}
    for action, log_prior in node.logP_A.items():
        prior = np.exp(log_prior)

        if action in node.children:
            child = node.children[action]
            q_value = child.Q
            n_visits = child.N
        else:
            q_value = 0.0
            n_visits = 0

        # PUCT formula
        u_value = c_puct * prior * sqrt_total / (1 + n_visits)
        scores[action] = q_value + u_value

    return scores


def select_action_puct(
    node: Node[Action],
    c_puct: float,
) -> Action:
    """Select action using PUCT algorithm.

    Args:
        node: Current node
        c_puct: Exploration constant

    Returns:
        Selected action
    """
    scores = compute_puct_scores(node, c_puct)
    return max(scores, key=lambda a: scores[a])


def is_game_over(state: GameState[Action]) -> bool:
    """Check if game state is terminal."""
    return state.is_terminal()


def get_utility_of_game_outcome(
    state: GameState[Action],
    player: int,
) -> float:
    """Get utility for a terminal game state.

    Args:
        state: Terminal game state
        player: Player to get reward for

    Returns:
        Utility value (typically 0 for loss, 1 for win)
    """
    return state.get_reward(player)


def fast_rollout(
    state: GameState[Action],
    select_action_fn: Callable[[GameState[Action]], Action],
    player: int,
    max_depth: int = 100,
    trace_steps: list[RolloutStep[Action]] | None = None,
) -> float:
    """Perform fast rollout using a simple policy.

    Args:
        state: Starting state for rollout
        select_action_fn: Function to select actions during rollout
        player: Player to get final reward for
        max_depth: Maximum rollout depth
        trace_steps: Optional list to append rollout steps to

    Returns:
        Utility estimate from rollout
    """
    current = state.clone()
    depth = 0

    while not current.is_terminal() and depth < max_depth:
        action = select_action_fn(current)
        if trace_steps is not None:
            trace_steps.append(RolloutStep(action=action))
        current = current.apply_action(action)
        depth += 1

    return current.get_reward(player)


def _perform_alphago_playout_impl(
    node: Node[Action],
    config: MCTSConfig,
    get_policy_and_value_fn: Callable[
        [GameState[Action]], tuple[dict[Action, float], float]
    ],
    rollout_policy_fn: Callable[[GameState[Action]], Action] | None = None,
    trace: PlayoutTrace[Action] | None = None,
    current_depth: int = 0,
) -> float:
    """Internal implementation of MCTS playout with depth tracking.

    This function implements:
    1. Selection: Use PUCT to traverse tree to leaf
    2. Expansion: Create new node if child doesn't exist
    3. Evaluation: Combine value network and rollout (AlphaGo) or just value (AlphaZero)
    4. Backup: Update N and Q values along path

    The value estimate is computed as:
        U = (1 - lambda) * v_theta + lambda * z_L

    Where:
        - v_theta: Value from neural network
        - z_L: Value from fast rollout
        - lambda: Mixing parameter (0 for AlphaZero, >0 for AlphaGo)

    Args:
        node: Root node to start playout from
        config: MCTS configuration
        get_policy_and_value_fn: Function that returns (policy_dict, value) for a state.
            Policy dict maps actions to log-probabilities.
            Value is from perspective of current player (0 to 1).
        rollout_policy_fn: Function to select actions during rollout (if lambda > 0)
        trace: Optional trace object to record playout steps
        current_depth: Current depth in the tree (for depth limiting)

    Returns:
        Value estimate of the playout (from perspective of node's parent's player)
    """
    if trace is not None:
        trace.nodes_visited.append(node)

    # Determine which player we're evaluating for (the player who took the action to reach this node)
    player_perspective = node.player_at_parent

    # Case 1: Terminal node - game is over
    if is_game_over(node.game_state):
        U = get_utility_of_game_outcome(node.game_state, player_perspective)
        if trace is not None:
            trace.is_terminal = True
            trace.leaf_value = U
            trace.leaf_state = node.game_state

    # Case 2: Depth limit reached - return value estimate without expanding
    elif current_depth >= config.max_rollout_depth:
        _, v_theta = get_policy_and_value_fn(node.game_state)
        current_player = node.game_state.current_player()
        if current_player != player_perspective:
            v_theta = 1.0 - v_theta
        U = v_theta
        if trace is not None:
            trace.leaf_value = U
            trace.nn_value = v_theta
            trace.leaf_state = node.game_state

    # Case 3: Leaf node not yet visited - evaluate and expand
    elif node.N == 0:
        # Get policy and value from neural network
        policy_dict, v_theta = get_policy_and_value_fn(node.game_state)

        # Store log-priors for this node
        node.logP_A = {
            action: np.log(prob + 1e-8) for action, prob in policy_dict.items()
        }

        # Value network gives value from current player's perspective
        # We need to convert to parent player's perspective
        current_player = node.game_state.current_player()
        if current_player != player_perspective:
            v_theta = 1.0 - v_theta

        # Optional: Mix with fast rollout (AlphaGo style)
        if config.lambda_ > 0 and rollout_policy_fn is not None:
            rollout_steps: list[RolloutStep[Action]] = [] if trace is not None else []
            z_L = fast_rollout(
                node.game_state,
                rollout_policy_fn,
                player_perspective,
                config.max_rollout_depth,
                trace_steps=rollout_steps if trace is not None else None,
            )
            U = (1 - config.lambda_) * v_theta + config.lambda_ * z_L
            if trace is not None:
                trace.rollout_steps = rollout_steps
                trace.rollout_value = z_L
        else:
            # AlphaZero style: pure value network
            U = v_theta

        if trace is not None:
            trace.expansion_occurred = True
            trace.leaf_value = U
            trace.nn_value = v_theta
            trace.leaf_state = node.game_state

    # Case 4: Internal node - select action and recurse
    else:
        action = select_action_puct(node, config.c_puct)

        if trace is not None:
            trace.actions_taken.append(action)

        # Expand if child doesn't exist
        if action not in node.children:
            new_game_state = node.game_state.apply_action(action)
            node.children[action] = Node(
                game_state=new_game_state,
                N=0,
                Q=0.0,
                parent=node,
                action_from_parent=action,
                player_at_parent=node.game_state.current_player(),
            )

        child = node.children[action]
        # Recurse - get value from child's perspective (child's parent = current node's player)
        child_value = _perform_alphago_playout_impl(
            child, config, get_policy_and_value_fn, rollout_policy_fn, trace,
            current_depth + 1,
        )
        # Child value is from current player's perspective, we need parent's perspective
        U = 1.0 - child_value

    # Backup: Update visit count and action value
    node.N += 1
    # Incremental mean update: Q = Q + (U - Q) / N
    node.Q = node.Q + (U - node.Q) / node.N

    return U


def perform_alphago_playout(
    node: Node[Action],
    config: MCTSConfig,
    get_policy_and_value_fn: Callable[
        [GameState[Action]], tuple[dict[Action, float], float]
    ],
    rollout_policy_fn: Callable[[GameState[Action]], Action] | None = None,
    trace: PlayoutTrace[Action] | None = None,
) -> float:
    """Perform a single MCTS playout following AlphaGo paper.

    This is a wrapper that calls the internal implementation with depth=0.
    See _perform_alphago_playout_impl for full documentation.

    Args:
        node: Root node to start playout from
        config: MCTS configuration
        get_policy_and_value_fn: Function that returns (policy_dict, value) for a state.
        rollout_policy_fn: Function to select actions during rollout (if lambda > 0)
        trace: Optional trace object to record playout steps

    Returns:
        Value estimate of the playout (from perspective of node's parent's player)
    """
    return _perform_alphago_playout_impl(
        node, config, get_policy_and_value_fn, rollout_policy_fn, trace, 0
    )


def add_dirichlet_noise(
    node: Node[Action],
    alpha: float,
    weight: float,
) -> None:
    """Add Dirichlet noise to root node priors for exploration.

    P'(s,a) = (1 - epsilon) * P(s,a) + epsilon * Dir(alpha)

    Args:
        node: Root node to add noise to
        alpha: Dirichlet distribution parameter
        weight: Weight of noise (epsilon)
    """
    if not node.logP_A:
        return

    actions = list(node.logP_A.keys())
    noise = np.random.dirichlet([alpha] * len(actions))

    for action, noise_val in zip(actions, noise):
        log_prior = node.logP_A[action]
        prior = np.exp(log_prior)
        noisy_prior = (1 - weight) * prior + weight * noise_val
        node.logP_A[action] = np.log(noisy_prior + 1e-8)

def run_mcts(
    root_state: GameState[Action],
    num_simulations: int,
    config: MCTSConfig,
    get_policy_and_value_fn: Callable[
        [GameState[Action]], tuple[dict[Action, float], float]
    ],
    rollout_policy_fn: Callable[[GameState[Action]], Action] | None = None,
) -> Node[Action]:
    """Run MCTS from a root state.

    Args:
        root_state: Starting game state
        num_simulations: Number of MCTS simulations to run
        config: MCTS configuration
        get_policy_and_value_fn: Policy and value function
        rollout_policy_fn: Rollout policy (if lambda > 0)

    Returns:
        Root node after search
    """
    # Initialize root
    root = Node(
        game_state=root_state,
        N=0,
        Q=0.0,
        player_at_parent=1 - root_state.current_player(),  # Opponent made last move
    )

    # Get initial policy for root
    policy_dict, _ = get_policy_and_value_fn(root_state)
    root.logP_A = {action: np.log(prob + 1e-8) for action, prob in policy_dict.items()}

    # Add exploration noise at root
    if config.dirichlet_alpha > 0:
        add_dirichlet_noise(root, config.dirichlet_alpha, config.dirichlet_weight)

    # Run simulations
    for _ in range(num_simulations):
        perform_alphago_playout(
            root, config, get_policy_and_value_fn, rollout_policy_fn, None
        )

    return root


def get_action_probabilities(
    root: Node[Action],
    temperature: float = 1.0,
) -> dict[Action, float]:
    """Get action probabilities from root node visit counts.

    pi(a|s) = N(s,a)^(1/tau) / sum_b N(s,b)^(1/tau)

    Args:
        root: Root node after MCTS
        temperature: Temperature parameter (tau). Lower = more deterministic.

    Returns:
        Dict mapping actions to probabilities
    """
    if not root.children:
        # No children expanded - return uniform over legal actions
        actions = list(root.logP_A.keys())
        return {a: 1.0 / len(actions) for a in actions}

    visit_counts = {
        action: child.N for action, child in root.children.items()
    }

    if temperature == 0:
        # Deterministic: pick action with most visits
        best_action = max(visit_counts, key=lambda a: visit_counts[a])
        return {a: 1.0 if a == best_action else 0.0 for a in visit_counts}

    # Apply temperature
    visits_temp = {
        a: count ** (1.0 / temperature) for a, count in visit_counts.items()
    }
    total = sum(visits_temp.values())

    if total == 0:
        return {a: 1.0 / len(visits_temp) for a in visits_temp}

    return {a: v / total for a, v in visits_temp.items()}


def select_action_from_mcts(
    root: Node[Action],
    temperature: float = 1.0,
) -> Action:
    """Select action from MCTS results.

    Args:
        root: Root node after MCTS
        temperature: Temperature for selection (0 = deterministic)

    Returns:
        Selected action
    """
    probs = get_action_probabilities(root, temperature)
    actions = list(probs.keys())
    probabilities = [probs[a] for a in actions]

    if temperature == 0:
        return actions[np.argmax(probabilities)]

    return actions[np.random.choice(len(actions), p=probabilities)]


# ============================================================================
# Compatibility layer for existing code
# ============================================================================


class State(Protocol[Action]):
    """Protocol alias for backward compatibility."""

    def get_legal_actions(self) -> list[Action]: ...
    def apply_action(self, action: Action) -> "State[Action]": ...
    def is_terminal(self) -> bool: ...
    def get_reward(self, player: int) -> float: ...
    def current_player(self) -> int: ...
    def clone(self) -> "State[Action]": ...


class Evaluator(Protocol[Action]):
    """Protocol for state evaluators (backward compatibility)."""

    def evaluate(
        self, state: State[Action]
    ) -> tuple[dict[Action, float], float]: ...


class MCTS(Generic[Action]):
    """High-level MCTS interface for backward compatibility with existing code.

    Wraps the functional MCTS implementation for easier use.
    """

    def __init__(
        self,
        evaluator: Evaluator[Action],
        c_puct: float = 1.0,
        num_simulations: int = 100,
        add_noise: bool = False,
        noise_alpha: float = 0.3,
        noise_weight: float = 0.25,
        lambda_: float = 0.0,
    ) -> None:
        """Initialize MCTS.

        Args:
            evaluator: Object with evaluate(state) method returning (policy, value)
            c_puct: Exploration constant
            num_simulations: Number of simulations per search
            add_noise: Whether to add Dirichlet noise at root
            noise_alpha: Dirichlet alpha parameter
            noise_weight: Weight of noise
            lambda_: Mixing parameter (0 for AlphaZero style)
        """
        self.evaluator = evaluator
        self.config = MCTSConfig(
            c_puct=c_puct,
            lambda_=lambda_,
            dirichlet_alpha=noise_alpha if add_noise else 0.0,
            dirichlet_weight=noise_weight,
        )
        self.num_simulations = num_simulations

    def _get_policy_and_value(
        self, state: State[Action]
    ) -> tuple[dict[Action, float], float]:
        """Wrapper around evaluator."""
        return self.evaluator.evaluate(state)

    def search(self, state: State[Action]) -> Node[Action]:
        """Run MCTS and return root node.

        Args:
            state: Starting state

        Returns:
            Root node after search
        """
        return run_mcts(
            root_state=state,
            num_simulations=self.num_simulations,
            config=self.config,
            get_policy_and_value_fn=self._get_policy_and_value,
        )

    def select_action(
        self, state: State[Action], temperature: float = 1.0
    ) -> Action:
        """Run MCTS and select action.

        Args:
            state: Starting state
            temperature: Temperature for action selection

        Returns:
            Selected action
        """
        root = self.search(state)
        return select_action_from_mcts(root, temperature)

    def get_action_probs(
        self, root: Node[Action], temperature: float = 1.0
    ) -> dict[Action, float]:
        """Get action probabilities from search result.

        Args:
            root: Root node from search
            temperature: Temperature parameter

        Returns:
            Action probability distribution
        """
        return get_action_probabilities(root, temperature)


# ============================================================================
# Trace serialization utilities
# ============================================================================


def trace_to_dict(trace: PlayoutTrace[Action]) -> dict[str, Any]:
    """Convert a PlayoutTrace to a serializable dictionary.

    Args:
        trace: The playout trace to serialize

    Returns:
        Dictionary with trace data suitable for JSON/pickle serialization
    """
    return {
        "actions_taken": list(trace.actions_taken),
        "leaf_value": trace.leaf_value,
        "nn_value": trace.nn_value,
        "rollout_value": trace.rollout_value,
        "is_terminal": trace.is_terminal,
        "expansion_occurred": trace.expansion_occurred,
        "rollout_actions": [step.action for step in trace.rollout_steps],
        "num_nodes_visited": len(trace.nodes_visited),
    }


def traces_to_list(traces: list[PlayoutTrace[Action]]) -> list[dict[str, Any]]:
    """Convert a list of traces to serializable format.

    Args:
        traces: List of playout traces

    Returns:
        List of dictionaries suitable for saving
    """
    return [trace_to_dict(t) for t in traces]
