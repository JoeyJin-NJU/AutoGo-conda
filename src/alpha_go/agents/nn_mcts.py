"""Neural network MCTS agent that uses MCTS with a neural network for Go."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable
import os
import alpha_go_cpp
import grpc
import numpy as np
import torch
import torch.nn.functional as F

from alpha_go.agents.base import Agent, PASS, RESIGN, get_pass_index, register_agent
from alpha_go.go import EMPTY, BLACK, WHITE, FastGoBoard, GoState
from alpha_go.mcts import (
    MCTSConfig,
    Node,
    State,
    Evaluator,
    run_mcts,
    select_action_from_mcts,
    get_action_probabilities,
)
from alpha_go.model import (
    SizeInvariantGoResNet,
    create_mup_model,
    MuPModelConfig,
    upgrade_state_dict_for_pass,
)
from alpha_go.proto import inference_pb2, inference_pb2_grpc


# ============================================================================
# MCTS Evaluator Protocol (for C++ MCTS)
# ============================================================================


@runtime_checkable
class CppMCTSEvaluator(Protocol):
    """Protocol for C++ MCTS evaluators.

    Evaluators provide policy and value estimates for board positions.
    Different implementations can use local models, RPC servers, or other backends.
    """

    board_size: int

    def evaluate(self, cpp_board: Any) -> tuple[dict[int, float], float]:
        """Evaluate a board position.

        Args:
            cpp_board: C++ GoBoard instance

        Returns:
            Tuple of (policy_dict, value) where:
            - policy_dict maps flat action indices to probabilities
            - value is the estimated win probability for current player
        """
        ...

    def close(self) -> None:
        """Clean up any resources (e.g., gRPC channels)."""
        ...


# ============================================================================
# Evaluator Implementations
# ============================================================================


class LocalNNEvaluator:
    """Evaluator using a local neural network model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        board_size: int = 9,
        model_config: str = "10M",
        device: str | None = None,
        policy_temperature: float = 1.0,
    ) -> None:
        """Initialize with a local model checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint (.pt file)
            board_size: Size of the Go board
            model_config: Model config name (e.g., "10M", "100M")
            device: Device to run inference on (default: auto-detect)
            policy_temperature: Temperature for policy softmax (< 1 = more peaked, > 1 = more uniform)
        """
        self._checkpoint_path = Path(checkpoint_path)
        self.board_size = board_size
        self.policy_temperature = policy_temperature
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Create and load model (with backward compatibility for old checkpoints)
        self.model = create_mup_model(
            config=model_config, board_size=board_size, device=self.device
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        state_dict = upgrade_state_dict_for_pass(checkpoint["model_state_dict"], board_size)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, cpp_board: Any) -> tuple[dict[int, float], float]:
        """Evaluate C++ board using neural network."""
        board_np = cpp_board.to_numpy().astype(np.float32)

        # Normalize: current player sees themselves as 1
        if cpp_board.to_play() == alpha_go_cpp.GoBoard.WHITE:
            board_np = np.where(board_np == 1, 2.0, np.where(board_np == 2, 1.0, board_np))

        board_tensor = torch.from_numpy(board_np).unsqueeze(0).to(self.device)

        # Forward pass - policy now has n_actions = board_size^2 + 1 (including pass)
        policy_logits, value_logits = self.model(board_tensor)
        value = torch.sigmoid(value_logits).item()

        # Get legal moves for masking
        legal_moves = cpp_board.get_legal_moves_flat()
        pass_index = get_pass_index(self.board_size)
        n_actions = pass_index + 1

        # Create mask for legal moves (including pass)
        policy_np = policy_logits.squeeze(0).cpu().numpy()
        mask = np.full(n_actions, float("-inf"))
        for idx in legal_moves:
            mask[idx] = 0.0
        # Pass is always legal
        mask[pass_index] = 0.0

        masked_logits = policy_np + mask

        # Apply temperature: lower temp = more peaked distribution
        scaled_logits = masked_logits / self.policy_temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / exp_logits.sum()

        # Build policy dict including pass action
        policy_dict: dict[int, float] = {
            idx: float(probs[idx]) for idx in legal_moves
        }
        # Add pass action with its probability (PASS_ACTION = -1 in C++)
        policy_dict[alpha_go_cpp.PASS_ACTION] = float(probs[pass_index])

        return (policy_dict, value)

    @property
    def checkpoint_path(self) -> str:
        """Return checkpoint path."""
        return str(self._checkpoint_path)

    def close(self) -> None:
        """No resources to clean up for local model."""
        pass


class LeafBatchedNNEvaluator:
    """Evaluator that exposes `batch_evaluate(list[cpp_board]) -> list[(policy_dict, value)]`
    for use with MCTSTree.run_simulations_batched (leaf-parallel MCTS + virtual loss).

    Owns its own model. Single Python callback per batch of leaves, one GPU forward
    per batch. Intended for single-game use (num_workers=1).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        board_size: int = 9,
        model_config: str = "18M",
        device: str | None = None,
        policy_temperature: float = 1.0,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self.board_size = board_size
        self.policy_temperature = policy_temperature
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        state_dict = checkpoint["model_state_dict"]
        config_name = str(checkpoint.get("config", ""))
        if config_name.startswith("SizeInvariantGoResNet") or "pass_fc.weight" in state_dict:
            # Shape fingerprint: channels from input_conv, n_blocks by counting
            # block conv1 params, value_hidden from value_fc2. Also detect the
            # optional GroupNorm + SE variant from state_dict key signature:
            #   - "input_bn.running_mean" present → BN (legacy / v0 default)
            #     absent                            → GroupNorm
            #   - "blocks.0.se.fc1.weight" present → SE blocks enabled
            channels = int(state_dict["input_conv.weight"].shape[0])
            n_blocks = sum(
                1 for k in state_dict if k.startswith("blocks.") and k.endswith(".conv1.weight")
            )
            value_hidden = int(state_dict["value_fc2.weight"].shape[1])
            norm_type = "bn" if "input_bn.running_mean" in state_dict else "gn"
            use_se = "blocks.0.se.fc1.weight" in state_dict
            se_reduction = 8
            if use_se:
                se_hidden = int(state_dict["blocks.0.se.fc1.weight"].shape[0])
                se_reduction = max(1, channels // max(1, se_hidden))
            self.model = SizeInvariantGoResNet(
                channels=channels, n_blocks=n_blocks, value_hidden=value_hidden,
                norm_type=norm_type, use_se=use_se, se_reduction=se_reduction,
            ).to(self.device)
            self.model.load_state_dict(state_dict)
        else:
            self.model = create_mup_model(
                config=model_config, board_size=board_size, device=self.device
            )
            state_dict = upgrade_state_dict_for_pass(state_dict, board_size)
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self._pass_index = get_pass_index(board_size)
        self._n_actions = self._pass_index + 1

    @torch.no_grad()
    def evaluate(self, cpp_board: Any) -> tuple[dict[int, float], float]:
        """Single-board eval (used by MCTSTree.run_simulations fallback)."""
        return self.batch_evaluate([cpp_board])[0]

    @torch.no_grad()
    def batch_evaluate(
        self, cpp_boards: list[Any]
    ) -> list[tuple[dict[int, float], float]]:
        B = len(cpp_boards)
        boards_np = np.empty((B, self.board_size, self.board_size), dtype=np.float32)
        for i, b in enumerate(cpp_boards):
            arr = b.to_numpy().astype(np.float32)
            if b.to_play() == alpha_go_cpp.GoBoard.WHITE:
                arr = np.where(arr == 1, 2.0, np.where(arr == 2, 1.0, arr))
            boards_np[i] = arr
        boards_t = torch.from_numpy(boards_np).to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            policy_logits, value_logits = self.model(boards_t)
        values = torch.sigmoid(value_logits).float().cpu().numpy().reshape(-1)
        policy_np = policy_logits.float().cpu().numpy()  # (B, n_actions)

        results: list[tuple[dict[int, float], float]] = []
        for i, b in enumerate(cpp_boards):
            legal = b.get_legal_moves_flat()
            mask = np.full(self._n_actions, -np.inf, dtype=np.float32)
            for idx in legal:
                mask[idx] = 0.0
            mask[self._pass_index] = 0.0
            logits = policy_np[i] + mask
            logits = logits / self.policy_temperature
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            d: dict[int, float] = {int(idx): float(probs[idx]) for idx in legal}
            d[alpha_go_cpp.PASS_ACTION] = float(probs[self._pass_index])
            results.append((d, float(values[i])))
        return results

    @property
    def checkpoint_path(self) -> str:
        return str(self._checkpoint_path)

    def close(self) -> None:
        pass


class BatchedLocalNNEvaluator:
    """Evaluator using LocalBatchedInferenceEngine for high-throughput multi-thread inference.

    Unlike LocalNNEvaluator which does single-position inference, this evaluator
    submits requests to a shared batching engine. When multiple game threads use
    evaluators backed by the same engine, their requests are batched together for
    efficient GPU utilization.

    Usage:
        # Create shared engine and model
        model = create_mup_model(config="10M", board_size=9, device=device)
        engine = LocalBatchedInferenceEngine(model, device, batch_size=64)
        engine.start()

        # Create evaluators for each thread (all share the same engine)
        evaluator1 = BatchedLocalNNEvaluator(engine, board_size=9)
        evaluator2 = BatchedLocalNNEvaluator(engine, board_size=9)

        # Use with CppMCTSAgent
        agent1 = CppMCTSAgent(evaluator=evaluator1, ...)
        agent2 = CppMCTSAgent(evaluator=evaluator2, ...)

        # Clean up
        engine.stop()
    """

    def __init__(
        self,
        engine: "LocalBatchedInferenceEngine",  # type: ignore
        board_size: int = 9,
        policy_temperature: float = 1.0,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        """Initialize with a shared batched inference engine.

        Args:
            engine: Shared LocalBatchedInferenceEngine instance
            board_size: Size of the Go board
            policy_temperature: Temperature for policy softmax
            checkpoint_path: Optional path for tracking (not used for inference)
        """
        self.engine = engine
        self.board_size = board_size
        self.policy_temperature = policy_temperature
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    def evaluate(self, cpp_board: Any) -> tuple[dict[int, float], float]:
        """Evaluate C++ board using batched inference engine. Board size is
        read from `cpp_board.size()` per call so a single shared evaluator
        works across mixed-size games (the engine pads + masks internally
        and returns native-sized logits)."""
        board_np = cpp_board.to_numpy().astype(np.float32)

        # Normalize: current player sees themselves as 1
        if cpp_board.to_play() == alpha_go_cpp.GoBoard.WHITE:
            board_np = np.where(board_np == 1, 2.0, np.where(board_np == 2, 1.0, board_np))

        # Submit to batched engine and wait for result
        future = self.engine.submit(board_np)
        policy_logits, value, _entropy = future.result()

        # Get legal moves for masking. Use the live board's size so a single
        # evaluator instance can serve any size the engine accepts.
        bs = int(cpp_board.size())
        legal_moves = cpp_board.get_legal_moves_flat()
        pass_index = get_pass_index(bs)
        n_actions = pass_index + 1

        # Create mask for legal moves (including pass)
        mask = np.full(n_actions, float("-inf"))
        for idx in legal_moves:
            mask[idx] = 0.0
        mask[pass_index] = 0.0  # Pass is always legal

        masked_logits = policy_logits[:n_actions] + mask

        # Apply temperature
        scaled_logits = masked_logits / self.policy_temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / exp_logits.sum()

        # Build policy dict including pass action
        policy_dict: dict[int, float] = {
            idx: float(probs[idx]) for idx in legal_moves
        }
        policy_dict[alpha_go_cpp.PASS_ACTION] = float(probs[pass_index])

        return (policy_dict, value)

    @property
    def checkpoint_path(self) -> str | None:
        """Return checkpoint path if set."""
        return str(self._checkpoint_path) if self._checkpoint_path else None

    def close(self) -> None:
        """No resources to clean up - engine is managed externally."""
        pass


# Import for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from alpha_go.inference import LocalBatchedInferenceEngine


class RPCEvaluator:
    """Evaluator using a remote gRPC inference server."""

    def __init__(
        self,
        inference_address: str,
        board_size: int = 9,
        rpc_timeout: float = 30.0,
        checkpoint_id: str = "",
    ) -> None:
        """Initialize with a gRPC server address.

        Args:
            inference_address: gRPC server address (e.g., "localhost:50051")
            board_size: Size of the Go board
            rpc_timeout: Timeout for gRPC calls in seconds
            checkpoint_id: ID for multi-model server routing (empty for single-model server)
        """
        self.board_size = board_size
        self.rpc_timeout = rpc_timeout
        self.checkpoint_id = checkpoint_id
        self._channel: grpc.Channel | None = grpc.insecure_channel(inference_address)
        self._stub: inference_pb2_grpc.NNInferenceStub | None = (
            inference_pb2_grpc.NNInferenceStub(self._channel)
        )

    def evaluate(self, cpp_board: Any) -> tuple[dict[int, float], float]:
        """Evaluate C++ board using remote inference server."""
        board_np = cpp_board.to_numpy().astype(np.float32)

        # Normalize: current player sees themselves as 1
        if cpp_board.to_play() == alpha_go_cpp.GoBoard.WHITE:
            board_np = np.where(board_np == 1, 2.0, np.where(board_np == 2, 1.0, board_np))

        # Build gRPC request
        request = inference_pb2.EvaluateRequest(
            board=board_np.flatten().astype(int).tolist(),
            board_size=self.board_size,
            checkpoint_id=self.checkpoint_id,
        )

        # RPC call
        response = self._stub.Evaluate(request, timeout=self.rpc_timeout)
        value = response.value_logit

        # Get legal moves for masking
        legal_moves = cpp_board.get_legal_moves_flat()
        pass_index = get_pass_index(self.board_size)

        # Server returns probabilities for all n_actions = board_size^2 + 1
        policy_probs = np.array(response.policy_logits)

        # Collect probabilities for legal moves and pass, then renormalize
        legal_and_pass = list(legal_moves) + [pass_index]
        selected_probs = np.array([policy_probs[idx] for idx in legal_and_pass])
        selected_probs = selected_probs / selected_probs.sum()  # Renormalize

        # Build policy dict
        policy_dict: dict[int, float] = {}
        for i, idx in enumerate(legal_moves):
            policy_dict[idx] = float(selected_probs[i])
        # Add pass action with its probability (PASS_ACTION = -1 in C++)
        policy_dict[alpha_go_cpp.PASS_ACTION] = float(selected_probs[-1])

        return (policy_dict, value)

    def close(self) -> None:
        """Close gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __del__(self) -> None:
        self.close()


# ============================================================================
# Neural Network Evaluator (for Python MCTS)
# ============================================================================


class NNEvaluator(Evaluator[tuple[int, int] | None]):
    """Evaluator that uses a neural network for policy and value."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        board_size: int = 9,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model: Neural network model with policy and value heads
            device: Device to run inference on
            board_size: Size of the Go board
        """
        self.model = model
        self.device = device
        self.board_size = board_size
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self, state: State[tuple[int, int] | None]
    ) -> tuple[dict[tuple[int, int] | None, float], float]:
        """Evaluate state using neural network.

        Args:
            state: Current game state (must be GoState)

        Returns:
            priors: Dictionary mapping actions to prior probabilities
            value: Value estimate for current player (0 to 1)
        """
        assert isinstance(state, GoState), "NNEvaluator requires GoState"
        go_state: GoState = state

        # Convert board to tensor (use numpy array directly for speed)
        board_np = go_state.board_array
        board_BHW = torch.from_numpy(board_np).float().unsqueeze(0).to(self.device)

        # Normalize: current player should see themselves as 1
        if go_state.to_play == WHITE:
            board_BHW = torch.where(
                board_BHW == 1,
                torch.tensor(2.0, device=self.device),
                torch.where(
                    board_BHW == 2,
                    torch.tensor(1.0, device=self.device),
                    board_BHW,
                ),
            )

        # Forward pass - policy_C has n_actions = board_size^2 + 1 (including pass)
        policy_BC, value_B = self.model(board_BHW)
        policy_C = policy_BC.squeeze(0)  # (n_actions,)
        value = torch.sigmoid(value_B.squeeze(0)).item()

        # Get legal actions
        legal_actions = go_state.get_legal_actions()
        pass_index = get_pass_index(self.board_size)
        n_actions = pass_index + 1

        # Create mask for legal moves (including pass)
        legal_mask = torch.zeros(n_actions, device=self.device)
        for action in legal_actions:
            if action is not None:
                row, col = action
                idx = row * self.board_size + col
                legal_mask[idx] = 1.0
        # Pass is always legal
        legal_mask[pass_index] = 1.0

        # Apply mask and softmax
        masked_logits = policy_C.masked_fill(legal_mask == 0, float("-inf"))
        probs = F.softmax(masked_logits, dim=0).cpu().numpy()

        # Build prior dictionary including pass (None)
        priors: dict[tuple[int, int] | None, float] = {}
        for action in legal_actions:
            if action is not None:
                row, col = action
                idx = row * self.board_size + col
                priors[action] = float(probs[idx])
        # Add pass action (represented as None in Python MCTS)
        priors[None] = float(probs[pass_index])

        return priors, value


# ============================================================================
# MCTS Agent
# ============================================================================


class NNMCTSAgent(Agent):
    """Agent that uses MCTS with a neural network for policy and value.

    Combines neural network evaluation with Monte Carlo Tree Search
    for stronger play than raw policy sampling.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        board_size: int = 9,
        channels: int = 192,
        n_blocks: int = 12,
        num_simulations: int = 10,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        add_noise: bool = False,
        noise_alpha: float = 0.3,
        noise_weight: float = 0.25,
        lambda_: float = 0.5,
        rollout_temperature: float = 1.0,
        device: str | None = None,
    ) -> None:
        """Initialize the MCTS agent.

        Args:
            checkpoint_path: Path to the model checkpoint (.pt file)
            board_size: Size of the Go board (default 9)
            channels: Number of channels in the model (default 192)
            n_blocks: Number of residual blocks (default 12)
            num_simulations: Number of MCTS simulations per move (default 10)
            c_puct: Exploration constant for UCB (default 1.0)
            temperature: Temperature for action selection (default 1.0)
            add_noise: Whether to add Dirichlet noise at root (default False)
            noise_alpha: Dirichlet noise alpha parameter (default 0.3)
            noise_weight: Weight of noise vs prior (default 0.25)
            lambda_: Mixing parameter for value network vs rollout (0=pure value, 1=pure rollout)
            rollout_temperature: Temperature for sampling during fast rollouts (default 1.0)
            device: Device to run inference on (default: auto-detect)
        """
        self._checkpoint_path = Path(checkpoint_path)
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.lambda_ = lambda_
        self.rollout_temperature = rollout_temperature

        # Setup device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Create and load model (with backward compatibility for old checkpoints)
        config = MuPModelConfig(channels=channels, n_blocks=n_blocks, name="MCTS")
        self.model = create_mup_model(
            config=config, board_size=board_size, device=self.device
        )
        checkpoint = torch.load(
            self._checkpoint_path, map_location=self.device, weights_only=False
        )
        state_dict = upgrade_state_dict_for_pass(checkpoint["model_state_dict"], board_size)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Create evaluator and MCTS config
        self.evaluator = NNEvaluator(self.model, self.device, board_size)
        self.config = MCTSConfig(
            c_puct=c_puct,
            dirichlet_alpha=noise_alpha if add_noise else 0.0,
            dirichlet_weight=noise_weight,
            lambda_=lambda_,
        )

        # Create rollout policy function if lambda > 0
        self.rollout_policy_fn = self._create_rollout_policy() if lambda_ > 0 else None

    @property
    def checkpoint_path(self) -> str | None:
        """Return checkpoint path if agent uses one, None otherwise."""
        return str(self._checkpoint_path) if self._checkpoint_path else None

    def _create_rollout_policy(self) -> Callable[[Any], tuple[int, int] | None]:
        """Create a fast rollout policy that samples from the NN policy.

        Returns:
            Function that takes a GoState and returns an action sampled from NN policy.
        """

        def rollout_policy(state: Any) -> tuple[int, int] | None:
            """Sample action from neural network policy for fast rollout."""
            policy_dict, _ = self.evaluator.evaluate(state)

            # Apply temperature and sample
            actions = list(policy_dict.keys())
            probs = np.array([policy_dict[a] for a in actions])

            if self.rollout_temperature != 1.0:
                # Apply temperature
                log_probs = np.log(probs + 1e-8)
                log_probs = log_probs / self.rollout_temperature
                probs = np.exp(log_probs)

            # Normalize for floating-point precision
            probs = probs / probs.sum()

            # Sample action
            idx = np.random.choice(len(actions), p=probs)
            return actions[idx]

        return rollout_policy

    def select_move(self, board: alpha_go_cpp.GoBoard, seed: int) -> tuple[int, int]:
        """Select a move using MCTS.

        Args:
            board: C++ GoBoard with current game state
            seed: Random seed for reproducibility

        Returns:
            (row, col) tuple or PASS
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create state from C++ board
        state = GoState.from_cpp_board(board)

        # Run MCTS using functional API
        root = run_mcts(
            root_state=state,
            num_simulations=self.num_simulations,
            config=self.config,
            get_policy_and_value_fn=self.evaluator.evaluate,
            rollout_policy_fn=self.rollout_policy_fn,
        )

        # Select action from search result
        action = select_action_from_mcts(root, temperature=self.temperature)
        if action is None:
            return PASS
        return action

    def search(self, state: GoState) -> Node[tuple[int, int] | None]:
        """Run MCTS search and return root node.

        Args:
            state: Current game state

        Returns:
            Root node after search
        """
        return run_mcts(
            root_state=state,
            num_simulations=self.num_simulations,
            config=self.config,
            get_policy_and_value_fn=self.evaluator.evaluate,
            rollout_policy_fn=self.rollout_policy_fn,
        )


# ============================================================================
# C++ MCTS Search Result (compatible with Python Node interface)
# ============================================================================


@dataclass
class CppSearchResult:
    """Result from C++ MCTS search, compatible with Python Node for comparison.

    This wraps the C++ MCTSTree to provide a similar interface to Python's Node.
    """

    tree: Any  # alpha_go_cpp.MCTSTree
    board_size: int

    @property
    def N(self) -> int:
        """Root visit count."""
        return int(self.tree.get_root_visit_count())

    @property
    def Q(self) -> float:
        """Root Q-value."""
        return float(self.tree.get_root_q_value())

    def get_child_visits(self) -> dict[tuple[int, int] | None, int]:
        """Get visit counts for children, converting flat indices to (row, col)."""
        flat_visits = self.tree.get_child_visit_counts()
        result: dict[tuple[int, int] | None, int] = {}
        for flat_idx, visits in flat_visits.items():
            if flat_idx == alpha_go_cpp.PASS_ACTION:
                result[None] = visits
            else:
                row = flat_idx // self.board_size
                col = flat_idx % self.board_size
                result[(row, col)] = visits
        return result

    def get_child_q_values(self) -> dict[tuple[int, int] | None, float]:
        """Get Q-values for children, converting flat indices to (row, col)."""
        flat_q = self.tree.get_child_q_values()
        result: dict[tuple[int, int] | None, float] = {}
        for flat_idx, q in flat_q.items():
            if flat_idx == alpha_go_cpp.PASS_ACTION:
                result[None] = q
            else:
                row = flat_idx // self.board_size
                col = flat_idx % self.board_size
                result[(row, col)] = q
        return result

    def get_action_probabilities(self, temperature: float = 1.0) -> dict[tuple[int, int] | None, float]:
        """Get action probabilities, converting flat indices to (row, col)."""
        flat_probs = self.tree.get_action_probabilities(temperature)
        result: dict[tuple[int, int] | None, float] = {}
        for flat_idx, prob in flat_probs.items():
            if flat_idx == alpha_go_cpp.PASS_ACTION:
                result[None] = prob
            else:
                row = flat_idx // self.board_size
                col = flat_idx % self.board_size
                result[(row, col)] = prob
        return result


# ============================================================================
# C++ MCTS Agent
# ============================================================================


class CppMCTSAgent(Agent):
    """Agent that uses C++ MCTS with a pluggable evaluator.

    Combines C++ MCTS implementation with any evaluator that implements
    the CppMCTSEvaluator protocol (LocalNNEvaluator, RPCEvaluator, etc.).
    """

    def __init__(
        self,
        evaluator: CppMCTSEvaluator,
        num_simulations: int = 32,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        add_noise: bool = False,
        noise_alpha: float = 0.3,
        noise_weight: float = 0.25,
        lambda_: float = 0.0,
        rollout_temperature: float = 1.0,
        max_depth: int = 150,
        post_move_callback_fn: Callable[[CppSearchResult], None] | None = None,
        resign_threshold: float = 0.05,
        resign_consec_turns: int = 5,
        min_turns_before_resign: int = 0,
        pcr_sims: list[int] | None = None,
        pcr_probs: list[float] | None = None,
        leaf_batch_size: int = 0,
    ) -> None:
        """Initialize the C++ MCTS agent.

        Args:
            evaluator: Evaluator for policy/value estimation (LocalNNEvaluator, RPCEvaluator, etc.)
            num_simulations: Number of MCTS simulations per move (default 10)
            c_puct: Exploration constant for UCB (default 1.0)
            temperature: Temperature for action selection (default 1.0)
            add_noise: Whether to add Dirichlet noise at root (default False)
            noise_alpha: Dirichlet noise alpha parameter (default 0.3)
            noise_weight: Weight of noise vs prior (default 0.25)
            lambda_: Mixing parameter for value network vs rollout (0=pure value, 1=pure rollout)
            rollout_temperature: Temperature for sampling during fast rollouts (default 1.0)
            max_depth: Maximum total depth from game start (tree + rollout combined, default 100)
            post_move_callback_fn: Optional callback invoked after select_move with search result.
            resign_threshold: Resign when root value (win prob) is below this. 0 disables.
            resign_consec_turns: Number of consecutive turns below threshold before resigning.
            min_turns_before_resign: Gate on the agent's own move count — resign
                is only returned once this agent has played at least this many
                moves. Useful with random-init / early-iter ckpts whose value
                estimates are untrustworthy at game start (default 0 = always
                eligible to resign once resign_consec_turns is satisfied).
        """
        self.evaluator = evaluator
        self.board_size = evaluator.board_size
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.post_move_callback_fn = post_move_callback_fn
        self.resign_threshold = resign_threshold
        self.resign_consec_turns = resign_consec_turns
        self.min_turns_before_resign = min_turns_before_resign
        self._consec_below: int = 0
        self._turns_played: int = 0
        self.last_search_result: CppSearchResult | None = None

        # Create C++ MCTS config
        self.cpp_config = alpha_go_cpp.MCTSConfig()
        self.cpp_config.c_puct = c_puct
        self.cpp_config.dirichlet_alpha = noise_alpha if add_noise else 0.0
        self.cpp_config.dirichlet_weight = noise_weight
        self.cpp_config.temperature = temperature
        self.cpp_config.lambda_ = lambda_
        self.cpp_config.rollout_temperature = rollout_temperature
        self.cpp_config.max_depth = max_depth
        if pcr_sims is not None and pcr_probs is not None:
            assert len(pcr_sims) == len(pcr_probs) and abs(sum(pcr_probs) - 1.0) < 1e-4
            self.cpp_config.pcr_sims = list(pcr_sims)
            self.cpp_config.pcr_probs = list(pcr_probs)
        self.leaf_batch_size = leaf_batch_size

        # Also keep Python config for compatibility
        self.config = MCTSConfig(
            c_puct=c_puct,
            dirichlet_alpha=noise_alpha if add_noise else 0.0,
            dirichlet_weight=noise_weight,
            lambda_=lambda_,
            max_rollout_depth=max_depth,
        )

    @property
    def checkpoint_path(self) -> str | None:
        """Return checkpoint path if evaluator has one, None otherwise."""
        return getattr(self.evaluator, "checkpoint_path", None)

    def close(self) -> None:
        """Close evaluator resources."""
        self.evaluator.close()

    def __del__(self) -> None:
        if hasattr(self, "evaluator"):
            self.evaluator.close()

    def search(self, state: GoState) -> CppSearchResult:
        """Run C++ MCTS search and return result.

        Args:
            state: Current game state

        Returns:
            CppSearchResult wrapping the C++ MCTSTree
        """
        # Convert GoState to C++ board by copying the board array
        cpp_board = alpha_go_cpp.GoBoard(self.board_size)
        cpp_board.set_from_numpy(
            state.board_array.astype(np.int8),
            alpha_go_cpp.GoBoard.BLACK if state.to_play == BLACK else alpha_go_cpp.GoBoard.WHITE,
        )

        # Run MCTS
        tree = alpha_go_cpp.MCTSTree(cpp_board, self.cpp_config)
        tree.run_simulations(self.num_simulations, self.evaluator.evaluate)

        return CppSearchResult(tree=tree, board_size=self.board_size)

    # Alias for backward compatibility
    search_from_pystate = search

    def search_from_cpp_board(self, cpp_board: Any) -> CppSearchResult:
        """Run C++ MCTS search from an existing C++ board.

        Args:
            cpp_board: C++ GoBoard instance

        Returns:
            CppSearchResult wrapping the C++ MCTSTree
        """
        tree = alpha_go_cpp.MCTSTree(cpp_board, self.cpp_config)
        if self.leaf_batch_size > 0 and hasattr(self.evaluator, "batch_evaluate"):
            tree.run_simulations_batched(
                self.num_simulations,
                self.leaf_batch_size,
                self.evaluator.batch_evaluate,
            )
        else:
            tree.run_simulations(self.num_simulations, self.evaluator.evaluate)
        return CppSearchResult(tree=tree, board_size=self.board_size)

    def select_move(self, board: alpha_go_cpp.GoBoard, seed: int) -> tuple[int, int]:
        """Select a move using C++ MCTS.

        Args:
            board: C++ GoBoard with current game state
            seed: Random seed for reproducibility

        Returns:
            (row, col) tuple, PASS, or RESIGN
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        result = self.search_from_cpp_board(board)
        self.last_search_result = result
        self._turns_played += 1

        # Check resignation. result.Q is stored from player_at_parent
        # (opponent) perspective, so flip to our win-prob.
        if self.resign_threshold > 0:
            root_q = 1.0 - result.Q
            if root_q < self.resign_threshold:
                self._consec_below += 1
            else:
                self._consec_below = 0
            if (self._consec_below >= self.resign_consec_turns
                    and self._turns_played >= self.min_turns_before_resign):
                return RESIGN

        # Select action
        flat_action = result.tree.select_action(self.temperature)

        if flat_action == alpha_go_cpp.PASS_ACTION:
            move = PASS
        else:
            row, col = board.row_col(flat_action)
            move = (row, col)

        # Invoke callback if provided
        if self.post_move_callback_fn is not None:
            self.post_move_callback_fn(result)

        return move

# ============================================================================
# Registered C++ MCTS Agents
# ============================================================================

@register_agent("cpp-mcts-18m-v3-eval")
class CppMCTS18Mv3EvalAgent(CppMCTSAgent):
    def __init__(self) -> None:
        ckpt = "/nfs/checkpoints/2026-04-12_00-08-learngo-v3/iter6_best.pt"
        evaluator = LeafBatchedNNEvaluator(ckpt, 9, "18M")
        super().__init__(
            evaluator=evaluator,
            num_simulations=256,
            c_puct=0.5,
            temperature=0.1,
            resign_threshold=0.01,
            resign_consec_turns=8,
            leaf_batch_size=_lb_override(8),  # smaller helps sequential parity at low sims
        )


@register_agent("cpp-mcts-18m-v4")
class CppMCTS18Mv4Agent(CppMCTSAgent):
    def __init__(self) -> None:
        ckpt = "/nfs/checkpoints/2026-04-13_16-21-learngo-local-teacher-v4/iter4_best.pt"
        evaluator = LeafBatchedNNEvaluator(ckpt, 9, "18M")
        super().__init__(
            evaluator=evaluator,
            num_simulations=1024,
            c_puct=0.5,
            temperature=0.3,
            add_noise=False,
            resign_threshold=0.01,
            resign_consec_turns=8,
            leaf_batch_size=_lb_override(8),
        )

def _lb_override(default: int) -> int:
    return int(os.environ.get("LEAF_BATCH_SIZE_OVERRIDE", default))


@register_agent("cpp-mcts-18m-v3-lb")
class CppMCTS18Mv3LBAgent(CppMCTSAgent):
    """Leaf-parallel (virtual loss + batched eval) variant of cpp-mcts-18m-v3."""

    def __init__(self) -> None:
        ckpt = "/nfs/checkpoints/2026-04-12_00-08-learngo-v3/iter6_best.pt"
        evaluator = LeafBatchedNNEvaluator(
            checkpoint_path=ckpt, board_size=9, model_config="18M",
        )
        super().__init__(
            evaluator=evaluator,
            num_simulations=128,
            pcr_sims=[128, 256, 2000],
            pcr_probs=[0.80, 0.18, 0.02],
            leaf_batch_size=_lb_override(16),
        )


def create_batched_cpp_mcts_agent(
    engine: "LocalBatchedInferenceEngine",
    checkpoint_path: str | Path,
    board_size: int = 9,
    model_config: str = "18M",
    policy_temperature: float = 1.0,
    **mcts_kwargs: Any,
) -> CppMCTSAgent:
    """Create a CppMCTSAgent backed by a shared LocalBatchedInferenceEngine.

    All agents sharing the same engine will have their inference requests
    batched together for efficient GPU utilization.
    """
    evaluator = BatchedLocalNNEvaluator(
        engine=engine,
        board_size=board_size,
        policy_temperature=policy_temperature,
        checkpoint_path=checkpoint_path,
    )
    return CppMCTSAgent(evaluator=evaluator, **mcts_kwargs)


def _build_model_from_checkpoint(
    checkpoint_path: str | Path,
    board_size: int,
    model_config: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load a checkpoint and instantiate the matching model class.

    Auto-detects SizeInvariantGoResNet vs MuPGoResNet using both the
    serialized `config` string and a state-dict key fingerprint (same rule
    LeafBatchedNNEvaluator uses). Returns the model already loaded and
    moved onto `device`, in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config_name = str(checkpoint.get("config", ""))
    if config_name.startswith("SizeInvariantGoResNet") or "pass_fc.weight" in state_dict:
        # Recover the hyperparameters from the state_dict shape fingerprint:
        # channels = input_conv output channels; n_blocks = number of block
        # conv1 params; value_hidden = value_fc2 input dim; norm variant
        # determined by the presence of input_bn running stats; SE blocks
        # implied by blocks.0.se.fc1.
        channels = int(state_dict["input_conv.weight"].shape[0])
        n_blocks = sum(
            1 for k in state_dict if k.startswith("blocks.") and k.endswith(".conv1.weight")
        )
        value_hidden = int(state_dict["value_fc2.weight"].shape[1])
        norm_type = "bn" if "input_bn.running_mean" in state_dict else "gn"
        use_se = "blocks.0.se.fc1.weight" in state_dict
        se_reduction = 8
        if use_se:
            se_hidden = int(state_dict["blocks.0.se.fc1.weight"].shape[0])
            se_reduction = max(1, channels // max(1, se_hidden))
        model: torch.nn.Module = SizeInvariantGoResNet(
            channels=channels, n_blocks=n_blocks, value_hidden=value_hidden,
            norm_type=norm_type, use_se=use_se, se_reduction=se_reduction,
        ).to(device)
        model.load_state_dict(state_dict)
    else:
        model = create_mup_model(
            config=model_config, board_size=board_size, device=device,
        )
        state_dict = upgrade_state_dict_for_pass(state_dict, board_size)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def create_shared_inference_engine(
    checkpoint_path: str | Path,
    model_config: str = "18M",
    board_size: int = 9,
    batch_size: int = 64,
    batch_timeout_ms: float = 1.0,
    device: str | None = None,
) -> "LocalBatchedInferenceEngine":
    """Create a LocalBatchedInferenceEngine with a loaded model.

    Returns a started engine ready to accept inference requests. The model
    class is auto-detected from the checkpoint (MuPGoResNet vs
    SizeInvariantGoResNet) via `_build_model_from_checkpoint`.
    """
    from alpha_go.inference import LocalBatchedInferenceEngine

    device_t = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = _build_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        board_size=board_size,
        model_config=model_config,
        device=device_t,
    )

    engine = LocalBatchedInferenceEngine(
        model=model,
        device=device_t,
        batch_size=batch_size,
        batch_timeout_ms=batch_timeout_ms,
        board_size=board_size,
    )
    engine.start()
    return engine


@register_agent("cpp-mcts-rpc")
class CppMCTSRPCAgent(CppMCTSAgent):
    """C++ MCTS agent using gRPC inference server. Independent of model params.

    Start inference server first with:
    uv run -m alpha_go.inference_server --model-type mup --port 50051 \
        --checkpoint experiments/2026-01-05_15-12-replay-buffer-with-offline-pusher/checkpoints/step_50000.pt \
        --board-size 9 --batch-size 4 --model-config 10M
    """

    def __init__(self) -> None:
        evaluator = RPCEvaluator(
            inference_address="localhost:50051",
            board_size=9,
        )
        super().__init__(
            evaluator=evaluator,
            num_simulations=20,
            max_depth=120,
        )
