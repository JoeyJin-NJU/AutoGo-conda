"""Neural network agent that samples moves from a policy network."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import alpha_go_cpp
import grpc
import numpy as np
import torch
import torch.nn.functional as F

from alpha_go.agents.base import Agent, PASS, get_pass_index, register_agent
from alpha_go.model import GoTransformer, create_mup_model, MODEL_CONFIGS, upgrade_state_dict_for_pass
from alpha_go.proto import inference_pb2, inference_pb2_grpc

if TYPE_CHECKING:
    from alpha_go.inference import LocalBatchedInferenceEngine

# Type alias for supported model classes
ModelType = Literal["GoTransformer", "MuPGoResNet"]


class NNAgent(Agent):
    """Agent that samples moves from a neural network policy head.

    Supports three modes:
    - Local with checkpoint: Load model from checkpoint and run inference in-process
    - Remote: Use gRPC to call an inference server
    - Random: Initialize model with random weights (no training)

    Provide exactly one of: checkpoint_path, inference_address, or model_config.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        inference_address: str | None = None,
        model_config: str | None = None,
        model_type: ModelType = "MuPGoResNet",
        board_size: int = 9,
        temperature: float = 1.0,
        device: str | None = None,
        rpc_timeout: float = 30.0,
        checkpoint_id: str = "",
        **model_kwargs,
    ) -> None:
        """Initialize the neural network agent.

        Args:
            checkpoint_path: Path to model checkpoint for local inference.
            inference_address: gRPC server address (e.g., "localhost:50051") for remote inference.
            model_config: Model config name (e.g., "10M", "100M") for random initialization.
            model_type: Model architecture (only used for local inference).
            board_size: Size of the Go board (default 9).
            temperature: Sampling temperature (default 1.0). Higher = more random.
            device: Device for local inference. Defaults to CUDA if available.
            rpc_timeout: Timeout for gRPC calls in seconds (default 30).
            checkpoint_id: Checkpoint ID for multi-model inference server routing.
                          Only used with inference_address.
            **model_kwargs: Additional model arguments for local inference
                (e.g., channels, n_blocks for GoResNet/MuPGoResNet).
        """
        # Validate: exactly one inference mode
        modes = sum([checkpoint_path is not None, inference_address is not None, model_config is not None])
        if modes != 1:
            raise ValueError("Must provide exactly one of checkpoint_path, inference_address, or model_config")

        self.board_size = board_size
        self.temperature = temperature
        self.rpc_timeout = rpc_timeout
        self._last_entropy: float | None = None

        # Remote inference mode
        if inference_address is not None:
            self._channel: grpc.Channel | None = grpc.insecure_channel(inference_address)
            self._stub: inference_pb2_grpc.NNInferenceStub | None = inference_pb2_grpc.NNInferenceStub(self._channel)
            self._checkpoint_id = checkpoint_id
            self.model = None
            self.device = None
            self._checkpoint_path: Path | None = None
            self.model_type = None
        else:
            # Local inference mode (with checkpoint or random)
            self._channel = None
            self._stub = None
            self._checkpoint_id = ""  # Not used for local inference
            self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
            self.model_type = model_type
            self.device = torch.device(
                device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            )

            # Create model
            if model_type == "MuPGoResNet":
                if model_config is not None:
                    # Use predefined config for random initialization
                    self.model = create_mup_model(config=model_config, board_size=board_size, device=self.device)
                else:
                    # Use model_kwargs for checkpoint loading
                    from alpha_go.model import MuPModelConfig
                    channels = model_kwargs.get("channels", 192)
                    n_blocks = model_kwargs.get("n_blocks", 12)
                    config = MuPModelConfig(channels=channels, n_blocks=n_blocks, name="Agent")
                    self.model = create_mup_model(config=config, board_size=board_size, device=self.device)
            elif model_type == "GoResNet":
                self.model = GoResNet(board_size=board_size, **model_kwargs)
                self.model.to(self.device)
            else:
                self.model = GoTransformer(board_size=board_size, **model_kwargs)
                self.model.to(self.device)

            # Load checkpoint if provided (with backward compatibility for old checkpoints)
            if checkpoint_path is not None:
                checkpoint = torch.load(self._checkpoint_path, map_location=self.device, weights_only=False)
                state_dict = upgrade_state_dict_for_pass(checkpoint["model_state_dict"], board_size)
                self.model.load_state_dict(state_dict)

            self.model.eval()

    def close(self) -> None:
        """Close gRPC channel if using remote inference."""
        if getattr(self, "_channel", None) is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __del__(self) -> None:
        self.close()

    @property
    def checkpoint_path(self) -> str | None:
        """Return checkpoint path if agent uses one, None otherwise."""
        return str(self._checkpoint_path) if self._checkpoint_path else None

    def get_last_entropy(self) -> float | None:
        """Get policy entropy from the last select_move call.

        Returns entropy of the policy distribution before legal move masking.
        Returns None if no move has been selected yet.
        """
        return self._last_entropy

    def select_move(self, board: alpha_go_cpp.GoBoard, seed: int) -> tuple[int, int]:
        """Select a move by sampling from the policy network.

        Args:
            board: C++ GoBoard with current game state.
            seed: Random seed for reproducibility.

        Returns:
            (row, col) tuple or PASS.
        """
        if self._stub is not None:
            return self._select_move_remote(board, seed)
        else:
            return self._select_move_local(board, seed)

    def _select_move_remote(self, board: alpha_go_cpp.GoBoard, seed: int) -> tuple[int, int]:
        """Select move using gRPC inference server."""
        board_np = board.to_numpy().astype(np.float32)
        if board.to_play() == alpha_go_cpp.GoBoard.WHITE:
            board_np = np.where(board_np == 1, 2.0, np.where(board_np == 2, 1.0, board_np))

        request = inference_pb2.EvaluateRequest(
            board=board_np.flatten().astype(int).tolist(),
            board_size=self.board_size,
            checkpoint_id=self._checkpoint_id,
        )
        response = self._stub.Evaluate(request, timeout=self.rpc_timeout)

        self._last_entropy = response.policy_entropy
        policy = np.array(response.policy_logits)
        pass_index = get_pass_index(self.board_size)
        n_actions = pass_index + 1

        # Create legal mask of size n_actions (board_size^2 + 1)
        legal_mask = np.zeros(n_actions)

        # Mark legal board moves
        legal_flat = board.get_legal_moves_flat()
        for idx in legal_flat:
            legal_mask[idx] = 1.0

        # Pass is always legal (game termination is handled at game loop level)
        legal_mask[pass_index] = 1.0

        masked_logits = np.where(legal_mask == 1, policy, float("-inf"))

        # Apply temperature and sample
        if self.temperature == 0:
            idx = int(np.argmax(masked_logits))
        else:
            max_logit = np.max(masked_logits[masked_logits > float("-inf")])
            exp_logits = np.exp((masked_logits - max_logit) / self.temperature)
            probs = exp_logits / exp_logits.sum()
            np.random.seed(seed)
            idx = int(np.random.choice(len(probs), p=probs))

        # Check if selected action is pass
        if idx == pass_index:
            return PASS

        row, col = board.row_col(idx)
        return (row, col)

    @torch.no_grad()
    def _select_move_local(self, board: alpha_go_cpp.GoBoard, seed: int) -> tuple[int, int]:
        """Select move using local model inference."""
        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Get board state and convert to tensor
        board_np = board.to_numpy().astype(np.float32)
        board_BHW = torch.from_numpy(board_np).unsqueeze(0).to(self.device)

        # Normalize: 0=empty, current player=1, opponent=2
        if board.to_play() == alpha_go_cpp.GoBoard.WHITE:
            board_BHW = torch.where(
                board_BHW == 1, torch.tensor(2.0, device=self.device),
                torch.where(board_BHW == 2, torch.tensor(1.0, device=self.device), board_BHW)
            )

        # Forward pass - policy_C has n_actions = board_size^2 + 1 (including pass)
        policy_BC, value_B = self.model(board_BHW)
        policy_C = policy_BC.squeeze(0)

        # Compute entropy before legal move masking
        probs_C = F.softmax(policy_C, dim=0)
        entropy = -torch.sum(probs_C * torch.log(probs_C + 1e-8)).item()
        self._last_entropy = entropy

        # Create legal mask of size n_actions (board_size^2 + 1)
        pass_index = get_pass_index(self.board_size)
        n_actions = pass_index + 1
        legal_mask = torch.zeros(n_actions, device=self.device)

        # Mark legal board moves
        legal_flat = board.get_legal_moves_flat()
        for flat_idx in legal_flat:
            legal_mask[flat_idx] = 1.0

        # Pass is always legal (game termination is handled at game loop level)
        legal_mask[pass_index] = 1.0

        masked_logits = policy_C.masked_fill(legal_mask == 0, float("-inf"))

        # Apply temperature and sample
        if self.temperature == 0:
            idx = masked_logits.argmax().item()
        else:
            probs = F.softmax(masked_logits / self.temperature, dim=0)
            idx = torch.multinomial(probs, num_samples=1).item()

        # Check if selected action is pass
        if idx == pass_index:
            return PASS

        row, col = board.row_col(idx)
        return (row, col)


class BatchedNNAgent(Agent):
    """Agent that uses a shared LocalBatchedInferenceEngine for high-throughput inference.

    This agent is designed for collectors running on machines with GPU access.
    Multiple BatchedNNAgent instances share a single inference engine, which
    batches their requests for efficient GPU utilization.

    Unlike NNAgent, this agent does not manage its own model - it uses an
    externally provided inference engine that should be started/stopped by
    the caller (e.g., CollectorWorker).

    Args:
        engine: Shared LocalBatchedInferenceEngine instance
        board_size: Size of the Go board (default 9)
        temperature: Sampling temperature (default 1.0). Higher = more random.

    Profiling metrics recorded:
    - batched_agent_select_move_ns: Total select_move time
    - batched_agent_future_wait_ns: Time waiting for inference result
    """

    def __init__(
        self,
        engine: "LocalBatchedInferenceEngine",
        board_size: int = 9,
        temperature: float = 1.0,
    ) -> None:
        self.engine = engine
        self.board_size = board_size
        self.temperature = temperature
        self._last_entropy: float | None = None

    def get_last_entropy(self) -> float | None:
        """Get policy entropy from the last select_move call."""
        return self._last_entropy

    def select_move(self, board: alpha_go_cpp.GoBoard, seed: int) -> tuple[int, int]:
        """Select a move by sampling from the policy network via batched inference.

        Args:
            board: C++ GoBoard with current game state.
            seed: Random seed for reproducibility.

        Returns:
            (row, col) tuple or PASS.
        """
        # Get board state and normalize to current_player=1
        board_np = board.to_numpy().astype(np.float32)
        if board.to_play() == alpha_go_cpp.GoBoard.WHITE:
            board_np = np.where(board_np == 1, 2.0, np.where(board_np == 2, 1.0, board_np))

        # Submit to batched inference engine and wait for the result.
        future = self.engine.submit(board_np)
        policy, value, entropy = future.result()

        # Store entropy
        self._last_entropy = entropy

        # Get legal moves
        pass_index = get_pass_index(self.board_size)
        n_actions = pass_index + 1
        legal_mask = np.zeros(n_actions)

        legal_flat = board.get_legal_moves_flat()
        for idx in legal_flat:
            legal_mask[idx] = 1.0
        legal_mask[pass_index] = 1.0  # Pass is always legal

        masked_logits = np.where(legal_mask == 1, policy, float("-inf"))

        # Apply temperature and sample
        if self.temperature == 0:
            idx = int(np.argmax(masked_logits))
        else:
            max_logit = np.max(masked_logits[masked_logits > float("-inf")])
            exp_logits = np.exp((masked_logits - max_logit) / self.temperature)
            probs = exp_logits / exp_logits.sum()
            np.random.seed(seed)
            idx = int(np.random.choice(len(probs), p=probs))

        # Check if selected action is pass
        if idx == pass_index:
            return PASS

        row, col = board.row_col(idx)
        return (row, col)
