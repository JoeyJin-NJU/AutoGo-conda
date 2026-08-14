"""Tests for the Go transformer model."""
import pytest
import torch

from alpha_go.model import GoTransformer, MuPGoResNet, create_mup_model


class TestGoTransformer:
    """Tests for GoTransformer."""

    def test_forward_shape(self) -> None:
        """Test that forward pass produces correct output shapes."""
        model = GoTransformer(board_size=9, d_model=64, n_heads=4, n_layers=2)
        board = torch.zeros(2, 9, 9)  # Batch of 2 empty boards

        policy, value = model(board)

        # Policy now has board_size^2 + 1 actions (including pass)
        assert policy.shape == (2, 82), f"Expected (2, 82), got {policy.shape}"
        assert value.shape == (2,), f"Expected (2,), got {value.shape}"

    def test_forward_with_stones(self) -> None:
        """Test forward pass with actual stones on board."""
        model = GoTransformer(board_size=9, d_model=64, n_heads=4, n_layers=2)
        board = torch.zeros(1, 9, 9)
        board[0, 4, 4] = 1  # Current player stone at center
        board[0, 4, 5] = 2  # Opponent stone

        policy, value = model(board)

        assert policy.shape == (1, 82)
        assert value.shape == (1,)
        assert torch.isfinite(policy).all()
        assert torch.isfinite(value).all()

    def test_compute_loss(self) -> None:
        """Test loss computation."""
        model = GoTransformer(board_size=9, d_model=64, n_heads=4, n_layers=2)
        board = torch.zeros(4, 9, 9)
        move = torch.tensor([[4, 4], [3, 3], [-1, -1], [0, 0]])  # One pass
        winner = torch.tensor([1, 0, 1, 0])

        total_loss, policy_loss, value_loss = model.compute_loss(board, move, winner)

        assert torch.isfinite(total_loss)
        assert torch.isfinite(policy_loss)
        assert torch.isfinite(value_loss)
        assert total_loss.item() > 0

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        model = GoTransformer(board_size=9, d_model=64, n_heads=4, n_layers=2)
        board = torch.zeros(2, 9, 9)
        move = torch.tensor([[4, 4], [3, 3]])
        winner = torch.tensor([1, 0])

        total_loss, _, _ = model.compute_loss(board, move, winner)
        total_loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_different_board_sizes(self) -> None:
        """Test model with different board sizes."""
        for size in [9, 13, 19]:
            model = GoTransformer(board_size=size, d_model=64, n_heads=4, n_layers=2)
            board = torch.zeros(1, size, size)

            policy, value = model(board)

            # Policy has n_actions = size*size + 1 (including pass)
            assert policy.shape == (1, size * size + 1)
            assert value.shape == (1,)

class TestDatasetIntegration:
    """Integration tests with actual dataset."""

    @pytest.fixture
    def dataset(self):
        """Load dataset if available."""
        from pathlib import Path
        from alpha_go.dataset import GoDataset

        # Try common data locations
        for data_path in [
            Path("game_data/9x9/dev-train"),
            Path("game_data/9x9/dev-val"),
            Path("game_data/9x9"),
        ]:
            if data_path.exists() and list(data_path.glob("*.npz")):
                try:
                    ds = GoDataset(data_path)
                    if len(ds) > 0:
                        return ds
                except FileNotFoundError:
                    continue

        pytest.skip("No game data available for testing")

    def test_forward_with_dataset_sample(self, dataset) -> None:
        """Test forward pass with a real dataset sample."""
        model = GoTransformer(board_size=9, d_model=64, n_heads=4, n_layers=2)

        sample = dataset[0]
        board = sample["board"].unsqueeze(0)  # Add batch dim

        policy, value = model(board)

        # Policy has n_actions = 82 (81 + 1 for pass)
        assert policy.shape == (1, 82)
        assert value.shape == (1,)
        assert torch.isfinite(policy).all()
        assert torch.isfinite(value).all()

    def test_loss_with_dataset_sample(self, dataset) -> None:
        """Test loss computation with a real dataset sample."""
        model = GoTransformer(board_size=9, d_model=64, n_heads=4, n_layers=2)

        sample = dataset[0]
        board = sample["board"].unsqueeze(0)
        move = sample["move"].unsqueeze(0)
        winner = torch.tensor([sample["winner"]])

        total_loss, policy_loss, value_loss = model.compute_loss(board, move, winner)

        assert torch.isfinite(total_loss)
        assert total_loss.item() > 0

    def test_batch_from_dataset(self, dataset) -> None:
        """Test with a batch of samples from dataset."""
        from torch.utils.data import DataLoader

        model = GoTransformer(board_size=9, d_model=64, n_heads=4, n_layers=2)

        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        board = batch["board"]
        move = batch["move"]
        winner = batch["winner"]

        total_loss, policy_loss, value_loss = model.compute_loss(board, move, winner)

        assert torch.isfinite(total_loss)
        assert total_loss.item() > 0


class TestMuPGoResNet:
    """Tests for MuPGoResNet with muP parameterization.

    Note: MuPGoResNet requires set_base_shapes() to be called before use,
    which is handled by the create_mup_model() factory function.
    """

    @pytest.fixture
    def model(self):
        """Create a MuPGoResNet model via factory."""
        return create_mup_model(config="3M", board_size=9)

    def test_forward_shape(self, model) -> None:
        """Test that forward pass produces correct output shapes."""
        board = torch.zeros(2, 9, 9)

        policy, value = model(board)

        # Policy now has board_size^2 + 1 actions (including pass)
        assert policy.shape == (2, 82), f"Expected (2, 82), got {policy.shape}"
        assert value.shape == (2,), f"Expected (2,), got {value.shape}"

    def test_forward_with_stones(self, model) -> None:
        """Test forward pass with actual stones on board."""
        board = torch.zeros(1, 9, 9)
        board[0, 4, 4] = 1
        board[0, 4, 5] = 2

        policy, value = model(board)

        assert policy.shape == (1, 82)
        assert value.shape == (1,)
        assert torch.isfinite(policy).all()
        assert torch.isfinite(value).all()

    def test_compute_loss(self, model) -> None:
        """Test loss computation."""
        board = torch.zeros(4, 9, 9)
        move = torch.tensor([[4, 4], [3, 3], [-1, -1], [0, 0]])
        winner = torch.tensor([1, 0, 1, 0])

        total_loss, policy_loss, value_loss = model.compute_loss(board, move, winner)

        assert torch.isfinite(total_loss)
        assert torch.isfinite(policy_loss)
        assert torch.isfinite(value_loss)
        assert total_loss.item() > 0

    def test_gradient_flow(self, model) -> None:
        """Test that gradients flow through the model."""
        board = torch.zeros(2, 9, 9)
        move = torch.tensor([[4, 4], [3, 3]])
        winner = torch.tensor([1, 0])

        total_loss, _, _ = model.compute_loss(board, move, winner)
        total_loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_create_mup_model_configs(self) -> None:
        """Test create_mup_model factory function with different configs."""
        for config in ["3M", "18M"]:
            model = create_mup_model(config=config, board_size=9)
            board = torch.zeros(1, 9, 9)

            policy, value = model(board)

            # Policy has n_actions = 82 (81 + 1 for pass)
            assert policy.shape == (1, 82)
            assert value.shape == (1,)
            assert torch.isfinite(policy).all()
            assert torch.isfinite(value).all()

    def test_depth_scaling(self, model) -> None:
        """Test that d-muP depth scaling is applied (1/sqrt(n_blocks))."""
        import math
        # Check the depth scale on the residual blocks
        expected_scale = 1.0 / math.sqrt(model.n_blocks)
        first_block = model.blocks[0]
        assert abs(first_block.depth_scale - expected_scale) < 1e-6
