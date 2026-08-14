"""Models for Go policy and value prediction.

Dimension key:
    B: batch size
    L: sequence length (H*W + 1 for CLS token)
    H: board height
    W: board width
    D: model dimension
    F: feed-forward hidden size
    A: number of attention heads
    K: attention head dimension (D // A)
    C: number of classes (H*W for policy)
    Ch: number of channels (for CNN)

Model sizes (MuPGoResNet):
    | Model         | Channels | Blocks | Actual Params |
    |---------------|----------|--------|---------------|
    | GoResNet-10M  | 256      | 10     | 11.85M        |
    | GoResNet-100M | 512      | 20     | 94.46M        |
    | GoResNet-500M | 1024     | 25     | 472.03M       |
    | GoResNet-1B   | 1024     | 53     | 1000.63M      |
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from mup import MuReadout, set_base_shapes


# ============================================================================
# Transformer Components
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention using F.scaled_dot_product_attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        B, L, D = x_BLD.shape

        # Project to Q, K, V and reshape to (B, A, L, K)
        q_BALK = self.w_q(x_BLD).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k_BALK = self.w_k(x_BLD).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v_BALK = self.w_v(x_BLD).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention (uses Flash Attention when available)
        dropout_p = self.dropout if self.training else 0.0
        out_BALK = F.scaled_dot_product_attention(q_BALK, k_BALK, v_BALK, dropout_p=dropout_p)

        out_BLD = out_BALK.transpose(1, 2).contiguous().view(B, L, D)
        return self.w_o(out_BLD)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x_BLD))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        x_BLD = x_BLD + self.dropout(self.attn(self.norm1(x_BLD)))
        x_BLD = x_BLD + self.dropout(self.ff(self.norm2(x_BLD)))
        return x_BLD


class GoTransformer(nn.Module):
    """Transformer model for Go with policy and value heads.

    Architecture:
        - Input: (B, H, W) board with values 0 (empty), 1 (current player), 2 (opponent)
        - Embedding: 3-class embedding for each cell + positional embedding
        - CLS token prepended for value prediction
        - 13 transformer layers
        - Policy head: nxn logits from position tokens
        - Value head: binary logit from CLS token
    """

    def __init__(
        self,
        board_size: int = 9,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 13,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.d_model = d_model
        self.n_positions = board_size * board_size
        self.n_actions = self.n_positions + 1  # Include pass action

        # Embeddings
        self.cell_embed = nn.Embedding(3, d_model)  # 0=empty, 1=self, 2=opponent
        self.pos_embed = nn.Embedding(self.n_positions + 1, d_model)  # +1 for CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Output heads
        self.policy_head = nn.Linear(d_model, 1)  # Per-position logit
        self.pass_head = nn.Linear(d_model, 1)    # Pass logit from CLS token
        self.value_head = nn.Linear(d_model, 1)   # Binary logit from CLS

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self, board_BHW: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            board_BHW: Board state (B, H, W) with values 0, 1, 2

        Returns:
            policy_BC: Policy logits (B, H*W+1) over board positions and pass
            value_B: Value logits (B,) for win probability
        """
        B, H, W = board_BHW.shape
        assert H == self.board_size and W == self.board_size

        # Flatten board and embed cells
        board_BL = board_BHW.view(B, -1).long()  # (B, H*W)
        x_BLD = self.cell_embed(board_BL)  # (B, H*W, D)

        # Add positional embeddings (positions 1 to H*W, 0 reserved for CLS)
        positions = torch.arange(1, self.n_positions + 1, device=board_BHW.device)
        x_BLD = x_BLD + self.pos_embed(positions)

        # Prepend CLS token with position 0
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        cls_tokens = cls_tokens + self.pos_embed(torch.zeros(1, dtype=torch.long, device=board_BHW.device))
        x_BLD = torch.cat([cls_tokens, x_BLD], dim=1)  # (B, 1 + H*W, D)

        x_BLD = self.dropout(x_BLD)

        # Transformer layers
        for layer in self.layers:
            x_BLD = layer(x_BLD)
        x_BLD = self.norm(x_BLD)

        # Split CLS and position tokens
        cls_BD = x_BLD[:, 0]  # (B, D)
        pos_BLD = x_BLD[:, 1:]  # (B, H*W, D)

        # Policy head: logit per position + pass logit from CLS
        pos_logits_BC = self.policy_head(pos_BLD).squeeze(-1)  # (B, H*W)
        pass_logit_B1 = self.pass_head(cls_BD)  # (B, 1)
        policy_BC = torch.cat([pos_logits_BC, pass_logit_B1], dim=1)  # (B, H*W+1)

        # Value head: single logit from CLS
        value_B = self.value_head(cls_BD).squeeze(-1)  # (B,)

        return policy_BC, value_B

    def compute_loss(
        self,
        board_BHW: torch.Tensor,
        target_move_B2: torch.Tensor,
        target_winner_B: torch.Tensor,
        is_expert_B: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute policy and value losses.

        Args:
            board_BHW: Board states
            target_move_B2: Target moves as (row, col), -1 for pass
            target_winner_B: 1 if current player wins, 0 otherwise
            is_expert_B: Boolean mask, policy loss only applied where True.
                         If None, applies to all samples.

        Returns:
            total_loss, policy_loss, value_loss
        """
        policy_BC, value_B = self.forward(board_BHW)

        # Policy loss: cross-entropy (including pass moves)
        # Convert pass (-1, -1) to flat index n_positions (board_size^2)
        is_pass = target_move_B2[:, 0] < 0
        target_idx_B = torch.where(
            is_pass,
            torch.full_like(target_move_B2[:, 0], self.n_positions),  # Pass index
            target_move_B2[:, 0] * self.board_size + target_move_B2[:, 1],
        )

        # Apply expert mask if provided
        if is_expert_B is not None:
            valid_mask = is_expert_B
        else:
            valid_mask = torch.ones(board_BHW.shape[0], dtype=torch.bool, device=board_BHW.device)

        if valid_mask.sum() > 0:
            policy_loss = F.cross_entropy(
                policy_BC[valid_mask],
                target_idx_B[valid_mask].long(),
            )
        else:
            policy_loss = torch.tensor(0.0, device=board_BHW.device)

        # Value loss: binary cross-entropy (always applied)
        value_loss = F.binary_cross_entropy_with_logits(
            value_B,
            target_winner_B.float(),
        )

        total_loss = policy_loss + value_loss
        return total_loss, policy_loss, value_loss


# ============================================================================
# MuP Model Configurations
# ============================================================================

@dataclass
class MuPModelConfig:
    """Configuration for a muP-parameterized GoResNet model."""
    channels: int
    n_blocks: int
    name: str = ""
    policy_channels: int = 2
    value_channels: int = 1

# For muP, base and delta must have SAME depth (n_blocks), only width differs
MUP_BASE_WIDTH = 32
MUP_DELTA_WIDTH = 64

GORESNET_3M = MuPModelConfig(channels=128, n_blocks=10, policy_channels=32, value_channels=32, name="GoResNet-3M")
GORESNET_18M = MuPModelConfig(channels=256, n_blocks=14, policy_channels=64, value_channels=64, name="GoResNet-18M")

MODEL_CONFIGS: dict[str, MuPModelConfig] = {
    "3M": GORESNET_3M,
    "18M": GORESNET_18M,
}


# ============================================================================
# MuP ResNet Components
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x_BChHW: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_BChHW.shape
        scale_BC = x_BChHW.mean(dim=(2, 3))  # global avg pool
        scale_BC = F.relu(self.fc1(scale_BC))
        scale_BC = torch.sigmoid(self.fc2(scale_BC))
        return x_BChHW * scale_BC.unsqueeze(-1).unsqueeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, depth_scale: float = 1.0, use_se: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else None
        self.depth_scale = depth_scale

    def forward(self, x_BChHW: torch.Tensor) -> torch.Tensor:
        residual = x_BChHW
        x_BChHW = F.relu(self.bn1(self.conv1(x_BChHW)))
        x_BChHW = self.bn2(self.conv2(x_BChHW))
        if self.se is not None:
            x_BChHW = self.se(x_BChHW)
        return F.relu(residual + self.depth_scale * x_BChHW)


class MuPGoResNet(nn.Module):
    """GoResNet with wider policy/value heads + MuP for scalability."""
    def __init__(self, board_size=9, channels=72, n_blocks=10,
                 policy_channels=32, value_channels=32, init_scale=1.0):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.n_blocks = n_blocks
        self.n_positions = board_size * board_size
        self.n_actions = self.n_positions + 1
        self.init_scale = init_scale
        depth_scale = 1.0 / math.sqrt(n_blocks)
        self.input_conv = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels, depth_scale) for _ in range(n_blocks)])
        # Wider policy head with MuReadout
        self.policy_conv = nn.Conv2d(channels, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = MuReadout(policy_channels * self.n_positions, self.n_actions)
        # Wider value head with MuReadout
        self.value_conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * self.n_positions, 256)
        self.value_fc2 = MuReadout(256, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                module.weight.data *= self.init_scale
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear) and not isinstance(module, MuReadout):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                module.weight.data *= self.init_scale
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, MuReadout):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, board_BHW: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, W = board_BHW.shape
        board_long = board_BHW.long()
        x_B3HW = torch.zeros(B, 3, H, W, device=board_BHW.device, dtype=torch.float32)
        x_B3HW.scatter_(1, board_long.unsqueeze(1), 1.0)
        x_BChHW = F.relu(self.input_bn(self.input_conv(x_B3HW)))
        for block in self.blocks:
            x_BChHW = block(x_BChHW)
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x_BChHW)))
        policy_BC = self.policy_fc(p.view(B, -1))
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x_BChHW)))
        v = F.relu(self.value_fc1(v.view(B, -1)))
        value_B = self.value_fc2(v).squeeze(-1)
        return policy_BC, value_B

    def compute_loss(self, board_BHW, target_move_B2, target_winner_B, is_expert_B=None):
        policy_BC, value_B = self.forward(board_BHW)
        is_pass = target_move_B2[:, 0] < 0
        target_idx_B = torch.where(is_pass, torch.full_like(target_move_B2[:, 0], self.n_positions),
                                    target_move_B2[:, 0] * self.board_size + target_move_B2[:, 1])
        if is_expert_B is not None and is_expert_B.sum() > 0:
            policy_loss = F.cross_entropy(policy_BC[is_expert_B], target_idx_B[is_expert_B].long())
        else:
            policy_loss = F.cross_entropy(policy_BC, target_idx_B.long())
        value_loss = F.binary_cross_entropy_with_logits(value_B, target_winner_B.float())
        return policy_loss + value_loss, policy_loss, value_loss

    def compute_dense_loss(
        self,
        board_BHW: torch.Tensor,
        target_policy_BC: torch.Tensor,
        target_winner_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss against a dense policy distribution (e.g. MCTS visit counts).

        Uses cross-entropy between predicted log-probs and target distribution,
        equivalent to KL divergence up to a constant.

        Args:
            board_BHW: Board states (B, H, W)
            target_policy_BC: Target probability distribution over actions (B, C)
                              where C = board_size^2 + 1. Must sum to ~1 per row.
            target_winner_B: 1 if current player wins, 0 otherwise

        Returns:
            total_loss, policy_loss, value_loss
        """
        policy_BC, value_B = self.forward(board_BHW)
        policy_loss = F.cross_entropy(policy_BC, target_policy_BC)
        value_loss = F.binary_cross_entropy_with_logits(value_B, target_winner_B.float())
        return policy_loss + value_loss, policy_loss, value_loss



# ============================================================================
# Size-invariant ResNet — trains on mixed 9x9 + 19x19 boards in one net
# ============================================================================
# Best config from autoresearch sweep on
# experiments/2026-04-22_00-15-size-invariant-resnet (run 11, val_loss=3.71):
#   channels=128, n_blocks=10, value_hidden=64 (~3.0M params).
#
# Inputs are padded boards (B, H, W) plus a 0/1 mask (B, H, W) that marks the
# real region of each sample. Every op re-masks the excess region to zero, and
# spatial reductions (avg-pool inside the value/SE heads and masked BN stats)
# divide by the *true* spatial size (mask sum), not the tensor size.

class MaskedBatchNorm2d(nn.Module):
    """BatchNorm2d whose running stats are computed only over real (mask==1)
    positions. In training mode, per-channel mean/var are taken across the
    masked regions of the batch; in eval mode, running stats are used.
    Always re-masks the output so the excess region stays zero."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x_BCHW: torch.Tensor, mask_B1HW: torch.Tensor) -> torch.Tensor:
        if self.training:
            denom = mask_B1HW.sum().clamp(min=1.0)
            mean = (x_BCHW * mask_B1HW).sum(dim=(0, 2, 3)) / denom
            centered = (x_BCHW - mean.view(1, -1, 1, 1)) * mask_B1HW
            var = (centered * centered).sum(dim=(0, 2, 3)) / denom
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var
        inv_std = torch.rsqrt(var + self.eps)
        x_BCHW = (x_BCHW - mean.view(1, -1, 1, 1)) * inv_std.view(1, -1, 1, 1)
        x_BCHW = x_BCHW * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x_BCHW * mask_B1HW


class MaskedGroupNorm2d(nn.Module):
    """GroupNorm computed only over real (mask==1) positions. No running
    stats — train/eval parity. Stats never mix across samples of different
    real board sizes within a batch. `num_groups` defaults to `min(32, C)`."""

    def __init__(self, num_features: int, num_groups: int | None = None,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups if num_groups is not None else min(32, num_features)
        assert num_features % self.num_groups == 0
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x_BCHW: torch.Tensor, mask_B1HW: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_BCHW.shape
        G = self.num_groups
        Cg = C // G
        x = x_BCHW.view(B, G, Cg, H, W) * mask_B1HW.unsqueeze(1)
        denom = mask_B1HW.sum(dim=(1, 2, 3)).clamp(min=1.0) * Cg  # (B,)
        mean = x.sum(dim=(2, 3, 4)) / denom.view(B, 1)  # (B, G)
        centered = (x - mean.view(B, G, 1, 1, 1)) * mask_B1HW.unsqueeze(1)
        var = (centered * centered).sum(dim=(2, 3, 4)) / denom.view(B, 1)
        inv_std = torch.rsqrt(var + self.eps)
        y = centered * inv_std.view(B, G, 1, 1, 1)
        y = y.view(B, C, H, W)
        y = y * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return y * mask_B1HW


class MaskedSEBlock(nn.Module):
    """Squeeze-Excite with masked global-avg-pool. Reduction ratio r >= 1."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x_BCHW: torch.Tensor, mask_B1HW: torch.Tensor) -> torch.Tensor:
        spatial = mask_B1HW.sum(dim=(1, 2, 3)).clamp(min=1.0)  # (B,)
        pooled = (x_BCHW * mask_B1HW).sum(dim=(2, 3)) / spatial.unsqueeze(1)  # (B,C)
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))  # (B,C)
        return x_BCHW * gate.unsqueeze(-1).unsqueeze(-1)


class MaskedResBlock(nn.Module):
    """Two 3x3 convs + masked norm + ReLU, with a residual add and an
    optional Squeeze-Excite gate before the residual add. Re-masks after each
    conv so excess positions stay zero (convolutions at the real/excess
    boundary see zeros on the excess side — equivalent to zero-padding a
    correctly-sized input).

    Pass `norm_cls=MaskedGroupNorm2d` and `use_se=True` to match the
    best-val config from experiments/2026-04-22_07-56-19x19-arch-search.
    """

    def __init__(self, channels: int,
                 norm_cls: type[nn.Module] = MaskedBatchNorm2d,
                 use_se: bool = False, se_reduction: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = norm_cls(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = norm_cls(channels)
        self.se = MaskedSEBlock(channels, reduction=se_reduction) if use_se else None

    def forward(self, x_BCHW: torch.Tensor, mask_B1HW: torch.Tensor) -> torch.Tensor:
        residual = x_BCHW
        out = self.conv1(x_BCHW) * mask_B1HW
        out = F.relu(self.bn1(out, mask_B1HW))
        out = self.conv2(out) * mask_B1HW
        out = self.bn2(out, mask_B1HW)
        if self.se is not None:
            out = self.se(out, mask_B1HW) * mask_B1HW
        out = F.relu(residual + out) * mask_B1HW
        return out


class SizeInvariantGoResNet(nn.Module):
    """Fully convolutional Go net that runs on variable-size boards via
    per-batch zero-padding plus a 0/1 mask.

    - Input: one-hot (empty / self / opp), masked so channel-0 ('empty') is
      zero in the excess region (otherwise the untouched excess would read as
      solid 'empty' and leak into neighbor convolutions).
    - Tower: `n_blocks` of MaskedResBlock.
    - Policy head: 1x1 conv to a single channel, flattened to (B, H*W) with
      excess logits set to -inf (softmax ignores them). Pass logit is a
      linear readout from the masked-avg-pooled tower features.
    - Value head: masked-avg-pool (divisor = true spatial size) → FC → ReLU
      → FC → scalar.

    Call compute_loss(board, mask, move, winner) for CE(move) + BCE(winner);
    the loss indexes moves into the padded (H*W+1) layout using the batch's
    actual H, W, so collate must put each sample's (row, col) in padded
    coordinates (row and col below the sample's native size).
    """

    def __init__(self, channels: int = 128, n_blocks: int = 10,
                 value_hidden: int = 64,
                 norm_type: str = "bn", use_se: bool = False,
                 se_reduction: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.n_blocks = n_blocks
        self.norm_type = norm_type
        self.use_se = use_se
        norm_cls: type[nn.Module] = MaskedGroupNorm2d if norm_type == "gn" else MaskedBatchNorm2d

        self.input_conv = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.input_bn = norm_cls(channels)
        self.blocks = nn.ModuleList([
            MaskedResBlock(channels, norm_cls=norm_cls, use_se=use_se, se_reduction=se_reduction)
            for _ in range(n_blocks)
        ])

        self.policy_conv = nn.Conv2d(channels, 1, 1, bias=True)
        self.pass_fc = nn.Linear(channels, 1)

        self.value_fc1 = nn.Linear(channels, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (MaskedBatchNorm2d, MaskedGroupNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init readouts for calm initial logits/values.
        nn.init.zeros_(self.policy_conv.weight)
        nn.init.zeros_(self.policy_conv.bias)
        nn.init.zeros_(self.pass_fc.weight)
        nn.init.zeros_(self.pass_fc.bias)
        nn.init.zeros_(self.value_fc2.weight)
        nn.init.zeros_(self.value_fc2.bias)

    def forward(self, board_BHW: torch.Tensor,
                mask_BHW: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, W = board_BHW.shape
        device = board_BHW.device
        if mask_BHW is None:
            # No mask → every position is real (typical size-homogeneous batch).
            mask_B1HW = torch.ones(B, 1, H, W, device=device, dtype=torch.float32)
        else:
            mask_B1HW = mask_BHW.unsqueeze(1).float()

        board_long = board_BHW.long().clamp(min=0, max=2)
        x_B3HW = torch.zeros(B, 3, H, W, device=device, dtype=torch.float32)
        x_B3HW.scatter_(1, board_long.unsqueeze(1), 1.0)
        x_B3HW = x_B3HW * mask_B1HW

        x = self.input_conv(x_B3HW) * mask_B1HW
        x = F.relu(self.input_bn(x, mask_B1HW))
        for block in self.blocks:
            x = block(x, mask_B1HW)

        spatial_B = mask_B1HW.sum(dim=(1, 2, 3)).clamp(min=1.0)
        pooled_BC = (x * mask_B1HW).sum(dim=(2, 3)) / spatial_B.unsqueeze(1)

        p_B1HW = self.policy_conv(x) * mask_B1HW
        pos_logits_BL = p_B1HW.view(B, -1)
        mask_BL = mask_B1HW.view(B, -1)
        pos_logits_BL = pos_logits_BL + (1.0 - mask_BL) * (-1e9)
        pass_logit_B1 = self.pass_fc(pooled_BC)
        policy_BC = torch.cat([pos_logits_BL, pass_logit_B1], dim=1)

        v = F.relu(self.value_fc1(pooled_BC))
        value_B = self.value_fc2(v).squeeze(-1)
        return policy_BC, value_B

    def compute_loss(self, board_BHW: torch.Tensor,
                     mask_BHW: torch.Tensor | None,
                     target_move_B2: torch.Tensor, target_winner_B: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, W = board_BHW.shape
        policy_BC, value_B = self.forward(board_BHW, mask_BHW)
        n_positions = H * W
        is_pass = target_move_B2[:, 0] < 0
        target_idx_B = torch.where(
            is_pass,
            torch.full_like(target_move_B2[:, 0], n_positions),
            target_move_B2[:, 0] * W + target_move_B2[:, 1],
        ).long()
        policy_loss = F.cross_entropy(policy_BC, target_idx_B)
        value_loss = F.binary_cross_entropy_with_logits(value_B, target_winner_B.float())
        return policy_loss + value_loss, policy_loss, value_loss

    def compute_dense_loss(self, board_BHW: torch.Tensor,
                           mask_BHW: torch.Tensor | None,
                           target_policy_BC: torch.Tensor,
                           target_winner_B: torch.Tensor,
                           is_teacher_B: torch.Tensor | None = None,
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample CE against a dense policy target (MCTS visits post-
        temperature). target_policy_BC must be in the same (H*W+1) padded
        layout as the model's policy output. If is_teacher_B is given, the
        policy loss is averaged only over teacher samples (value loss always
        applies to all)."""
        policy_BC, value_B = self.forward(board_BHW, mask_BHW)
        logp = F.log_softmax(policy_BC, dim=-1)
        policy_ce_B = -(target_policy_BC * logp).sum(dim=-1)
        if is_teacher_B is None:
            policy_loss = policy_ce_B.mean()
        else:
            w = is_teacher_B.float()
            policy_loss = (policy_ce_B * w).sum() / w.sum().clamp_min(1.0)
        value_loss = F.binary_cross_entropy_with_logits(value_B, target_winner_B.float())
        return policy_loss + value_loss, policy_loss, value_loss


# ============================================================================
# Factory Functions
# ============================================================================

def create_mup_model(
    config: MuPModelConfig | str = "100M",
    board_size: int = 9,
    init_scale: float = 1.0,
    device: torch.device | str = "cpu",
    base_width: int | None = None,
    delta_width: int | None = None,
    base_depth: int | None = None,
    delta_depth: int | None = None,
) -> MuPGoResNet:
    """Create a muP-parameterized model with base shapes set.

    Note: muP requires base/delta/target to have the same depth (n_blocks).
    Only width (channels) can vary for muP scaling to work correctly.

    Args:
        config: Model configuration or name (e.g., "100M", "1B")
        board_size: Board size (default 9 for 9x9 Go)
        init_scale: Initialization scale for weights
        device: Device to place model on
        base_width: Override base model width (default: MUP_BASE_WIDTH)
        delta_width: Override delta model width (default: MUP_DELTA_WIDTH)
        base_depth: Override base model depth (default: config.n_blocks)
        delta_depth: Override delta model depth (default: config.n_blocks)

    Returns:
        MuPGoResNet model with muP base shapes configured
    """
    if isinstance(config, str):
        if config not in MODEL_CONFIGS:
            raise ValueError(f"Unknown config: {config}. Available: {list(MODEL_CONFIGS.keys())}")
        config = MODEL_CONFIGS[config]

    if isinstance(device, str):
        device = torch.device(device)

    # Use defaults if not specified
    base_w = base_width if base_width is not None else MUP_BASE_WIDTH
    delta_w = delta_width if delta_width is not None else MUP_DELTA_WIDTH
    base_d = base_depth if base_depth is not None else config.n_blocks
    delta_d = delta_depth if delta_depth is not None else config.n_blocks

    # Create base and delta models for shape inference
    base_model = MuPGoResNet(
        board_size=board_size,
        channels=base_w,
        n_blocks=base_d,
        init_scale=init_scale,
        policy_channels=config.policy_channels,
        value_channels=config.value_channels,
    )
    delta_model = MuPGoResNet(
        board_size=board_size,
        channels=delta_w,
        n_blocks=delta_d,
        init_scale=init_scale,
        policy_channels=config.policy_channels,
        value_channels=config.value_channels,
    )

    # Create target model
    model = MuPGoResNet(
        board_size=board_size,
        channels=config.channels,
        n_blocks=config.n_blocks,
        init_scale=init_scale,
        policy_channels=config.policy_channels,
        value_channels=config.value_channels,
    )

    # Set base shapes for muP
    set_base_shapes(model, base_model, delta=delta_model)

    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters())


def get_model_info(model: MuPGoResNet) -> dict[str, int | float]:
    """Get model information dictionary."""
    n_params = count_parameters(model)
    return {
        "channels": model.channels,
        "n_blocks": model.n_blocks,
        "policy_channels": model.policy_channels,
        "value_channels": model.value_channels,
        "n_params": n_params,
        "n_params_m": n_params / 1e6,
        "board_size": model.board_size,
        "init_scale": model.init_scale,
    }


def upgrade_state_dict_for_pass(
    state_dict: dict[str, torch.Tensor],
    board_size: int = 9,
) -> dict[str, torch.Tensor]:
    """Upgrade old state dict (81 outputs) to new format (82 outputs).

    Old checkpoints have policy_fc with board_size^2 outputs (81 for 9x9).
    New models have board_size^2 + 1 outputs (82 for 9x9) to include pass.

    This function pads policy_fc.weight and policy_fc.bias with zeros for
    the pass action if needed.

    Args:
        state_dict: Model state dict to upgrade (modified in place)
        board_size: Board size (default 9)

    Returns:
        Upgraded state dict (same reference, modified in place)
    """
    n_positions = board_size * board_size  # 81 for 9x9
    n_actions = n_positions + 1  # 82 for 9x9

    if "policy_fc.weight" in state_dict:
        old_weight = state_dict["policy_fc.weight"]
        if old_weight.shape[0] == n_positions:
            # Old checkpoint with 81 outputs - pad with zeros for pass action
            new_weight = torch.zeros(
                n_actions, old_weight.shape[1],
                dtype=old_weight.dtype, device=old_weight.device
            )
            new_weight[:n_positions] = old_weight
            state_dict["policy_fc.weight"] = new_weight

            if "policy_fc.bias" in state_dict:
                old_bias = state_dict["policy_fc.bias"]
                new_bias = torch.zeros(n_actions, dtype=old_bias.dtype, device=old_bias.device)
                new_bias[:n_positions] = old_bias
                state_dict["policy_fc.bias"] = new_bias

    return state_dict
