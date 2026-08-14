"""Experiment-local ResNet with one signed-komi absolute-color input plane."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import alpha_go_cpp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_go.agents.base import get_pass_index
from alpha_go.model import SizeInvariantGoResNet


class SignedKomiGoResNet(SizeInvariantGoResNet):
    """SizeInvariantGoResNet plus a constant current-player signed-komi plane."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.input_conv = nn.Conv2d(4, self.channels, 3, padding=1, bias=False)
        nn.init.kaiming_normal_(
            self.input_conv.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(
        self,
        board_BHW: torch.Tensor,
        signed_komi_B: torch.Tensor,
        mask_BHW: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, height, width = board_BHW.shape
        if signed_komi_B.shape != (batch,):
            raise ValueError(
                f"signed_komi must have shape ({batch},), got {tuple(signed_komi_B.shape)}"
            )
        device = board_BHW.device
        if mask_BHW is None:
            mask_B1HW = torch.ones(
                batch, 1, height, width, device=device, dtype=torch.float32
            )
        else:
            mask_B1HW = mask_BHW.unsqueeze(1).float()

        board_long = board_BHW.long().clamp(min=0, max=2)
        x_B3HW = torch.zeros(
            batch, 3, height, width, device=device, dtype=torch.float32
        )
        x_B3HW.scatter_(1, board_long.unsqueeze(1), 1.0)
        komi_B1HW = signed_komi_B.float().view(batch, 1, 1, 1).expand(
            -1, 1, height, width
        )
        x = torch.cat([x_B3HW, komi_B1HW], dim=1) * mask_B1HW

        x = self.input_conv(x) * mask_B1HW
        x = F.relu(self.input_bn(x, mask_B1HW))
        for block in self.blocks:
            x = block(x, mask_B1HW)

        spatial_B = mask_B1HW.sum(dim=(1, 2, 3)).clamp(min=1.0)
        pooled_BC = (x * mask_B1HW).sum(dim=(2, 3)) / spatial_B.unsqueeze(1)

        p_B1HW = self.policy_conv(x) * mask_B1HW
        pos_logits_BL = p_B1HW.view(batch, -1)
        mask_BL = mask_B1HW.view(batch, -1)
        pos_logits_BL = pos_logits_BL + (1.0 - mask_BL) * (-1e9)
        policy_BC = torch.cat([pos_logits_BL, self.pass_fc(pooled_BC)], dim=1)

        value_B = self.value_fc2(F.relu(self.value_fc1(pooled_BC))).squeeze(-1)
        return policy_BC, value_B


def signed_komi_for_current_player(
    is_white: torch.Tensor | bool,
    *,
    komi: float,
    board_size: int,
) -> torch.Tensor:
    """Return +komi/area for White to play and -komi/area for Black to play."""
    white = torch.as_tensor(is_white, dtype=torch.bool)
    magnitude = float(komi) / float(board_size * board_size)
    positive = torch.full(
        white.shape, magnitude, dtype=torch.float32, device=white.device
    )
    return torch.where(white, positive, -positive)


class SignedKomiLeafBatchedNNEvaluator:
    """Leaf-batched evaluator for the experiment's four-plane model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        board_size: int,
        komi: float,
        model_config: Mapping[str, Any],
        device: str | None = None,
        policy_temperature: float = 1.0,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self.board_size = int(board_size)
        self.komi = float(komi)
        self.policy_temperature = float(policy_temperature)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        payload = torch.load(
            self._checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model = SignedKomiGoResNet(
            channels=int(model_config["channels"]),
            n_blocks=int(model_config["residual_blocks"]),
            value_hidden=int(model_config["value_hidden"]),
            norm_type=str(model_config["norm_type"]),
            use_se=bool(model_config["use_se"]),
        ).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self._pass_index = get_pass_index(self.board_size)
        self._n_actions = self._pass_index + 1

    @torch.no_grad()
    def evaluate(self, cpp_board: Any) -> tuple[dict[int, float], float]:
        return self.batch_evaluate([cpp_board])[0]

    @torch.no_grad()
    def batch_evaluate(
        self, cpp_boards: list[Any]
    ) -> list[tuple[dict[int, float], float]]:
        batch = len(cpp_boards)
        boards_np = np.empty(
            (batch, self.board_size, self.board_size), dtype=np.float32
        )
        is_white = np.empty(batch, dtype=np.bool_)
        for index, board in enumerate(cpp_boards):
            array = board.to_numpy().astype(np.float32)
            white_to_play = board.to_play() == alpha_go_cpp.GoBoard.WHITE
            is_white[index] = white_to_play
            if white_to_play:
                array = np.where(
                    array == 1, 2.0, np.where(array == 2, 1.0, array)
                )
            boards_np[index] = array
        boards = torch.from_numpy(boards_np).to(self.device, non_blocking=True)
        signed_komi = signed_komi_for_current_player(
            torch.from_numpy(is_white), komi=self.komi, board_size=self.board_size
        ).to(self.device, non_blocking=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            policy_logits, value_logits = self.model(boards, signed_komi)
        values = torch.sigmoid(value_logits).float().cpu().numpy().reshape(-1)
        policy_np = policy_logits.float().cpu().numpy()

        results: list[tuple[dict[int, float], float]] = []
        for index, board in enumerate(cpp_boards):
            legal = board.get_legal_moves_flat()
            mask = np.full(self._n_actions, -np.inf, dtype=np.float32)
            for action in legal:
                mask[action] = 0.0
            mask[self._pass_index] = 0.0
            logits = (policy_np[index] + mask) / self.policy_temperature
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            policy = {
                int(action): float(probabilities[action]) for action in legal
            }
            policy[alpha_go_cpp.PASS_ACTION] = float(
                probabilities[self._pass_index]
            )
            results.append((policy, float(values[index])))
        return results

    @property
    def checkpoint_path(self) -> str:
        return str(self._checkpoint_path)

    def close(self) -> None:
        pass
