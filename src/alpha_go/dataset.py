"""Dataset for loading Go game data from self-play."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

BLACK = 1
WHITE = 2


class GoDataset(Dataset):
    """Dataset for Go game positions from self-play data.

    Supports loading from a single directory or multiple directories (combined).

    Each sample returns:
        board: Board state flipped so current player's stones are 1, opponent's are 2
        is_expert: Boolean indicating if current player won the game
        move: Next move as (row, col) or (-1, -1) for pass
        winner: 1 if current player wins, 0 otherwise
    """

    def __init__(
        self,
        data_dir: str | Path | list[str | Path],
        board_size: int | None = None,
        load_mcts_policy: bool = False,
        load_is_teacher: bool = False,
        in_memory: bool = False,
        policy_target_temperature: float | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            data_dir: Path(s) to directory(ies) containing .npz files.
                      Can be a single path or a list of paths to combine datasets.
                      If board_size is provided, appends "{board_size}x{board_size}".
            board_size: If provided, appends to data_dir. If None, uses data_dir directly
                        and infers board_size from first file.
            load_mcts_policy: If True, load MCTS visit counts and recompute improved
                             policy targets from NPZ files. Without an explicit target
                             temperature this also requires mcts_temperatures. Data that
                             lacks visit counts (e.g. katago teacher) gets a
                             label-smoothed one-hot target.
            policy_target_temperature: If set, use this temperature instead of the
                                       saved action-selection temperature when
                                       reconstructing MCTS policy targets. A value
                                       of 1.0 preserves the raw visit distribution.
        """
        self.load_mcts_policy = load_mcts_policy
        self.load_is_teacher = load_is_teacher
        if policy_target_temperature is not None:
            policy_target_temperature = float(policy_target_temperature)
            if (
                not np.isfinite(policy_target_temperature)
                or policy_target_temperature < 0
            ):
                raise ValueError("policy_target_temperature must be finite and non-negative")
        self.policy_target_temperature = policy_target_temperature
        # Normalize to list of paths
        if isinstance(data_dir, (str, Path)):
            data_dirs = [Path(data_dir)]
        else:
            data_dirs = [Path(d) for d in data_dir]

        if board_size is not None:
            data_dirs = [d / f"{board_size}x{board_size}" for d in data_dirs]

        # Validate all directories exist
        for d in data_dirs:
            if not d.exists():
                raise FileNotFoundError(f"Data directory not found: {d}")

        self.data_dirs = data_dirs

        # Build indices for each directory
        # Structure: list of (data_dir, files_list, cumsum_array)
        self._dir_info: list[tuple[Path, list[str], np.ndarray]] = []
        total = 0

        for data_dir_path in data_dirs:
            index = self._load_or_build_index(data_dir_path)
            files = list(index.keys())
            cumsum = np.cumsum([0] + [index[f] for f in files])
            self._dir_info.append((data_dir_path, files, cumsum))
            total += int(cumsum[-1])

        self.total_positions = total

        # Build global cumsum across directories
        dir_totals = [int(info[2][-1]) for info in self._dir_info]
        self._dir_cumsum = np.cumsum([0] + dir_totals)

        # Optional in-memory cache: pre-load every NPZ into a dict so __getitem__
        # hits RAM instead of re-opening the file each sample (critical when data
        # sits on NFS).
        self._npz_cache: dict[Path, dict] | None = None
        if in_memory:
            self._npz_cache = {}
            for data_dir_path, files, _ in self._dir_info:
                for fname in files:
                    p = data_dir_path / fname
                    self._npz_cache[p] = {k: v for k, v in np.load(p).items()}

        # Infer board_size from first file if not provided
        if board_size is not None:
            self.board_size = board_size
        elif self._dir_info and self._dir_info[0][1]:
            data_dir_path, files, _ = self._dir_info[0]
            data = np.load(data_dir_path / files[0])
            # Handle files that may not have board_size key
            if "board_size" in data.keys():
                self.board_size = int(data["board_size"])
            elif "boards" in data.keys() and len(data["boards"].shape) >= 2:
                # Infer from board shape
                self.board_size = data["boards"].shape[-1]
            else:
                self.board_size = 9  # Default
        else:
            self.board_size = 9  # Default

    def _load_or_build_index(self, data_dir: Path) -> dict[str, int]:
        """Load or build index for a single directory."""
        index_path = data_dir / "index.json"

        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            # Validate index against actual files
            npz_files = {p.name for p in data_dir.glob("*.npz")}
            if npz_files == set(index.keys()):
                return index

        # Build index
        index = {}
        for npz_path in sorted(data_dir.glob("*.npz")):
            data = np.load(npz_path)
            index[npz_path.name] = int(data["num_moves"])

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        return index

    def __len__(self) -> int:
        return self.total_positions

    def __getitem__(self, idx: int) -> dict:
        """Get a single position.

        Returns dict with keys:
            board: (H, W) tensor, current player stones=1, opponent=2, empty=0
            is_expert: bool, True if current player is the expert agent
            move: (2,) tensor, next move (row, col) or (-1, -1) for pass
            winner: int, 1 if current player wins, 0 otherwise
        """
        if idx < 0 or idx >= self.total_positions:
            raise IndexError(f"Index {idx} out of range [0, {self.total_positions})")

        # Find which directory this index belongs to
        dir_idx = int(np.searchsorted(self._dir_cumsum[1:], idx, side="right"))
        idx_in_dir = idx - int(self._dir_cumsum[dir_idx])

        data_dir, files, cumsum = self._dir_info[dir_idx]

        # Find which file within the directory
        file_idx = int(np.searchsorted(cumsum[1:], idx_in_dir, side="right"))
        local_idx = idx_in_dir - int(cumsum[file_idx])

        # Load data
        npz_path = data_dir / files[file_idx]
        data = self._npz_cache[npz_path] if self._npz_cache is not None else np.load(npz_path)

        board = data["boards"][local_idx].copy()  # (H, W)
        move = data["moves"][local_idx]  # (2,) or [-1, -1] for pass
        winner = int(data["winner"])  # 1=BLACK, 2=WHITE, 0=draw

        # Determine current player (even idx = BLACK, odd idx = WHITE)
        is_white = (local_idx % 2 == 1)
        current_player = WHITE if is_white else BLACK

        # Expert = current player won the game
        is_expert = (winner == current_player)

        # Flip board so current player's stones are always 1
        if is_white:
            board = np.where(board == BLACK, WHITE,
                            np.where(board == WHITE, BLACK, board))

        # Determine if current player wins (1=win, 0=loss/draw)
        current_wins = 1 if winner == current_player else 0

        result = {
            "board": torch.from_numpy(board).float(),
            "is_expert": is_expert,
            "move": torch.tensor([move[0], move[1]], dtype=torch.long),
            "winner": current_wins,
        }

        # Load MCTS improved policy or label-smoothed ground truth
        if self.load_mcts_policy:
            bs = board.shape[-1]
            n_actions = bs * bs + 1

            if "mcts_visits" in data:
                visits = data["mcts_visits"][local_idx].astype(np.float32)  # (H*W+1,)
                temperature = self.policy_target_temperature
                if temperature is None:
                    temperature = float(data["mcts_temperatures"][local_idx])
                root_value = float(data["mcts_root_values"][local_idx])

                # Recompute policy: N^(1/tau) / sum(N^(1/tau))
                # The optional override decouples this target from move sampling.
                if temperature == 0:
                    mcts_policy = np.zeros(n_actions, dtype=np.float32)
                    if visits.sum() > 0:
                        mcts_policy[np.argmax(visits)] = 1.0
                else:
                    visits_temp = np.power(visits, 1.0 / temperature)
                    total = visits_temp.sum()
                    mcts_policy = visits_temp / total if total > 0 else np.zeros(n_actions, dtype=np.float32)

                has_mcts = True
            else:
                # Fallback: label-smoothed one-hot from ground-truth move
                smooth_eps = 0.1
                mcts_policy = np.full(n_actions, smooth_eps / n_actions, dtype=np.float32)
                r, c = int(move[0]), int(move[1])
                if r < 0:
                    # Pass move
                    target_idx = n_actions - 1
                else:
                    target_idx = r * bs + c
                mcts_policy[target_idx] += 1.0 - smooth_eps
                root_value = float(current_wins)
                has_mcts = False

            result["mcts_policy"] = torch.from_numpy(mcts_policy).float()
            result["mcts_root_value"] = torch.tensor(root_value, dtype=torch.float32)
            result["has_mcts"] = has_mcts

        if self.load_is_teacher:
            flag = bool(data["is_teacher"][local_idx]) if "is_teacher" in data else False
            result["is_teacher"] = torch.tensor(flag, dtype=torch.float32)

        return result

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        return {
            "total_positions": self.total_positions,
            "num_dirs": len(self.data_dirs),
            "num_files": sum(len(info[1]) for info in self._dir_info),
            "board_size": self.board_size,
            "data_dirs": [str(d) for d in self.data_dirs],
        }
