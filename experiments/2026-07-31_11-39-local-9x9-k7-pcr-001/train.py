#!/usr/bin/env python
"""Bounded, resumable candidate training with correct two-sided MCTS masks."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from color_model import (
    SignedKomiGoResNet,
    signed_komi_for_current_player,
)
from common import (
    CHECKPOINT_ROOT,
    EXP_NAME,
    GAME_DATA_ROOT,
    RUNTIME_DIR,
    atomic_write_json,
    load_config,
    sha256_file,
)
from packed_dataset import PackedDrawAwareGoDataset, prepare_packed_cache
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from alpha_go.dataset import GoDataset
from alpha_go.model import count_parameters

sys.stdout.reconfigure(line_buffering=True)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def initialize_distributed(device_mode: str) -> DistributedContext:
    """Initialize one-process-per-GPU DDP from torchrun environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(
            f"invalid distributed ranks rank={rank} local_rank={local_rank} world_size={world_size}"
        )
    if world_size > 1:
        if device_mode != "cuda":
            raise ValueError("multi-process training requires --device cuda")
        if not 0 <= local_rank < torch.cuda.device_count():
            raise ValueError(
                f"LOCAL_RANK={local_rank} outside visible CUDA devices={torch.cuda.device_count()}"
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def distributed_sum(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.enabled:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def distributed_max(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.enabled:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def synchronize_running_stats(
    model: torch.nn.Module,
    context: DistributedContext,
) -> None:
    """Average custom masked-BN running statistics before eval/checkpoint I/O."""
    if not context.enabled:
        return
    buffers = [
        buffer
        for name, buffer in model.named_buffers()
        if name.endswith("running_mean") or name.endswith("running_var")
    ]
    if not buffers:
        return
    sizes = [buffer.numel() for buffer in buffers]
    flattened = torch.cat([buffer.detach().reshape(-1) for buffer in buffers])
    dist.all_reduce(flattened, op=dist.ReduceOp.SUM)
    flattened.div_(context.world_size)
    offset = 0
    with torch.no_grad():
        for buffer, size in zip(buffers, sizes, strict=True):
            buffer.copy_(flattened[offset : offset + size].view_as(buffer))
            offset += size


def broadcast_flag(
    value: bool,
    *,
    device: torch.device,
    context: DistributedContext,
) -> bool:
    values = [int(value)] if context.is_primary else [0]
    flags = torch.tensor(values, dtype=torch.int32, device=device)
    if context.enabled:
        dist.broadcast(flags, src=0)
    return bool(flags[0].item())


def gather_scalar(
    value: float,
    *,
    device: torch.device,
    context: DistributedContext,
) -> list[float]:
    local = torch.tensor([value], dtype=torch.float64, device=device)
    if not context.enabled:
        return [value]
    gathered = [torch.zeros_like(local) for _ in range(context.world_size)]
    dist.all_gather(gathered, local)
    return [float(item.item()) for item in gathered]


class DrawAwareGoDataset(GoDataset):
    """Preserve draws and expose the current-player signed-komi feature."""

    def __init__(self, *args: Any, komi: float, **kwargs: Any) -> None:
        self.komi = float(komi)
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        result = super().__getitem__(idx)
        dir_idx = int(np.searchsorted(self._dir_cumsum[1:], idx, side="right"))
        idx_in_dir = idx - int(self._dir_cumsum[dir_idx])
        data_dir, files, cumsum = self._dir_info[dir_idx]
        file_idx = int(np.searchsorted(cumsum[1:], idx_in_dir, side="right"))
        local_idx = idx_in_dir - int(cumsum[file_idx])
        path = data_dir / files[file_idx]
        data = self._npz_cache[path] if self._npz_cache is not None else np.load(path)
        raw_winner = int(data["winner"])
        sample_komi = self.komi
        if "komi" in data:
            stored_komi = np.asarray(data["komi"])
            if stored_komi.size != 1:
                raise ValueError(f"invalid komi shape {stored_komi.shape}: {path}")
            sample_komi = float(stored_komi.item())
            if not np.isfinite(sample_komi):
                raise ValueError(f"non-finite komi {sample_komi}: {path}")
        value_target = 0.5 if raw_winner == 0 else float(result["winner"])
        result["winner"] = torch.tensor(value_target, dtype=torch.float32)
        result["signed_komi"] = signed_komi_for_current_player(
            bool(local_idx % 2), komi=sample_komi, board_size=self.board_size
        )
        return result


def load_manifest(path: Path) -> list[Path]:
    paths: list[Path] = []
    for line in path.read_text().splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        resolved = (GAME_DATA_ROOT / item).resolve()
        if not resolved.is_relative_to(GAME_DATA_ROOT):
            raise ValueError(f"manifest path escapes data root: {item}")
        if not resolved.is_dir():
            raise FileNotFoundError(f"manifest directory missing: {resolved}")
        paths.append(resolved)
    if not paths:
        raise ValueError(f"manifest has no data directories: {path}")
    return paths


def ensure_dataset_indexes(
    paths: list[Path],
    context: DistributedContext,
) -> None:
    """Have rank 0 atomically build missing indexes before parallel dataset load."""
    if context.is_primary:
        for directory in paths:
            npz_paths = sorted(directory.glob("*.npz"))
            expected_names = {path.name for path in npz_paths}
            index_path = directory / "index.json"
            try:
                existing = json.loads(index_path.read_text()) if index_path.exists() else {}
            except (OSError, ValueError, TypeError):
                existing = {}
            if set(existing) == expected_names:
                continue
            index: dict[str, int] = {}
            for path in npz_paths:
                with np.load(path, allow_pickle=False) as data:
                    index[path.name] = int(data["num_moves"])
            atomic_write_json(index_path, index)
    if context.enabled:
        dist.barrier()


def augment_batch_dense(
    board: torch.Tensor,
    policy: torch.Tensor,
    board_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independent random D4 transform to each board and policy."""
    board = board.clone()
    policy = policy.clone()
    batch_size = board.shape[0]
    spatial = policy[:, : board_size * board_size].view(batch_size, board_size, board_size)
    pass_prob = policy[:, board_size * board_size :]
    transforms = torch.randint(0, 8, (batch_size,), device=board.device)
    flip = transforms >= 4
    if flip.any():
        board[flip] = board[flip].flip(-1)
        spatial[flip] = spatial[flip].flip(-1)
    rotations = transforms % 4
    for k in (1, 2, 3):
        selected = rotations == k
        if selected.any():
            board[selected] = torch.rot90(board[selected], k, (-2, -1))
            spatial[selected] = torch.rot90(spatial[selected], k, (-2, -1))
    transformed = torch.cat([spatial.reshape(batch_size, -1), pass_prob], dim=-1)
    return board, transformed


def build_policy_mask(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Use full-search teachers; bootstrap fallback only for winning moves."""
    has_mcts = batch["has_mcts"].bool()
    is_teacher = batch["is_teacher"].bool()
    is_expert = batch["is_expert"].bool()
    invalid = is_teacher & ~has_mcts
    if bool(invalid.any()):
        raise ValueError(
            f"found {int(invalid.sum())} teacher positions without MCTS visits"
        )
    fallback = ~has_mcts & is_expert
    mask = is_teacher | fallback
    return mask.float(), fallback


def dense_loss(
    model: torch.nn.Module,
    board: torch.Tensor,
    signed_komi: torch.Tensor,
    policy_target: torch.Tensor,
    winner: torch.Tensor,
    policy_mask: torch.Tensor,
    policy_weight: float,
    value_weight: float,
    context: DistributedContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_logits, value_logits = model(board, signed_komi)
    log_probs = F.log_softmax(policy_logits, dim=-1)
    per_sample = -(policy_target * log_probs).sum(dim=-1)
    policy_sum = (per_sample * policy_mask).sum()
    policy_count = policy_mask.sum().detach()
    if context is not None and context.enabled:
        global_policy_count = distributed_sum(policy_count.clone(), context)
        # DDP averages parameter gradients across ranks. Multiplying each local
        # numerator by world_size preserves the exact global masked mean.
        policy_loss = (
            policy_sum * context.world_size / global_policy_count.clamp_min(1.0)
        )
    else:
        policy_loss = policy_sum / policy_count.clamp_min(1.0)
    value_loss = F.binary_cross_entropy_with_logits(value_logits, winner.float())
    total = policy_weight * policy_loss + value_weight * value_loss
    return total, policy_loss, value_loss, policy_logits, value_logits


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def rng_state(device: torch.device | None = None) -> dict[str, Any]:
    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available():
        if device is not None and device.type == "cuda":
            cuda_states = [torch.cuda.get_rng_state(device)]
        else:
            cuda_states = torch.cuda.get_rng_state_all()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
    }


def gather_rng_states(
    device: torch.device,
    context: DistributedContext,
) -> list[dict[str, Any]] | None:
    local_state = rng_state(device)
    if not context.enabled:
        return [local_state]
    gathered: list[dict[str, Any] | None] | None = (
        [None] * context.world_size if context.is_primary else None
    )
    dist.gather_object(local_state, gathered, dst=0)
    if not context.is_primary:
        return None
    assert gathered is not None
    return [state for state in gathered if state is not None]


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    with open(tmp, "rb") as handle:
        os.fsync(handle.fileno())
    check = torch.load(tmp, map_location="cpu", weights_only=False)
    if "model_state_dict" not in check or "iteration" not in check:
        raise ValueError(f"checkpoint validation failed: {tmp}")
    os.replace(tmp, path)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    iteration: int,
    step: int,
    epoch: int,
    batch_offset: int,
    manifest: Path,
    source_champion: str,
    config: dict[str, Any],
    rng_states: list[dict[str, Any]] | None = None,
    world_size: int = 1,
) -> dict[str, Any]:
    states = rng_states or [rng_state()]
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "iteration": iteration,
        "step": step,
        "epoch": epoch,
        "batch_offset": batch_offset,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "source_champion": source_champion,
        "model_config": config["model"],
        "training_config": config["training"],
        "training_world_size": world_size,
        "experiment_name": EXP_NAME,
        "rng_state": states[0],
        "rng_state_by_rank": states,
        "completed": True,
    }


def make_loader(
    dataset: GoDataset,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    rank: int = 0,
    world_size: int = 1,
    persistent_workers: bool | None = None,
) -> DataLoader:
    sampler: DistributedSampler | None = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=shuffle,
        )
        sampler.set_epoch(epoch)
    generator = torch.Generator()
    generator.manual_seed(seed + epoch + rank * 1_000_003)
    keep_workers_alive = shuffle if persistent_workers is None else persistent_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=workers > 0 and keep_workers_alive,
        generator=generator,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    context: DistributedContext | None = None,
    max_batches: int = 50,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    policy_correct = 0
    policy_samples = 0
    value_correct = 0
    value_samples = 0
    batches = 0
    use_cuda = device.type == "cuda"
    for batch in loader:
        if batches >= max_batches:
            break
        policy_mask, _ = build_policy_mask(batch)
        board = batch["board"].to(device, non_blocking=True)
        signed_komi = batch["signed_komi"].to(device, non_blocking=True)
        winner = batch["winner"].to(device, non_blocking=True)
        policy = batch["mcts_policy"].to(device, non_blocking=True)
        policy_mask = policy_mask.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            loss, _, _, policy_logits, value_logits = dense_loss(
                model,
                board,
                signed_komi,
                policy,
                winner,
                policy_mask,
                float(config["training"]["policy_loss_weight"]),
                float(config["training"]["value_loss_weight"]),
                context,
            )
        total_loss += float(loss)
        policy_correct += int(((policy_logits.argmax(-1) == policy.argmax(-1)) * policy_mask.bool()).sum())
        policy_samples += int(policy_mask.sum())
        decisive = winner != 0.5
        value_correct += int(
            (((value_logits > 0).long() == winner.long()) & decisive).sum()
        )
        value_samples += int(decisive.sum())
        batches += 1
    statistics = torch.tensor(
        [
            total_loss,
            policy_correct,
            policy_samples,
            value_correct,
            value_samples,
            batches,
        ],
        dtype=torch.float64,
        device=device,
    )
    if context is not None:
        distributed_sum(statistics, context)
    (
        total_loss,
        policy_correct,
        policy_samples,
        value_correct,
        value_samples,
        batches,
    ) = statistics.tolist()
    model.train()
    return {
        "loss": total_loss / max(1, batches),
        "policy_accuracy": policy_correct / max(1, policy_samples),
        "value_accuracy": value_correct / max(1, value_samples),
        "policy_samples": float(policy_samples),
    }


def main() -> int:
    config = load_config()
    train_cfg = config["training"]
    model_cfg = config["model"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--source-champion", default="")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--time-budget", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dataloader-workers", type=int)
    parser.add_argument("--ddp-bucket-cap-mb", type=float)
    parser.add_argument(
        "--dataset-storage",
        choices=("legacy_in_memory", "packed_mmap"),
    )
    parser.add_argument("--dataset-cache-root", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but torch.cuda.is_available() is false")
    context = initialize_distributed(args.device)
    device = torch.device(
        f"cuda:{context.local_rank}" if args.device == "cuda" else "cpu"
    )
    use_cuda = device.type == "cuda"
    seed = int(config["base_seed"]) + args.iteration * 100_000
    rank_seed = seed + context.rank * 1_000_003
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if use_cuda:
        torch.cuda.init()
        torch.cuda.manual_seed(rank_seed)
        torch.cuda.reset_peak_memory_stats(device)

    manifest = args.manifest.resolve()
    paths = load_manifest(manifest)
    ensure_dataset_indexes(paths, context)
    started_loading = time.time()
    policy_target_temperature = config["dataset"].get("policy_target_temperature")
    target_temperature = (
        None
        if policy_target_temperature is None
        else float(policy_target_temperature)
    )
    dataset_storage = str(
        args.dataset_storage
        or config["dataset"].get("storage", "legacy_in_memory")
    )
    cache_root: Path | None = None
    cache_summary: dict[str, int | float] | None = None
    if dataset_storage == "packed_mmap":
        cache_root = (
            args.dataset_cache_root or RUNTIME_DIR / "dataset-cache-v2"
        ).resolve()
        if context.is_primary:
            cache_summary = prepare_packed_cache(
                paths,
                cache_root=cache_root,
                board_size=int(config["rules"]["board_size"]),
            )
        if context.enabled:
            dist.barrier()
        dataset = PackedDrawAwareGoDataset(
            paths,
            cache_root=cache_root,
            board_size=int(config["rules"]["board_size"]),
            komi=float(config["rules"]["komi"]),
            load_mcts_policy=True,
            load_is_teacher=True,
            policy_target_temperature=target_temperature,
        )
    else:
        dataset = DrawAwareGoDataset(
            paths,
            komi=float(config["rules"]["komi"]),
            load_mcts_policy=True,
            load_is_teacher=True,
            in_memory=bool(config["dataset"]["in_memory"]),
            policy_target_temperature=target_temperature,
        )
    global_batch_size = int(args.batch_size or train_cfg["batch_size"])
    if global_batch_size % context.world_size != 0:
        raise ValueError(
            f"global batch_size={global_batch_size} is not divisible by "
            f"world_size={context.world_size}"
        )
    batch_size = global_batch_size // context.world_size
    workers = int(
        train_cfg["dataloader_workers"]
        if args.dataloader_workers is None
        else args.dataloader_workers
    )
    ddp_bucket_cap_mb = float(
        train_cfg.get("ddp_bucket_cap_mb", 25.0)
        if args.ddp_bucket_cap_mb is None
        else args.ddp_bucket_cap_mb
    )
    if ddp_bucket_cap_mb <= 0.0:
        raise ValueError("ddp_bucket_cap_mb must be positive")
    probe_loader = make_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        seed=seed,
        epoch=0,
        shuffle=True,
        rank=context.rank,
        world_size=context.world_size,
    )
    steps_per_epoch = len(probe_loader)
    if steps_per_epoch == 0:
        raise ValueError(
            f"dataset has {len(dataset)} positions, smaller than "
            f"global_batch_size={global_batch_size}"
        )
    expected_steps = len(dataset) // global_batch_size
    if steps_per_epoch != expected_steps:
        raise ValueError(
            f"distributed loader has {steps_per_epoch} steps, expected {expected_steps} "
            f"for positions={len(dataset)} global_batch={global_batch_size}"
        )
    configured_max = max(
        int(train_cfg["minimum_steps"]),
        int(train_cfg["max_epochs"]) * steps_per_epoch,
    )
    max_steps = int(args.max_steps or configured_max)
    time_budget = int(args.time_budget or train_cfg["time_budget_seconds"])
    load_seconds = torch.tensor(
        [time.time() - started_loading], dtype=torch.float64, device=device
    )
    distributed_max(load_seconds, context)
    if context.is_primary:
        print(
            f"dataset_dirs={len(paths)} positions={len(dataset):,} "
            f"load_seconds={float(load_seconds.item()):.1f} "
            f"storage={dataset_storage} "
            f"world_size={context.world_size} global_batch={global_batch_size} "
            f"per_rank_batch={batch_size} workers_per_rank={workers} "
            f"ddp_bucket_cap_mb={ddp_bucket_cap_mb:g} "
            f"steps_per_epoch={steps_per_epoch} max_steps={max_steps}",
            flush=True,
        )
        if cache_summary is not None:
            print(
                f"dataset_cache_root={cache_root} "
                f"cache_summary={json.dumps(cache_summary, sort_keys=True)}",
                flush=True,
            )

    raw_model = SignedKomiGoResNet(
        channels=int(model_cfg["channels"]),
        n_blocks=int(model_cfg["residual_blocks"]),
        value_hidden=int(model_cfg["value_hidden"]),
        norm_type=str(model_cfg["norm_type"]),
        use_se=bool(model_cfg["use_se"]),
    ).to(device)
    parameters = count_parameters(raw_model)
    if parameters != int(model_cfg["parameters"]):
        raise ValueError(f"parameter count {parameters} != configured {model_cfg['parameters']}")
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=tuple(float(x) for x in train_cfg["adam_betas"]),
        eps=float(train_cfg["adam_epsilon"]),
    )
    scheduler = cosine_with_warmup(optimizer, int(train_cfg["warmup_steps"]), max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    step = 0
    epoch = 0
    batch_offset = 0
    source_champion = args.source_champion
    if source_champion:
        payload = torch.load(source_champion, map_location=device, weights_only=False)
        raw_model.load_state_dict(payload["model_state_dict"])
        if context.is_primary:
            print(f"initialized_from_champion={source_champion}")
    elif context.is_primary:
        print("initialized_from_random_weights")

    model: torch.nn.Module
    if context.enabled:
        model = DistributedDataParallel(
            raw_model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            bucket_cap_mb=ddp_bucket_cap_mb,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    else:
        model = raw_model

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    eval_loader = make_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        seed=seed + 77_000,
        epoch=0,
        shuffle=False,
        rank=context.rank,
        world_size=context.world_size,
        persistent_workers=True,
    )
    train_started = time.time()
    optimization_seconds = 0.0
    last_metrics: dict[str, float] = {}
    stop_reason = "max_steps"
    next_epoch = epoch
    next_batch_offset = batch_offset
    while step < max_steps:
        loader = make_loader(
            dataset,
            batch_size=batch_size,
            workers=workers,
            seed=seed,
            epoch=epoch,
            shuffle=True,
            rank=context.rank,
            world_size=context.world_size,
        )
        for batch_index, batch in enumerate(loader):
            if batch_index < batch_offset:
                continue
            if step >= max_steps:
                break
            optimization_started = time.time()
            policy_mask, fallback = build_policy_mask(batch)
            policy = batch["mcts_policy"]
            if not bool(torch.isfinite(policy).all()) or not bool(torch.allclose(
                policy.sum(-1), torch.ones(policy.shape[0]), atol=1e-5
            )):
                raise ValueError("non-finite or non-normalized policy target in training batch")
            board = batch["board"].to(device, non_blocking=True)
            signed_komi = batch["signed_komi"].to(device, non_blocking=True)
            winner = batch["winner"].to(device, non_blocking=True)
            policy = policy.to(device, non_blocking=True)
            policy_mask = policy_mask.to(device, non_blocking=True)
            board, policy = augment_batch_dense(board, policy, int(config["rules"]["board_size"]))

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                loss, policy_loss, value_loss, _, _ = dense_loss(
                    model,
                    board,
                    signed_komi,
                    policy,
                    winner,
                    policy_mask,
                    float(train_cfg["policy_loss_weight"]),
                    float(train_cfg["value_loss_weight"]),
                    context,
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_cfg["gradient_clip_norm"])
            )
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            optimization_seconds += time.time() - optimization_started
            next_epoch = epoch
            next_batch_offset = batch_index + 1
            if next_batch_offset >= steps_per_epoch:
                next_epoch = epoch + 1
                next_batch_offset = 0

            if (
                step % int(train_cfg["progress_interval_steps"]) == 0
                and step < max_steps
            ):
                synchronize_running_stats(raw_model, context)
                last_metrics = evaluate(
                    model,
                    eval_loader,
                    device,
                    config,
                    context=context,
                    max_batches=20,
                )
                progress_values = torch.tensor(
                    [
                        float(loss.detach()),
                        float(policy_loss.detach()),
                        float(value_loss.detach()),
                        float(policy_mask.sum()),
                        float(fallback.sum()),
                        float(policy_mask.numel()),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                distributed_sum(progress_values, context)
                if context.is_primary:
                    denominator = max(1.0, float(progress_values[5]))
                    print(
                        f"step={step} loss={float(progress_values[0])/context.world_size:.5f} "
                        f"policy={float(progress_values[1])/context.world_size:.5f} "
                        f"value={float(progress_values[2])/context.world_size:.5f} "
                        f"grad_norm={float(grad_norm.detach()):.4f} "
                        f"teacher_fraction={float(progress_values[3])/denominator:.4f} "
                        f"bootstrap_fallback={int(progress_values[4])}/{int(progress_values[5])} "
                        f"policy_acc={last_metrics['policy_accuracy']:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.8f}",
                        flush=True,
                    )
                if (
                    step >= int(train_cfg["minimum_steps"])
                    and last_metrics["policy_accuracy"] >= float(train_cfg["target_policy_accuracy"])
                ):
                    stop_reason = "target_policy_accuracy"
                    break

            time_expired = broadcast_flag(
                (
                    step >= int(train_cfg["minimum_steps"])
                    and time.time() - train_started >= time_budget
                ),
                device=device,
                context=context,
            )
            if time_expired:
                stop_reason = "time_budget"
                break
        else:
            epoch += 1
            batch_offset = 0
            continue
        break

    synchronize_running_stats(raw_model, context)
    last_metrics = evaluate(
        model,
        eval_loader,
        device,
        config,
        context=context,
        max_batches=50,
    )
    states = gather_rng_states(device, context)
    local_peak_mib = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if use_cuda else 0.0
    )
    peak_vram_by_rank = gather_scalar(
        local_peak_mib,
        device=device,
        context=context,
    )
    optimization_seconds_by_rank = gather_scalar(
        optimization_seconds,
        device=device,
        context=context,
    )
    optimization_seconds = max(optimization_seconds_by_rank)
    elapsed_tensor = torch.tensor(
        [time.time() - train_started], dtype=torch.float64, device=device
    )
    distributed_max(elapsed_tensor, context)
    elapsed = float(elapsed_tensor.item())
    candidate = CHECKPOINT_ROOT / f"iter{args.iteration:04d}-candidate.pt"
    if context.is_primary:
        assert states is not None
        atomic_torch_save(
            checkpoint_payload(
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                iteration=args.iteration,
                step=step,
                epoch=next_epoch,
                batch_offset=next_batch_offset,
                manifest=manifest,
                source_champion=source_champion,
                config=config,
                rng_states=states,
                world_size=context.world_size,
            ),
            candidate,
        )
        optimizer_steps_this_run = step
        result = {
            "iteration": args.iteration,
            "checkpoint": str(candidate),
            "steps_completed": step,
            "optimizer_steps_this_run": optimizer_steps_this_run,
            "stop_reason": stop_reason,
            "elapsed_seconds": round(elapsed, 3),
            "training_samples_per_second": round(
                optimizer_steps_this_run * global_batch_size / max(elapsed, 1e-9), 3
            ),
            "optimization_seconds": round(optimization_seconds, 3),
            "optimization_samples_per_second": round(
                optimizer_steps_this_run
                * global_batch_size
                / max(optimization_seconds, 1e-9),
                3,
            ),
            "train_loss": round(last_metrics["loss"], 6),
            "train_policy_accuracy": round(last_metrics["policy_accuracy"], 6),
            "train_value_accuracy": round(last_metrics["value_accuracy"], 6),
            "peak_vram_mib": round(max(peak_vram_by_rank), 3),
            "peak_vram_mib_by_rank": [round(value, 3) for value in peak_vram_by_rank],
            "parameters": parameters,
            "world_size": context.world_size,
            "global_batch_size": global_batch_size,
            "per_rank_batch_size": batch_size,
            "ddp_bucket_cap_mb": ddp_bucket_cap_mb,
            "dataset_storage": dataset_storage,
            "dataset_cache_summary": cache_summary,
            "manifest_sha256": sha256_file(manifest),
        }
        atomic_write_json(
            CHECKPOINT_ROOT / f"iter{args.iteration:04d}-train-result.json",
            result,
        )
        print("===RESULT===")
        print(json.dumps(result, sort_keys=True))
    if context.enabled:
        dist.barrier()
    cleanup_distributed(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
