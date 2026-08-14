"""Local batched inference engine for efficient GPU utilization.

This module provides a batched inference engine that runs locally on the collector,
eliminating gRPC overhead while maintaining high throughput through batching.

Performance profiling is integrated using infra/profiling.py to track:
- Inference time (GPU forward pass)
- Queue wait time
- Batch sizes
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F



@dataclass
class InferenceRequest:
    """Request for batched inference."""
    board_np: np.ndarray  # (H, W) float32 — H == W == native_size
    native_size: int      # H/W of the *unpadded* board, also the slice
                          # boundary the engine uses to trim policy logits
                          # before returning the result.
    result_future: Future
    submit_time_ns: int  # Time when request was submitted


class LocalBatchedInferenceEngine:
    """Engine that batches inference requests and processes them locally on GPU.

    This is the high-performance alternative to gRPC-based inference. All worker
    threads share a single queue, and a dedicated worker thread processes batches
    on the GPU.

    Variable-size inputs: each `submit(board_np)` call may carry a board at any
    `H == W <= max_board_size`. The batching loop zero-pads every request to
    `(max_board_size, max_board_size)` and feeds a per-row mask into the
    SizeInvariantGoResNet, then slices each row's policy logits back to
    `(native_H * native_W + 1)` so the caller sees the same shape it would
    have on a size-pinned engine. This lets one fleet of workers transparently
    serve 9x9 and 19x19 requests in the same batch.

    Args:
        model: PyTorch model for inference. Must have
            `forward(board_BHW, mask_BHW) -> (policy_BC, value_B)` if
            `board_size > min_request_size` is expected (i.e. mixed-size
            batches), or the legacy `forward(board_BHW) -> (...)` for
            homogeneous batches at the pinned size.
        device: Torch device for inference
        batch_size: Maximum batch size (default 64)
        batch_timeout_ms: Grace period to collect batch after first request (default 1.0ms)
        board_size: Padded H/W the GPU sees per forward (default 9) — i.e.
            the maximum native size any submitted request may carry. Every
            request smaller than this is zero-padded + masked. The legacy
            kwarg name is kept for backward compat with existing callers
            that wired this in as the model's pinned size.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        batch_size: int = 64,
        batch_timeout_ms: float = 1.0,
        board_size: int = 9,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0  # Convert to seconds
        # Naming gymnastics: external callers (and the legacy single-size
        # callers in `experiments/`) still pass `board_size=`, but the value
        # is now semantically "the largest native size we'll ever pad to".
        self.max_board_size = board_size
        self.board_size = board_size  # legacy alias for any reader

        self.request_queue: queue.Queue[InferenceRequest] = queue.Queue()
        self.running = False
        self.worker_thread: threading.Thread | None = None

        # Stats (thread-safe via atomic operations)
        self.total_requests = 0
        self.total_batches = 0
        self.batch_sizes: list[int] = []
        self._stats_lock = threading.Lock()

    def start(self) -> None:
        """Start the batching worker thread."""
        if self.running:
            return
        self.running = True
        self.model.eval()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        """Stop the worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)

    def submit(self, board_np: np.ndarray) -> Future:
        """Submit a board for evaluation. Returns a Future for the result.

        Args:
            board_np: Board state as numpy array (H, W) with values 0, 1, 2.
                H == W and must be ≤ `max_board_size`. The batching loop
                zero-pads to `max_board_size` and masks the excess.

        Returns:
            Future that resolves to (policy_C, value, entropy) where:
            - policy_C: numpy array of policy logits at the *native* size,
              shape `(H*W + 1,)` (last entry is the pass logit) — same
              contract as a size-pinned engine.
            - value: float value estimate
            - entropy: float policy entropy at native size (after slicing).
        """
        if board_np.ndim != 2 or board_np.shape[0] != board_np.shape[1]:
            raise ValueError(
                f"engine.submit expects a square (H,W) board; got shape {board_np.shape}"
            )
        native_size = int(board_np.shape[0])
        if native_size > self.max_board_size:
            raise ValueError(
                f"native_size={native_size} exceeds engine max_board_size={self.max_board_size}"
            )
        submit_time = time.perf_counter_ns()
        future: Future = Future()
        request = InferenceRequest(
            board_np=board_np.astype(np.float32),
            native_size=native_size,
            result_future=future,
            submit_time_ns=submit_time,
        )
        self.request_queue.put(request)
        return future

    def get_stats(self) -> dict:
        """Get current statistics."""
        with self._stats_lock:
            return {
                "total_requests": self.total_requests,
                "total_batches": self.total_batches,
                "avg_batch_size": np.mean(self.batch_sizes) if self.batch_sizes else 0,
                "queue_size": self.request_queue.qsize(),
            }

    def pending_requests(self) -> int:
        """Current number of submitted requests waiting on the batching worker."""
        return self.request_queue.qsize()

    def _worker_loop(self) -> None:
        """Worker loop with adaptive batching for low-latency single requests."""
        while self.running:
            batch: list[InferenceRequest] = []

            # Step 1: Block until first request (1s timeout for shutdown check)
            try:
                request = self.request_queue.get(timeout=1.0)
                batch.append(request)
            except queue.Empty:
                continue

            # Step 2: Quick drain with grace period to catch request bursts
            drain_deadline = time.perf_counter() + self.batch_timeout
            while len(batch) < self.batch_size and time.perf_counter() < drain_deadline:
                try:
                    remaining = max(0, drain_deadline - time.perf_counter())
                    request = self.request_queue.get(timeout=remaining)
                    batch.append(request)
                except queue.Empty:
                    break

            if not batch:
                continue

            # Process batch
            try:
                self._process_batch(batch)
            except Exception as e:
                for request in batch:
                    request.result_future.set_exception(e)

            # Update stats
            with self._stats_lock:
                self.batch_sizes.append(len(batch))
                self.total_requests += len(batch)
                self.total_batches += 1

    @torch.no_grad()
    def _process_batch(self, batch: list[InferenceRequest]) -> None:
        """Process a batch of inference requests.

        Profiling metrics recorded:
        - local_inference_forward_ns: GPU forward pass time
        - local_inference_queue_wait_ns: Time from submit to batch start

        Variable-size handling: each request brings its native (H, W). The
        batch tensor is allocated at `(B, max, max)` and each row is filled
        in-place at `[:n_i, :n_i]` with the request's board, leaving the
        excess as zeros. The mask tensor mirrors that — 1 inside the
        native window, 0 outside. The model uses the mask to zero out
        excess channel-0 ('empty') so untouched padding doesn't leak into
        neighbor convolutions, then emits a `(B, max*max + 1)` policy. We
        slice each row back to the native `(n_i*n_i + 1,)` layout before
        returning so callers see the same shape they would on a size-
        pinned engine. Mixed-size batches preserve cross-game GPU
        utilization at the cost of `(max/n_i)^2` wasted FLOPs per small
        request — a 9-on-19 sample is ~4.5x larger than necessary, which
        is fine in exchange for not running two separate engines.
        """
        B = len(batch)
        max_n = self.max_board_size
        boards_np = np.zeros((B, max_n, max_n), dtype=np.float32)
        mask_np = np.zeros((B, max_n, max_n), dtype=np.float32)
        for i, r in enumerate(batch):
            n = r.native_size
            boards_np[i, :n, :n] = r.board_np
            mask_np[i, :n, :n] = 1.0
        # All rows full size (no padding) → engine can short-circuit the
        # mask and call the legacy `forward(board)` signature, which is
        # what the SizeInvariantGoResNet.forward also does internally
        # when mask is None.
        homogeneous = all(r.native_size == max_n for r in batch)
        board_tensor = torch.from_numpy(boards_np).to(self.device)
        mask_tensor = (
            None if homogeneous
            else torch.from_numpy(mask_np).to(self.device)
        )

        if mask_tensor is None:
            policy_logits_BC, value_logits_B = self.model(board_tensor)
        else:
            policy_logits_BC, value_logits_B = self.model(board_tensor, mask_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Move to CPU once; per-sample slicing happens in numpy below.
        policy_np = policy_logits_BC.float().cpu().numpy()  # (B, max*max + 1)
        value_np = torch.sigmoid(value_logits_B).float().cpu().numpy()
        pass_idx_padded = max_n * max_n

        # Distribute results, slicing the policy of each row back to its
        # native (n_i * n_i + 1,) layout. SizeInvariantGoResNet flattens
        # positions as `r * max_n + c` (row-major over the padded grid),
        # so the in-window indices for native size `n` are
        # `[r*max_n + c for r,c in 0..n)` — gathered via reshape + slice.
        for i, request in enumerate(batch):
            n = request.native_size
            if n == max_n:
                native_policy = policy_np[i]
            else:
                grid = policy_np[i, :pass_idx_padded].reshape(max_n, max_n)
                in_window = grid[:n, :n].reshape(-1)
                pass_logit = policy_np[i, pass_idx_padded:pass_idx_padded + 1]
                native_policy = np.concatenate([in_window, pass_logit], axis=0)

            # Recompute entropy on the *native* policy slice — the padded
            # logits are -inf inside the model, so a softmax over the full
            # (max*max + 1) row would already exclude them, but we want the
            # entropy figure to match the native action set the caller
            # actually sees.
            shifted = native_policy - native_policy.max()
            ex = np.exp(shifted)
            probs = ex / ex.sum()
            ent = float(-(probs * np.log(probs + 1e-8)).sum())

            request.result_future.set_result(
                (native_policy, float(value_np[i]), ent),
            )
