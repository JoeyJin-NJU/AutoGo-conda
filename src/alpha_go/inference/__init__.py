"""Inference engines for batched neural network evaluation.

This module provides inference engines for efficient GPU utilization:
- LocalBatchedInferenceEngine: Local batching for collectors with GPU access
- gRPC mode via NNAgent: Remote batching via inference server

Usage:
    from alpha_go.inference import LocalBatchedInferenceEngine

    engine = LocalBatchedInferenceEngine(model, device, batch_size=64)
    engine.start()

    # Submit from multiple threads
    future = engine.submit(board_np)
    policy, value, entropy = future.result()

    engine.stop()
"""

from alpha_go.inference.batched_engine import LocalBatchedInferenceEngine

__all__ = ["LocalBatchedInferenceEngine"]
