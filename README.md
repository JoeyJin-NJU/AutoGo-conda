# AutoGo-conda

> **Origin:** `AutoGo-conda` is based on and extends Eric Jang's MIT-licensed
> [AutoGo](https://github.com/ericjang/autogo). For the project's motivation,
> AlphaGo/MCTS, self-play, and their connections to LLM reinforcement learning and
> automated AI research, see Eric Jang's conversation with Dwarkesh Patel,
> [*Building AlphaGo from scratch*](https://www.dwarkesh.com/p/eric-jang).

A Conda-ready Go AI training framework. This repository preserves AutoGo's Python/C++
core, a complete `Self-play -> Train -> Arena -> Promote/Reject` experiment controller,
a trained 9x9 champion, and a browser interface for playing against the model while
visualizing its MCTS search.

![AutoGo checkpoint play interface](docs/images/checkpoint-play.png)

## Repository contents

- `src/alpha_go/`: Go rules, neural networks, MCTS, self-play, and inference code.
- `src/alpha_go/cpp/`: C++17 Go rules and batched MCTS, exposed to Python with pybind11.
- `experiments/2026-07-31_11-39-local-9x9-k7-pcr-001/`: 9x9 PCR self-play, DDP training,
  Arena promotion, and status-monitoring code.
- `local_data/checkpoints/.../iter1002-candidate.pt`: the bundled promoted champion.
- `checkpoint_play/`: human-vs-AI play, continuous MCTS visualization, and two Arena
  game replays.
- `tests/`: core Python/C++ tests for rules, MCTS, models, and data handling.

The original experiment directory is approximately 15 GB. This repository includes only
the reproducible code, final configuration, champion, and two small NPZ game records used
by the demo. Training caches, historical candidates, logs, manifests, and obsolete runtime
state are not included.

## Install with Conda

Requirements: Linux, Conda, an NVIDIA driver compatible with PyTorch, and sufficient disk
space. For automated setup:

```bash
git clone https://github.com/JoeyJin-NJU/AutoGo-conda.git
cd AutoGo-conda
bash scripts/setup_conda.sh
conda activate autogo-conda
```

Alternatively, run the setup steps manually:

```bash
conda env create -f environment.yml
conda activate autogo-conda
python -m pip install --editable '.[dev]'
bash scripts/build_cpp.sh
python scripts/verify_install.py
```

`build_cpp.sh` compiles `alpha_go_cpp` with Python, CMake, and Ninja from the active Conda
environment. It does not depend on `uv`, Docker, or any fixed server path.

## Verify the installation

```bash
python scripts/verify_install.py
pytest -q tests
pytest -q experiments/2026-07-31_11-39-local-9x9-k7-pcr-001/test_experiment.py \
  experiments/2026-07-31_11-39-local-9x9-k7-pcr-001/test_packed_dataset.py
```

## Launch the play interface

```bash
conda activate autogo-conda
CUDA_VISIBLE_DEVICES=0 python checkpoint_play/app.py --host 127.0.0.1 --port 8000
```

Open one of the following URLs in a browser:

- `http://127.0.0.1:8000/`: full human-vs-AI interface with continuous MCTS analysis.
- `http://127.0.0.1:8000/ppt`: a presentation-friendly 16:9 interface.
- `http://127.0.0.1:8000/replay`: two Arena game replays from iter152 against KataGo.

Moves played by the model use 600 MCTS simulations. Continuous analysis keeps accumulating
visits in the same search tree until it is stopped manually or the position changes.

## Run the training framework

The packaged experiment retains physical GPU IDs `2,3,4,5` from the original training
configuration. Before running it, update `collection.gpu_ids`, `training.gpu_ids`, and
`arena.gpu_ids` in `config.json` to match your machine. The global batch size must be
divisible by the number of training GPUs.

```bash
EXP=experiments/2026-07-31_11-39-local-9x9-k7-pcr-001

# Run in the foreground. The first launch verifies and freezes the bundled iter1002
# champion, then starts a new lineage at iteration 1.
bash "$EXP/launcher.sh"

# Inspect status
python "$EXP/status.py"

# Stop gracefully after the current atomic stage; no new work will be scheduled.
bash "$EXP/stop.sh"

# Resume the same run
bash "$EXP/launcher.sh" --resume
```

This release starts a new training lineage warm-started from `iter1002`. It does not include
the 15 GB of accumulated self-play data required to resume iteration 1003 from the original
server. New data, checkpoints, logs, and runtime state are written under `local_data/` and
the experiment directory and are excluded by `.gitignore`.

## Champion

| Field | Value |
|---|---|
| Checkpoint | `iter1002-candidate.pt` |
| Parameters | 2,967,683 |
| Board / komi | 9x9 / 7.5 |
| Promoted at | 2026-08-13 06:56:05 UTC |
| Arena score | 0.5154639175 |
| SHA-256 | `3bd995e19240b7d2e6511cbdba5a94b381c01a910ed974e97011704541b9467f` |

The file is approximately 35.8 MB, below GitHub's 100 MB per-file limit, so Git LFS is not
required.

## Origins, conversation, and license

AutoGo's original code and research concept are by Eric Jang:

- Upstream repository: [Eric Jang / AutoGo](https://github.com/ericjang/autogo)
- Interactive tutorial: [AutoGo: a Tutorial](https://evjang.com/2026/04/28/autogo.html)
- Eric Jang's conversation with Dwarkesh Patel:
  [*Building AlphaGo from scratch*](https://www.dwarkesh.com/p/eric-jang)

The conversation starts from rebuilding AlphaGo with modern AI tools and covers MCTS,
self-play, credit assignment in reinforcement learning, and which parts of AI research can
already be automated with LLMs. `AutoGo-conda` adds a Conda installation path, a
single-machine training experiment, a 9x9 champion, and a browser demo. It is not an
official release of Eric Jang's upstream project.

This project is released under the MIT License. See [`NOTICE.md`](NOTICE.md) for the exact
core-source snapshot and model provenance.
