# 9x9 K7 PCR reference experiment

这是 `AutoGo-conda` 随仓库发布的完整本地训练控制器。它实现同步循环：

```text
COLLECTING -> BUILDING_MANIFEST -> TRAINING -> ARENA -> PROMOTED / REJECTED
```

## 发布快照

- 规则：9x9、贴目 7.5、Tromp-Taylor area scoring、positional superko、连续两次 PASS 终局。
- 模型：`SignedKomiGoResNet`，128 channels、10 residual blocks、2,967,683 参数。
- 自对弈：PCR 以 95% 概率使用 1,024 simulations，以 5% 概率使用 2,048 simulations。
- 训练：AdamW、BF16、全局 batch 768，支持单机多 GPU DDP。
- Arena：候选与 champion 对局最多 200 局，每种颜色 100 局，600 simulations，晋级阈值 50%。
- Warm start：随仓库提供的 `iter1002-candidate.pt` 正式 champion。

原运行的累积自对弈数据、历史 checkpoint、日志、manifest 和 4,900 行 runtime state
没有被打包。首次启动会验证 champion SHA-256，将它冻结为新运行的 iteration-0 champion，
然后从新的 iteration 1 开始收集数据；它不是对原 iteration 1003 的断点恢复。

## 文件职责

- `controller.py`：持久状态机、GPU preflight、进程调度、checkpoint 校验和晋级。
- `run_games.py`：PCR collection 与固定预算 Arena 对局。
- `train.py`：累积数据集上的 DDP 训练。
- `arena.py`：Arena 统计、健康门和不可逆提前终止。
- `packed_dataset.py`：内存映射的打包数据缓存。
- `color_model.py`：带当前行棋方和 signed-komi 输入的模型。
- `common.py`：路径、原子写入、NPZ 校验和 GPU 状态检查。
- `status.py`、`stop.sh`：只读监控与安全停止。

## GPU 配置

`config.json` 保留了原服务器的物理 GPU `2,3,4,5`。迁移到其他机器时，应同时修改：

```json
"collection": { "gpu_ids": [0] },
"training":   { "gpu_ids": [0] },
"arena":      { "gpu_ids": [0] }
```

全局 `training.batch_size` 必须能被训练 GPU 数整除。单 GPU 显存较小时，还应下调
`batch_size`、`processes_per_gpu` 和 `dataloader_workers`。

## 操作

从仓库根目录执行：

```bash
EXP=experiments/2026-07-31_11-39-local-9x9-k7-pcr-001

bash "$EXP/launcher.sh"
python "$EXP/status.py"
bash "$EXP/stop.sh"
bash "$EXP/launcher.sh" --resume
```

数据默认写入 `local_data/game_data/experiments/<experiment-name>/`，checkpoint 写入
`local_data/checkpoints/<experiment-name>/`。可以分别用 `GAME_DATA_DIR` 和
`AUTOGO_CHECKPOINT_DIR` 环境变量覆盖。

## 测试

```bash
pytest -q \
  experiments/2026-07-31_11-39-local-9x9-k7-pcr-001/test_experiment.py \
  experiments/2026-07-31_11-39-local-9x9-k7-pcr-001/test_packed_dataset.py
```
