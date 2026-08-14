# AutoGo 人机对弈与棋谱回放

固定规则：9x9、贴目 7.5、最多 162 手、双方连续 PASS 终局。模型正式落子使用
Arena 的 600 次 MCTS simulations 并采用确定性选择。默认加载仓库内的正式 champion：

`local_data/checkpoints/2026-07-31_11-39-local-9x9-k7-pcr-001/iter1002-candidate.pt`

## 启动

从仓库根目录执行：

```bash
conda activate autogo-conda
CUDA_VISIBLE_DEVICES=0 python checkpoint_play/app.py --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/`。页面默认选中 iter1002 champion，可以选择执黑或执白。

## 页面

- `/`：完整人机对弈和持续 MCTS 分析台。
- `/ppt`：无滚动的 16:9 横屏演示版。
- `/replay`：仓库内置的两局 iter152 对 KataGo Arena 回放。

回放页面使用两个随仓库发布的 NPZ 文件，并在服务启动后按记录逐手验证棋盘：

- iter152 执黑第 1 局：84 手，`W+30.5`，iter152 负。
- iter152 执白第 14 局：67 手，`W+0.5`，iter152 胜。

## 持续 MCTS 分析

- 点击“启动 MCTS”后不设 visits 上限，并持续复用同一棵搜索树。
- 棋盘最多叠加前 10 个非 PASS 候选点；右侧显示 visits、搜索占比和当前方胜率。
- 点击“停止”、按 `Esc`、棋盘变化或服务退出都会停止分析。
- `Space` 可以开始或停止持续分析。

停止服务时在启动它的终端按 `Ctrl-C`。
