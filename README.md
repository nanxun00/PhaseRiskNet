# PhaseRiskNet

本仓库为论文 **PhaseRiskNet: Phase-Aware Risk-Adaptive Seismic Phase Picking with Uncertainty-Guided Selective Prediction** 的配套代码。

## 环境

- 建议使用 **Python 3.10+**。
- 安装依赖：

```bash
pip install -r requirements.txt
```

GPU 训练需安装带 CUDA 的 PyTorch，请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择与本地 CUDA 版本匹配的命令安装（可覆盖/补充 `requirements.txt` 中的 `torch`）。

## 运行方式

在项目根目录（本 `README.md` 所在目录）下执行：

```bash
python phase_run.py --include-ablation
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--include-ablation` | 运行 `phase_run.py` 中配置的消融实验（如 `phasenet_full_big`、`phasenet_full_small`） |
| `--quick` | 快速试跑（更少 epoch，依赖 `phase_core` 中的快速逻辑） |
| `--gpu 0` | 仅使用指定 GPU（设置 `CUDA_VISIBLE_DEVICES`） |
| `--ablation-keys phasenet_full_big,phasenet_full_small` | 只跑列出的配置 key，逗号分隔 |
| `--skip-baseline` | 跳过 baseline（若未在配置中定义 `phasenet_baseline` 则会自动跳过） |
| `--seed` | 随机种子（默认与下方 `PHASENET_SEED` 一致） |

示例：

```bash
python phase_run.py --include-ablation --quick --gpu 0
```

## 配置与隐私（环境变量）

为避免在代码中硬编码本机路径与随机种子，请在运行前通过环境变量设置（或在 shell / `.env` 中导出）：

| 变量 | 说明 |
|------|------|
| `PHASENET_SEED` | 全局默认随机种子；未设置时默认为 `42`（`phase_core.SEED`） |
| `PHASENET_OUTPUT_DIR` | 训练输出与指标根目录；未设置时优先与 `CEED_CACHE_DIR` 同盘，否则为当前目录 |
| `CEED_CACHE_DIR` | Hugging Face 数据集缓存目录；为空则使用系统默认缓存位置 |
| `CEED_LOCAL_DIR` | CEED 本地 `.h5` 数据目录；为空则按 `datasets` 行为联网或缓存加载 |
| `H5_THREE_CHANNEL_ROOT` | 三通道 H5 数据根目录（当 `phase_core.py` 中 `DATA_SOURCE == "h5_three_channel"` 时必填） |

数据源类型 `DATA_ROOT`、`DATA_SOURCE` 等仍在 **`phase_core.py`** 中按需修改。
