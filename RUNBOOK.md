# 运行手册：Elliptic 严格时间因果 AML 可复现管线

> 本文件是运行手册；原 `README.md` 保留为论文 Appendix 伪代码素材，不覆盖。

## 1. 环境

本机指定解释器：

```bash
C:/Users/apple/anaconda3/python.exe
```

已核验依赖：Python 3.11.5、torch 2.2.1、torch_geometric 2.8.1、numpy、pandas、scikit-learn、scipy、pytest、xgboost、lightgbm 均可用；CUDA 可用，默认 `--device auto`。

## 2. 数据：自动下载与断点续传

当前 `data/raw/` 已包含真实 Elliptic 三个 CSV 与 `SHA256SUMS`；后续命令会直接跳过下载。下载脚本仍支持在新机器上自动获取与续传。

### 自动方式（推荐）

直接运行任意实验命令并加 `--data-mode auto`（默认）。脚本按以下顺序处理：

1. `data/raw/` 中三个 CSV 都已存在 → 跳过下载，写入 `SHA256SUMS`；
2. 缺少任一 CSV → 从 Kaggle 官方公开接口下载 ZIP，支持 HTTP Range 断点续传（`.part` 文件），解压并抽取三个 CSV；
3. 下载失败且加 `--allow-toy` → 退回 toy 数据，并在结果中标记 `dataset_version: toy`。

下载源：

```text
https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic-data-set
```

### 手动方式（网络受限时）

```bash
kaggle datasets download -d ellipticco/elliptic-data-set
unzip elliptic-data-set.zip -d data/raw
```

最终 `data/raw/` 需包含：

```text
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
```

数据许可为 **CC BY-NC-ND 4.0**：仓库只发布代码、SHA256 和转换脚本，不重分发数据本体；`LICENSE` 只覆盖代码。

## 3. 运行实验

所有命令均使用指定 Python。首次运行会自动下载数据；若想先做冒烟，加 `--data-mode toy`。

### 主模型（5 种子）

```bash
for s in 0 1 2 3 4; do
  C:/Users/apple/anaconda3/python.exe scripts/run_pipeline.py \
    --data-mode auto --model causal_sage --seed $s \
    --epochs 30 --batch-size 256 --neighbors 10 10 \
    --alpha 0.5 --degree-weight --calibration temperature
done
```

### 树基线（同切分、同特征）

```bash
for m in lr rf xgboost lightgbm mlp; do
  for s in 0 1 2 3 4; do
    C:/Users/apple/anaconda3/python.exe scripts/run_pipeline.py \
      --data-mode auto --model $m --seed $s --calibration temperature
  done
done
```

### 结构反事实

```bash
for w in real no_time empty perm; do
  C:/Users/apple/anaconda3/python.exe scripts/run_pipeline.py \
    --data-mode auto --model causal_sage --seed 0 --epochs 10 \
    --edge-world $w --calibration none
done
```

### 算法消融（单向变量）

```bash
# A0：无度校正、无时间编码、均匀采样
C:/Users/apple/anaconda3/python.exe scripts/run_pipeline.py --model graphsage --alpha 0 --no-time-encoding --calibration none

# A1：仅时间因果采样（边世界 real 已内置）
C:/Users/apple/anaconda3/python.exe scripts/run_pipeline.py --model causal_sage --alpha 0 --no-time-encoding --calibration none

# A2：仅度校正
C:/Users/apple/anaconda3/python.exe scripts/run_pipeline.py --model causal_sage --alpha 0 --degree-weight --no-time-encoding --calibration none

# A3：因果采样 + 度校正 + 时间编码
C:/Users/apple/anaconda3/python.exe scripts/run_pipeline.py --model causal_sage --alpha 0.5 --degree-weight --calibration temperature
```

### 标签协议

```bash
for p in A B C D; do
  C:/Users/apple/anaconda3/python.exe scripts/run_pipeline.py \
    --model causal_sage --seed 0 --label-protocol $p --calibration none
done
```

## 4. 断点续跑

- 每个 `(config, seed)` 有稳定 `run_id`，结果写入 `outputs/results/<run_id>.json`。
- 若结果已存在，重跑同一命令直接 `[SKIP]`，不会重算。
- 训练中每轮保存 `outputs/checkpoints/<run_id>.pt`，包含 model / optimizer / epoch / best_state / numpy RNG 状态。
- 若脚本被关闭，重跑**完全相同的命令**（不要加 `--no-resume`）即可从下一轮继续；日志打印 `[RESUME] ... from epoch N`。
- 加 `--keep-checkpoint` 可在完成后保留最后检查点，便于审计或续跑测试。

已实测：toy 上人为中断后重跑，日志显示 `[RESUME] a66c4b2b8f7c from epoch 5`，并成功产出结果。

## 5. 测试

```bash
C:/Users/apple/anaconda3/python.exe -m pytest tests -q
```

本机结果：**9 passed**。覆盖未来边过滤、切分互斥、未知标签掩码、指标/ECE、采样器因果性、度校正零参数、数据跳过下载、检查点往返、端到端 toy 冒烟。

## 6. 输出工件

```text
outputs/
├── results/<run_id>.json        # 全指标、漂移、度子群、校准、边审计
├── results/leaderboard.json     # 跨 run 汇总
├── manifest/<run_id>.json       # 配置、种子、版本、命令
├── manifest/splits.json         # 显式时间步集合
└── checkpoints/<run_id>.pt      # 训练中检查点（完成后默认删除）
```

`results.json` 中 `dataset_version: toy` 时，**禁止**用于论文表格。


新增开关示例：

```bash
# 随机切分协议阶梯
--split-mode random

# 严格度保持重连（保持出度/入度序列与因果边约束）
--edge-world rewire

# 真 nnPU 风险（SCAR 敏感性）
--label-protocol D --pu-prior 0.0976

# 测试期攻击 + 鲁棒聚合
--attack edge_delete --attack-budget 0.1 --aggregator trim
--attack feature_flip --attack-budget 0.1 --aggregator median
```

> Elliptic 特征是匿名化的，公开资料没有给出“金额”对应的列号。因此论文不宣称金额子群公平性；只报告度子群与匿名局部特征索引的分位敏感性探索。
