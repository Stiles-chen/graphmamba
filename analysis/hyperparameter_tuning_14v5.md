# 训练日志 "14 v5" 分析与调参建议

> **背景说明**：本文档基于 `model/criss_stgcn_v5.py`（DSTSA-HyperGraph 模型）在 **DHG14-28 数据集 14 类**任务上的训练配置（对应 `config/dhg14-28/DHG14-28.yaml`）进行系统性分析，并针对常见的训练问题给出具体的调参建议。

---

## 一、当前训练配置一览

| 超参数 | 当前值 | 说明 |
|--------|--------|------|
| `base_lr` | 0.1 | SGD 初始学习率 |
| `lr_decay_rate` | 0.1 | 学习率衰减倍率 |
| `step` | [90, 130] | 衰减触发 epoch |
| `num_epoch` | 150 | 总训练轮数 |
| `batch_size` | 32 | 批大小 |
| `weight_decay` | 0.0001 | L2 正则化系数 |
| `warm_up_epoch` | 20 | Warmup 轮数 |
| `nesterov` | True | Nesterov SGD |
| `repeat` | 5 | 数据重复倍数（扩充小数据集）|
| `window_size` | 150 | 输入序列帧数 |
| `random_choose` | True | 时间维随机采样增强 |
| `label_flag` | 14 | DHG14-28 14 类设置 |

**模型关键超参数（v5 架构）：**

| 模块 | 超参数 | 默认值 |
|------|--------|--------|
| 骨干网络 | `base_channel` | 64 |
| 骨干网络 | `inflate_stages` | [5, 8] |
| 骨干网络 | `down_stages` | [5, 8] |
| MS-TCN | `ms_cfg` | [(3,1),(3,2),(3,3),(3,4),("max",3),"1x1"] |
| HyperGraph | `dynamic_scale`（可学习） | 初始化自 1.0 |
| GCN 融合 | `gamma`（可学习） | 初始化自 0 |
| VIB | KL 散度损失权重 | 需在训练脚本中设置 |
| Dropout | `drop_out` | 0（默认未使用）|

---

## 二、训练日志常见问题诊断

根据同类 STGCN 在小型手势数据集（DHG14-28、SHREC17 等）上的训练规律，以下是"14 v5"训练过程中**最常见的现象及其根因分析**：

### 2.1 训练精度高但验证精度低（过拟合）

**典型表现**：
- Train Acc 在 epoch 80 后持续逼近 99%
- Val Acc 在 epoch 60-80 到达峰值后停止提升，甚至略有下降
- 训练集与验证集的 loss gap 持续扩大

**根因**：DHG14-28 训练集规模小（约 1,680 样本），而 v5 模型参数量较大（HyperGraph + CoordAtt + MS-TCN + VIB），正则化力度不足时极易过拟合。

---

### 2.2 前期精度爬升缓慢

**典型表现**：
- Warmup（epoch 1-20）结束后，Val Acc 仍低于 70%
- epoch 90 LR 第一次衰减前，精度增长停滞

**根因**：初始 lr=0.1 对于小数据集偏高，Warmup 线性增长阶段可能导致早期训练震荡；或 HyperGraph 动态部分的 `dynamic_scale` 过早主导拓扑，掩盖了骨骼先验信息。

---

### 2.3 LR 衰减后精度跳升但不持久

**典型表现**：
- epoch 90 / 130 衰减后精度各有 2-5% 的跳升
- 但收益在后续 epoch 被迅速消化，最终 Val Acc 稳定但不理想

**根因**：衰减步长 [90, 130] 均匀分布，未与验证集的实际收敛进度挂钩；总 epoch 150 设置下剩余的 20 个 epoch（epoch 130-150）不足以充分利用低学习率阶段。

---

### 2.4 VIB KL 散度损失不稳定

**典型表现**：
- 总损失的波动幅度大于普通 CE-Loss 模型
- 训练中后期 KL 损失比例失调，分类 CE-Loss 相对被压制

**根因**：VIB 的 KL 权重（`beta`）未随训练进程动态调整，固定权重在训练初期可能过大，干扰分类主任务的学习。

---

## 三、具体调参建议

### 3.1 学习率调度优化

**问题**：当前固定步长衰减策略（MultiStep LR）在小数据集上容易错过最佳衰减时机。

**建议**：

```yaml
# 方案 A：调整衰减步长，保留更多低学习率训练时间
base_lr: 0.1
step: [80, 110]        # 提前衰减步（原值 [90, 130]）
num_epoch: 140         # 适当缩短（原值 150），减少过拟合风险
warm_up_epoch: 15      # 从 20 减少至 15

# 方案 B：使用余弦退火（更平滑的衰减曲线）
# 在训练脚本中将 scheduler 替换为 CosineAnnealingLR
# T_max = 140, eta_min = 1e-5
```

**原理**：更早的第一次衰减（epoch 80 vs 90）让模型有更多时间在 lr=0.01 阶段精细调整；余弦退火则避免了阶梯式衰减的突变，对小数据集更友好。

---

### 3.2 正则化强化（优先级：高）

**建议 1：开启 Dropout**

v5 默认 `drop_out=0`，在小数据集上强烈建议开启：

```python
# model_args 中添加
drop_out: 0.3   # 建议范围 0.2-0.5，从 0.3 开始调试
```

**建议 2：增大 weight_decay**

```yaml
weight_decay: 0.0004   # 从 0.0001 提升至 0.0004（参考 NTU-RGB+D 配置）
```

**建议 3：VIB KL 权重退火**

在训练脚本中添加 KL 权重的线性/余弦预热（Capacity Annealing），避免初期 KL 过大干扰分类学习：

```python
# 示例：线性预热 KL 权重
beta_max = 0.001   # 最大 KL 权重（通常远小于 CE-Loss 的量级）
beta = beta_max * min(1.0, epoch / 50)
loss = ce_loss + beta * kl_loss
```

推荐的 `beta_max` 搜索范围：`{0.0001, 0.0005, 0.001, 0.005}`

---

### 3.3 数据增强强化

**当前状态**：仅使用了 `random_choose=True`（随机时间采样），`random_shift` 和 `random_move` 均为 False。

**建议**：

```yaml
train_feeder_args:
  random_choose: True    # 保持
  random_shift: True     # 开启：随机时间平移，增强时间不变性
  random_move: True      # 开启：随机关节位移抖动，增强空间鲁棒性
  # 如果 feeder 支持以下参数，也建议开启：
  # random_rot: True     # 随机旋转，增强视角鲁棒性
  # flip: True           # 左右镜像翻转（需确认 DHG14 数据格式支持）
```

**建议**：将 `repeat` 从 5 调整为 8-10，进一步扩充小数据集的训练样本多样性。

---

### 3.4 超图相关参数调整

**`dynamic_scale`（超图动态缩放因子）**

此参数控制动态超图修正项 ΔH 相对于静态超图 H_static 的权重。训练初期建议从较小值开始，避免动态部分主导静态先验：

```python
# 初始化时可设置较小值，让网络先学习基础骨架拓扑
dynamic_scale = nn.Parameter(torch.tensor(0.1))  # 从 0.1 开始，不要初始化为 1.0
```

**`gamma`（超图融合权重）**

控制 `A_hyper` 叠加到骨架图 `A` 的强度：

```python
# 可以给 gamma 加上上界约束，避免超图完全压过骨架图
gamma = torch.clamp(self.gamma, 0, 1.0)
```

---

### 3.5 模型容量调整

如果过拟合严重（Train-Val gap > 15%），考虑适度缩小模型：

```yaml
model_args:
  base_channel: 48      # 从 64 减小至 48（参数量减约 43%）
  # 或保持 64 但减少超图超边数量 num_edges
```

如果欠拟合（Val Acc 在低水平停滞），则适当扩展：

```yaml
model_args:
  base_channel: 64      # 保持
  # 增加 ms_cfg 的分支，如添加 dilation=5/6 的分支
  ms_cfg: [(3,1),(3,2),(3,3),(3,4),(3,5),"max3","1x1"]
```

---

### 3.6 批大小与优化器

**批大小**：DHG14-28 数据集小，batch_size=32 合理，但可尝试：
- `batch_size: 16`：梯度噪声更大，可起到隐式正则化作用（配合更大 weight_decay）
- `batch_size: 64`：需相应增大 lr（按线性缩放规则：`base_lr = 0.1 * (64/32) = 0.2`）

**优化器**：当前使用 Nesterov SGD，也可尝试 AdamW：

```yaml
optimizer: AdamW      # 替换 SGD
base_lr: 0.001        # AdamW 的典型初始 lr
weight_decay: 0.01    # AdamW 通常需要更大的 weight_decay
```

> ⚠️ 注意：从 SGD 切换到 AdamW 可能需要重新调整所有学习率相关参数。

---

## 四、推荐调参优先级与实验顺序

按照**投入产出比**，建议按以下顺序实验：

| 优先级 | 调整项 | 预期收益 | 风险 |
|--------|--------|----------|------|
| ★★★★★ | 开启 `drop_out: 0.3` | 缓解过拟合，+2-5% Val Acc | 低 |
| ★★★★★ | `weight_decay: 0.0004` | 缓解过拟合 | 低 |
| ★★★★☆ | VIB KL 权重退火 | 训练更稳定 | 中（需改代码）|
| ★★★★☆ | `step: [80, 110]`，`num_epoch: 140` | 更充分利用低 LR | 低 |
| ★★★☆☆ | 开启 `random_shift + random_move` | 增强泛化 | 低 |
| ★★★☆☆ | `dynamic_scale` 初始值设为 0.1 | 稳定超图训练 | 低 |
| ★★☆☆☆ | `repeat: 8` | 更丰富的样本多样性 | 低 |
| ★★☆☆☆ | 余弦退火 LR | 更平滑的收敛 | 中 |
| ★☆☆☆☆ | 切换 AdamW | 可能改善也可能变差 | 高（需全面调参）|

---

## 五、快速验证方案

建议用以下对照实验快速验证最关键的调参效果：

```bash
# Baseline（当前配置）
python main.py --config config/dhg14-28/DHG14-28.yaml

# Exp-1：正则化增强
python main.py --config config/dhg14-28/DHG14-28.yaml \
  --model_args.drop_out 0.3 \
  --weight_decay 0.0004

# Exp-2：LR 调度优化
python main.py --config config/dhg14-28/DHG14-28.yaml \
  --step '[80, 110]' --num_epoch 140 --warm_up_epoch 15

# Exp-3：数据增强
python main.py --config config/dhg14-28/DHG14-28.yaml \
  --train_feeder_args.random_shift True \
  --train_feeder_args.random_move True

# Exp-4：组合最优策略
python main.py --config config/dhg14-28/DHG14-28.yaml \
  --model_args.drop_out 0.3 --weight_decay 0.0004 \
  --step '[80, 110]' --num_epoch 140 \
  --train_feeder_args.random_shift True \
  --train_feeder_args.random_move True
```

每个实验重复 3 次（不同随机种子），取平均 Val Acc 作为最终评估指标。

---

## 六、参考基准

| 数据集 | 模型 | Val Acc（14 类）| 关键超参数 |
|--------|------|----------------|-----------|
| DHG14-28 | TD-GCN（Baseline）| ~93-94% | lr=0.1, wd=0.0001, ep=150 |
| DHG14-28 | DSTSA-HyperGraph v5 | 目标 >95% | 参见本文档建议 |

> 📝 如需进一步分析具体的训练曲线（loss/acc 随 epoch 的变化），请将实际的 `log.txt` 文件添加到 `analysis/` 目录，便于进行定量分析和针对性调参。
