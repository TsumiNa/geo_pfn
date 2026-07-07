# TabPFN 源码/论文分析：测试集特征缺失（missing test features）的可行性与方案

> 图文版实验报告（交互图表）：[`docs/minipfn-report.html`](minipfn-report.html)——克隆后用浏览器打开即可。

本文回答三个问题：

1. TabPFN 要求 `X_train` / `X_test` 特征一致，能否改造为允许 `X_test` 缺失部分特征？
2. 若可行，预训练需要多少数据？能否用原版 TabPFN 为我们的模型生成合成数据？
3. 端到端 PyTorch demo（见 `src/geo_pfn/minipfn/`）。

分析基于 `tab_pfn_src/`（vendored TabPFN 源码，含 v2 / v2.5 / v2.6 / v3 四套架构）以及四篇论文：
PFN（ICLR 2022, arXiv 2112.10510）、TabPFN v1（ICLR 2023, arXiv 2207.01848）、
TabPFN v2（Nature 2025, doi 10.1038/s41586-024-08328-6）、TabPFN-2.5（arXiv 2511.08667）、
TabPFN-3 技术报告（arXiv 2605.13986）。

---

## 1. TabPFN 核心机制（理解方案的前提）

**PFN 目标**：在海量*合成*数据集上预训练一个 transformer，使其学会「给定 (X_train, y_train, X_test)
一次前向就输出 y_test 的后验预测分布」。训练损失是合成任务 held-out 行上的交叉熵，其最优解在数学上
恰好等于先验下的贝叶斯后验预测分布（PFN 论文 Corollary 1.1）。因此：

- `fit()` **没有任何梯度训练**，只做预处理并缓存训练张量
  （`tab_pfn_src/src/tabpfn/classifier.py:791`；推理引擎缓存见 `inference.py:662`）。
- `predict()` 把 train/test 拼接成一个张量走一次前向：
  `X_full = torch.cat([X_train, X_test], dim=0)`（`inference.py:1172`）。
- 「换一组特征重新 fit」的代价是**零**（没有训练），这是后面方案 B 的基础。

**v2 家族架构**（Nature 版，也是本 demo 模仿的对象）：表格的**每个单元格是一个 token**
（`features_per_group=2/3` 个标量一组），网络每层交替做两个方向的注意力：

- 行内跨特征注意力（`AlongRowAttention`，`architectures/tabpfn_v2.py:119`）；
- 列内跨样本注意力（`AlongColumnAttention`，`tabpfn_v2.py:151`），其中 **K/V 只从 train 行投影**
  （`tabpfn_v2.py:221`），所以 test 行只能看 train 行、彼此独立——train/test 分离不靠显式 mask。

y 作为**额外的一列 token** 拼在特征列之后（`tabpfn_v2.py:898`），test 行的 y 用 NaN 占位；
预测从 test 行的 y 列 token 读出（`tabpfn_v2.py:777`）。特征身份靠**随机特征嵌入**
（固定种子的 randn 经学习的线性层投影，`tabpfn_v2.py:593`），因此特征数量可变、顺序近似无关。

**v3** 改为三段式（分布嵌入器 → 行内聚合为 per-row token → 跨行 ICL，53M 参数），
思想不变，只是把跨行注意力从 cell 级压缩到 row 级以支持 1M 行。

**NaN 是一等公民**：所有版本的编码器都执行
「生成 NaN/Inf 指示通道 → 用 *train 行均值* 填补 → 与指示通道拼接后线性投影」
（`tabpfn_v2.py:666`、`tabpfn_v2.py:698`，常量 `NAN_INDICATOR=-2.0`）。
更关键的是 **v2 的预训练先验中显式模拟了缺失**：每个单元格以 ρ_miss 概率被 MCAR 置为缺失
（Nature Methods）。也就是说「带指示通道的缺失输入」是模型见过的分布内输入。

---

## 2. 问题 1：X_test 可以有 missing features 吗？—— 可以，且有四个层级的方案

**约束到底在哪里**：特征数一致的检查发生在 sklearn 校验层——
`predict` 时 `validate_data(reset=False)` 对照 `n_features_in_` 报
`TabPFNValidationError`（`validation.py:102`、`errors.py:27`）。
**架构层面**的真实约束只有一条：train+test 在同一个 `[rows, batch, features]`
张量里共享特征轴（`architectures/interface.py:205`）。
所以「X_test 物理上少几列」不能直接输入，但可以用两种等价表达绕开，模型本身完全支持。

### 方案 A（零改造，立即可用）：缺失列填 NaN

把 X_test 缺的列补成全 NaN，凑齐列数后正常 `predict`。

- 预处理管线全程 NaN-aware（分位数变换 `adaptive_quantile_transformer.py:127`、
  SVD 步骤内部有 `SimpleImputer(keep_empty_features=True)`，`steps/utils.py:96`），
  全 NaN 测试列**不会报错**；
- 编码器用 train 均值填补并附加缺失指示，模型依据完整的 train 上下文对缺失特征做隐式边缘化；
- 官方 issue #108 中维护者演示过 X_test 含 NaN 块开箱即用。

局限：v2 先验只模拟了 **cell 级 MCAR**，而「整列在 test 中缺失」是结构化缺失，
分布上有偏移——模型能处理但不是为此优化的（这正是方案 C/D 的价值）。

### 方案 B（零改造）：从上下文中同步删列（drop-context）

因为 `fit()` 没有训练，可以在预测时把缺失列从 X_train 里也删掉，用剩余列重新 fit + predict。
一次 fit 只是预处理+缓存（秒级），代价可忽略。这相当于在缩减后的特征空间里重新做贝叶斯推断。

A 与 B 的取舍是实证问题：A 保留了完整 train 上下文（模型可利用特征间相关性隐式填补），
B 输入完全分布内但丢掉了缺失列携带的关联信息。本 demo 的实验直接对比了两者（见 §5）。

### 方案 C（轻量训练）：用官方 finetuning 模块做「缺失鲁棒性微调」

仓库自带完整的梯度微调设施（`finetuning/finetuned_base.py:184`，AdamW、warmup+cosine、
早停、DDP）。做法：构造微调数据时随机把 test 部分的若干列置 NaN（模拟目标场景），
让 checkpoint 学会 column-level test-only 缺失。改动只在数据增强，不动架构。

### 方案 D（完全掌控）：自己预训练一个 missingness-aware PFN —— 即本 demo

在自己的合成先验里直接加入「test 行整列缺失」的增强，从零预训练小型 per-cell 双轴注意力模型。
适合最终要做领域专用 PFN（如 geo_pfn）的路线：架构、先验、缺失模式全部可控。
demo 证明了该路线端到端可行（§4、§5）。

---

## 2.5 问卷式逐行缺失（每行缺不同的特征）——比整列缺失更原生的场景

实际需求场景：1000 份问卷，每份（row）的作答完整度不同，train 和 test 都不完整。
这是 **cell 级缺失**，与 §2 的列级缺失不同，而且是 TabPFN 支持得**最好**的情况：

1. **shape 要求只是形式**：把所有问卷对齐到统一的问题全集，没答的填 NaN。
   `X_train` 和 `X_test` 自然同列数，不触发任何校验错误。
2. **这是 v2 的分布内输入**：Nature 论文 Methods 明确写明预训练先验对每个单元格以
   ρ_miss 概率独立 MCAR 置缺失；推理时每个 cell 编码为「train 均值填补值 + 缺失指示」。
   官方 issue #108 也演示过 X_train/X_test 带 NaN 开箱即用。Nature Fig. 5 显示含缺失值
   的数据集上没有相对性能下降。**结论：这个场景用原版 TabPFN 无需任何改造。**
3. 注意 `drop-context`（方案 B）对该场景**不适用**——不能只为某一行删掉一列。
   可用策略是：原生 NaN（推荐）、各种填补、以及经典的
   「填补 + missing-indicator 特征」。

真正值得投入的两个增量问题（demo 已扩展验证，见 §5.2）：

- **高缺失率**：问卷可能 30–60% 缺失，远超 v2 先验的典型缺失率。预训练/微调时
  见过高缺失率是否重要？（demo 消融：`minipfn_cells` vs `minipfn_vanilla`。）
- **信息性缺失（MNAR）**：「跳过某题」本身常与标签相关（收入高的人不答收入题）。
  带缺失指示的模型可以**把缺失模式当特征用**；而任何"纯填补"都会销毁这个信号。
  经典对照是 `SimpleImputer(add_indicator=True) + LR`。TabPFN 的先验只模拟了
  MCAR——若你的数据缺失是信息性的，在自己的先验/微调中加入 MNAR 模拟是
  超越原版 TabPFN 的机会点。

demo 对应实现：`prior.py` 的 `random_cell_missing`（每行完整度不同 + 30% 任务
标签相关缺失）作为训练增强（`--augmentation cells`，现为默认），
`eval.py --scenario cells` 输出 MCAR（0/20/40/60%）与 MNAR（30%）两组表。

---

## 3. 问题 2：预训练数据量 & 用 TabPFN 生成合成数据

### 3.1 官方各版本的预训练规模（论文原始数字）

| 版本 | 合成数据集数量 | 训练配置 | 参数量 | 硬件/时长 |
|---|---|---|---|---|
| v1 (ICLR23) | 9,216,000（18k steps × batch 512） | 每数据集 1,024 行、≤100 特征、≤10 类 | 25.8M | 8× RTX 2080 Ti，20 小时 |
| v2 (Nature25) | ~130,000,000（≈2M steps × batch 64） | ≤2,048 train 行 + 128 query 行、1–160 特征 | 7M (clf) | 8× RTX 2080 Ti，约 2 周 |
| v2.5 | 未披露 | — | 10.7M (clf, 24 层) | 未披露 |
| v3 | >8 万亿 token（steps 未披露） | 新增空间/时序/多类先验 | 53M (clf) | EuroHPC LUMI (MI250X) |

注意：**先验生成代码没有随推理仓库发布**（`tab_pfn_src` 只有推理与微调），
v1 的旧仓库（automl/TabPFN）里有可参考的 SCM 先验实现。

### 3.2 我们需要多少数据？

数据是免费生成的，真正的预算是**步数 × 批大小（等价于任务数）× 每任务规模**。经验锚点：

- v1 的学习曲线在 ~10M 数据集附近趋平（论文 App.）——十万级任务就能得到明显的 ICL 能力，
  百万~千万级达到 v1 水平；
- 本 demo：1.6M 参数模型，**384k 任务（12k steps × 32）在 Apple M 系列上约 1 小时**
  即可在合成分布上远超 logistic 回归基线并展现缺失鲁棒性；
- 若目标是「特定领域（特征数几十、行数几百）的专用模型」，v2 的 1.3 亿量级完全不必要，
  1M–10M 任务 + 5–20M 参数是合理起点；关键在**先验覆盖目标数据分布**（这比数据量重要）。

### 3.3 能否用原版 TabPFN 为我们的模型产生合成数据？—— 能，三种方式 + 一个许可陷阱

1. **自回归生成**（tabpfn-extensions `unsupervised` 模块，Apache 2.0）：
   按 p(x,y|D)=∏ p(x_j|x_<j,D) 逐特征采样（分类头 + 回归 bar-distribution 头），
   支持随机特征顺序和 DAG 约束。适合在少量真实数据上 fit 后批量扩增「逼真表格数据」，
   缓解纯 SCM 先验到真实数据的分布差距（Prior Labs 自己也这么干：Real-TabPFN-2.5
   用 43 个真实数据集做继续训练）。
2. **教师蒸馏**：任意方式采样 X，用 TabPFN 的预测分布打软标签训练学生模型。
   注意 in-context 教师给*自己的上下文行*打分会泄漏标签、软标签塌缩——要用
   **out-of-fold 标注**（Pocket Foundation Models, arXiv 2605.18654 的核心教训；
   学生保留教师 ~96% AUC）。
3. **能量模型采样**：TabPFGen（SGLD, arXiv 2406.05216）、TabEBM（NeurIPS 24）——冻结
   TabPFN 直接当 EBM 用，无需训练。
4. **许可（重要）**：
   - **v2 权重**：Prior Labs License（Apache 2.0 + 归属条款）。可商用地用它生成数据/蒸馏，
     但若发布由其输出训练的模型，需标注 “Built with PriorLabs-TabPFN” 且模型名以
     “TabPFN” 开头（License §10）。
   - **v2.5 / v2.6 / v3 权重**：**非商用许可**，且明确禁止用其输出
     「train, fine-tune, or distill a model that is competitive」。
   - 结论：**做数据生成/蒸馏请用 v2 权重**
     （`TabPFNClassifier.create_default_for_version(ModelVersion.V2)`）。

对本项目的建议组合：**自研 SCM 先验（本 demo 已实现雏形）为主 + TabPFN v2 自回归生成的
「逼真任务」为辅混入预训练**，再在下游少量真实数据上验证。

---

## 4. 问题 3：端到端 demo（`src/geo_pfn/minipfn/`）

1.6M 参数的迷你 TabPFN，忠实复刻 v2 的关键机制并加上「测试列缺失」训练策略：

| demo 组件 | 对应 TabPFN v2 机制 |
|---|---|
| cell token=`Linear([z值, 缺失flag]) + 随机列嵌入投影` | NaN 指示通道 + train 均值填补（此处 z-score 后填 0）+ 随机特征嵌入（`tabpfn_v2.py:593/666`） |
| 每层：特征轴 attention → 样本轴 attention（K/V 仅 train 行）→ MLP | `AlongRowAttention` / `AlongColumnAttention`（`tabpfn_v2.py:119/151/221`） |
| y 作为额外一列 token，test 行用学习的 mask 向量 | y 列拼接（`tabpfn_v2.py:898`）|
| 从 test 行 y 列 token 读出 logits | `_decode`（`tabpfn_v2.py:777`）|
| 随机 MLP/SCM 先验 + 等频 rank 分箱成类别 | v1/v2 SCM 先验（Nature Methods）|
| 先验含 cell 级 MCAR 缺失 | v2 先验 ρ_miss MCAR |
| **训练增强：50% 任务随机将 test 行的 1..F/2 列置 NaN** | **本 demo 新增（v2 没有 column 级 test-only 缺失）** |

文件：

- `config.py` — `PriorConfig` / `ModelConfig` / `TrainConfig`（dataclass + `__post_init__` 校验）
- `prior.py` — 批量向量化 SCM 任务生成器 + 缺失列腐蚀（`random_test_missing` 等）
- `model.py` — `MiniPFN`（per-cell token、双轴注意力、NaN 原生支持）
- `train.py` — 预训练脚本；`--drop-task-prob 0` 训练无增强消融模型
- `eval.py` — 缺失策略对比评估（合成任务 + sklearn 真实小数据集迁移）
- `eval_tabpfn.py` — 原版 TabPFN v2 配对基线（重放同一随机序列）
- 各 `*_test.py` — colocated pytest（含「test 行相互独立」「test 标签不泄漏」等性质测试）

运行：

```bash
# 预训练（MPS/CUDA/CPU 自动选择；~1 小时 @ Apple Silicon）
uv run python -m geo_pfn.minipfn.train --steps 12000 --out checkpoints/minipfn.pt
# 消融：无缺失增强
uv run python -m geo_pfn.minipfn.train --steps 12000 --drop-task-prob 0 --out checkpoints/minipfn_vanilla.pt
# 评估（合成 + wine/breast_cancer 迁移；0%/25%/50% 测试列缺失）
uv run python -m geo_pfn.minipfn.eval --checkpoint checkpoints/minipfn.pt
# 真实基线：原版 TabPFN v2（配对重放，MPS 约 2 小时）
uv run python -m geo_pfn.minipfn.eval_tabpfn
# 测试
uv run pytest src/geo_pfn/minipfn/
```

评估协议：对每个任务隐藏一部分特征列（只在 test 行），同一组隐藏列喂给所有策略——
`nan-fill`（方案 A）、`drop-context`（方案 B）、`mean-impute`（无缺失 flag 的均值填补）、
`logreg impute` / `logreg retrain`（经典基线）。

## 5. 实验结果

设置：两个 1.6M 参数模型各预训练 12,000 步 × batch 32（≈38.4 万个合成任务，Apple Silicon MPS，
单独运行约 1 小时）：**aug** 带 50% 任务的测试列 dropout 增强，**vanilla** 不带（消融）。
评估：400 个新采样合成任务 + 两个真实数据集（100 行 train 上下文、随机 10 列），
0% / 25% / 50% 的特征列只在 test 行缺失；同一组缺失列喂给所有策略（配对比较）。

### aug 模型（`checkpoints/minipfn.pt`）

合成任务（400 个）：

| 策略 | 0% | 25% 缺失 | 50% 缺失 |
|---|---|---|---|
| minipfn nan-fill | **0.704** | 0.662 | 0.616 |
| minipfn drop-context | 0.704 | **0.670** | **0.627** |
| minipfn mean-impute | 0.704 | 0.625 | 0.557 |
| logreg impute | 0.660 | 0.605 | 0.551 |
| logreg retrain | 0.660 | 0.630 | 0.599 |

真实数据（合成→真实零样本迁移，模型从未见过真实数据）：

| 数据集 / 策略 | 0% | 25% | 50% |
|---|---|---|---|
| breast_cancer · nan-fill | 0.940 | 0.935 | 0.919 |
| breast_cancer · drop-context | 0.940 | 0.940 | 0.910 |
| breast_cancer · mean-impute | 0.940 | 0.928 | 0.806 |
| breast_cancer · logreg retrain | 0.955 | 0.951 | 0.926 |
| wine · nan-fill | 0.962 | 0.882 | 0.803 |
| wine · drop-context | 0.962 | 0.905 | 0.867 |
| wine · mean-impute | 0.962 | 0.749 | 0.592 |
| wine · logreg retrain | 0.974 | 0.951 | 0.926 |

（vanilla 消融的完整表格：合成任务 nan-fill 0.709 / 0.655 / 0.615，与 aug 基本持平；
wine 上 aug 全面更好 0.962/0.882/0.803 vs 0.923/0.846/0.782，breast_cancer 上两者相当。）

### 结论

1. **测试特征缺失完全可行**：in-context 模型随缺失比例平滑退化（50% 列缺失仍保有大部分精度），
   在分布内任务上所有缺失水平都优于 logistic 基线。
2. **缺失指示（missing flag）价值巨大**：mean-impute（隐瞒缺失事实）在 50% 缺失时比
   nan-fill 差 6~21 个百分点（wine: 0.592 vs 0.803）。静默均值填补是最差的选择——
   这正是 TabPFN 用 NaN 指示通道的原因。
3. **drop-context 略优于 nan-fill**（本 demo 规模下）：对 in-context 学习器，"把缺失列从上下文
   一起删掉重新推断" 免费且稍好。用原版 TabPFN 时两条路线都值得尝试（fit 无训练成本）。
4. **消融的诚实结论**：列级 dropout 增强在合成分布上收益很小——因为先验里已有 cell 级 MCAR
   缺失，模型从中泛化到了列级缺失（与 TabPFN v2 只训 MCAR 也能处理 NaN 的观察一致）。
   增强在 wine 迁移上有明显收益但在 breast_cancer 上无差异。含义：**先验中包含缺失模拟是
   必要条件；针对目标缺失模式的增强是锦上添花的调优手段**，可在微调阶段（方案 C）再加。
5. **PFN 范式本身得到验证**：1.6M 参数、笔记本 1 小时预训练、纯合成数据，
   在真实数据集上零样本达到 0.94–0.96 准确率。

## 5.2 问卷式逐行缺失实验（`--scenario cells`）

设置：`minipfn_cells`（训练增强：70% 任务带每行不同的缺失率，最高 60%，其中 30%
为标签相关 MNAR）对比 `minipfn_vanilla`（只见过先验内建的 ≤10% MCAR）。
评估：所有行（train 上下文 + test）都按每行不同的比例缺失；400 个合成任务，
相同种子 → 结果完全配对。**真实基线**：原版 TabPFN v2 checkpoint（7M 参数、
约 1.3 亿合成数据集预训练、n_estimators=8）经 `eval_tabpfn.py` 重放同一随机序列
参战（logreg 行逐位一致证明掩码级配对）。注意合成任务对 mini 模型是分布内、
对 v2 是分布外（主场优势），跨模型水平以真实数据和 native−impute 差值为准。

**MCAR 扫描**（native = 直接喂 NaN；合成任务）：

| 模型 / 平均缺失率 | 0% | 20% | 40% | 60% |
|---|---|---|---|---|
| minipfn_cells · native | 0.669 | 0.638 | 0.605 | 0.568 |
| minipfn_vanilla · native | 0.666 | 0.635 | 0.601 | 0.564 |
| tabpfn-v2 · native（真实基线，分布外） | 0.652 | 0.609 | 0.572 | 0.539 |
| tabpfn-v2 · mean-impute | 0.653 | 0.608 | 0.570 | 0.538 |
| （同任务 logreg mean-impute） | 0.629 | 0.594 | 0.563 | 0.536 |
| （同任务 logreg knn-impute） | 0.633 | 0.603 | 0.570 | 0.539 |

**标签相关缺失（MNAR，30% 基准率；合成任务）——关键结果**：

| 策略 | minipfn_cells | minipfn_vanilla |
|---|---|---|
| minipfn native | **0.706** | 0.620 |
| minipfn mean-impute | 0.617 | 0.611 |
| logreg mean-impute | 0.578 | 0.578 |
| logreg mean+indicator | 0.653 | 0.653 |
| logreg knn-impute | 0.590 | 0.590 |
| tabpfn-v2 native（真实基线） | 0.635 | — |
| tabpfn-v2 mean-impute | 0.634 | — |

（tabpfn-v2 行与 mini 模型无关，同一批配对任务上单独评估；真实数据上
v2 native 为 breast_cancer 0.955/0.930/0.902/0.875、wine 0.974/0.967/0.905/0.823，
两个数据集上都与自身 mean-impute 打平。）

### 结论（问卷场景）

1. **MCAR 鲁棒性可以从低比例泛化到高比例**：vanilla 只见过 ≤10% 的 cell 缺失，
   在 60% 缺失下与专门训练的 cells 模型几乎打平（0.564 vs 0.568），且两者在
   所有缺失率下都优于均值/KNN 填补 + logistic 基线。含义：原版 TabPFN 处理
   高比例 MCAR 缺失大概率也够用。
2. **MNAR 利用能力必须靠训练分布获得，结构本身不够**：两个模型的 flag 通道结构
   完全相同，但 vanilla 的 native（0.620）输给经典的 `mean+indicator` 线性基线
   （0.653）；而先验中模拟过 MNAR 的 cells 模型达到 **0.706**——比 vanilla 高
   8.6 个百分点，比 indicator 基线高 5.3 个百分点。
3. **缺失模式本身成为了特征**：cells 模型在 30% MNAR 下的 0.706 甚至**高于它自己
   在 0% 缺失下的 0.669**——"哪些题没答"携带的标签信息超过了被抹掉的数值信息，
   而模型学会了在 in-context 中读取它。任何"先填补再建模"的流程都会销毁该信号
   （mean-impute 崩到 0.617/0.578-0.590）。
4. **对原版 TabPFN 的推论——已实测验证**：v2 的 native（0.635）与自身
   mean-impute（0.634）在 MNAR 下完全打平（MCAR 扫描下同样打平）——原版模型
   确实不利用缺失模式，且在同一批任务上被 1.6M 的 MNAR 预训练模型超出 7.1pt。
   **能力差距来自训练分布而非规模；在自研先验或微调数据中加入 MNAR 模拟是
   超越原版 TabPFN 的明确机会点**。真实数据上 v2 展现大先验的迁移优势
   （wine 60% 缺失 0.823 vs mini 0.746），两条结论互补：**先验的领域覆盖决定
   基础水平，先验的缺失机制决定缺失利用能力**。
