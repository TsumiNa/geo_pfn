# Haneda 真实数据实验方案

> 数据与领域背景见 [`haneda-su-dataset.md`](haneda-su-dataset.md)。
> 全部实验在本地 Mac (Apple Silicon, MPS) 上运行，端到端预算 ≤ 2 小时。

## 0. 要回答的三个问题

| # | 问题 | 对应对比 |
|---|---|---|
| Q1 | miniPFN vs TabPFN v2（均不微调）在真实数据上的差距；MNAR 增强训练（cells）在真实缺失上是否兑现优势 | 同一分箱分类任务上：miniPFN-cells vs miniPFN-vanilla vs v2 classifier |
| Q2 | 地质数据（土类 + 粒径）对性能的贡献 | 特征集消融 L → LC → LCS → LCSG（+ LSG 对照） |
| Q3 | 各 imputation 策略在真实块状 MNAR 上的效果（合成数据特征相关性过强，此前结论需复检） | 固定特征集下：native NaN vs mean vs KNN vs mean+indicator vs 物理填补 e |

## 1. 统一评估协议（所有 arm 严格共享）

- **划分**：`GroupKFold(n_splits=5, shuffle=True, random_state=42)`，
  分组键 = `BorSeq`。240 孔 → 每折约 48 个测试孔、约 700 测试行、
  约 2800 训练行。所有模型、所有策略、所有特征集使用**同一套折**。
- **任务形式**：
  - **回归（主任务）**：目标 Su 原值（不做 log 变换，skew 仅 0.53）。
    指标：RMSE、MAE、R²（按折算，报告 5 折 mean ± sem）。
    敏感性：Su cap at p99.5≈148 的对照表（仅对最终主要结论复算）。
  - **分类（miniPFN 对比用）**：每折用**训练折**的 Su 四分位数分箱
    → 4 个平衡类（miniPFN 分类头上限 max_classes=4）。测试行用同一
    边界赋标签。指标：accuracy、macro-F1。边界只来自训练折，无泄漏。
- **随机性**：全局 seed 42；miniPFN 上下文采样用独立
  `torch.Generator`，per-fold 派生子 seed，保证任意 arm 子集重跑
  结果不变。**不触碰 `evaluate_synthetic_cells` 的既有配对机制**
  （本实验完全独立于 minipfn/eval.py 的随机数消耗序列）。

## 2. 特征集（Q2 消融）

| 代号 | 列 | #特征 | 备注 |
|---|---|---|---|
| L | depth_m, X, Y | 3 | 纯空间插值 |
| LC | L + Wn, Gs, LL, PL, rho_t, e | 9 | 无地质；e 含 22% 缺失 |
| LCS | LC + soil_B02（编码） | 10 | + 分类地质，恰好 miniPFN 训练上限 |
| LCSG | LCS + gravel/sand/silt/clay_pct | 14 | 全量；粒径 52.5% 缺失。对 miniPFN 超出 2–10 特征训练范围（机制上可运行，标注 OOD） |
| LSG | L + soil + 粒径 | 8 | 只有地质、无便宜土工参数的对照 |

- `W` 弃用（与 Wn 相关 0.982，冗余）；`qu` 显然排除。
- soil_B02 编码：n≥30 的 9 类 + "other" → 整数码 0–9（全局固定词表，
  非训练折拟合——词表只用 X 边缘分布，无标签泄漏）。
  v2 通过 `categorical_features_indices` 声明；miniPFN 与线性模型
  按数值处理（已知局限，写入解读）。

## 3. 测试对象（Q1 + 基线）

| 代号 | 模型 | 任务 | 说明 |
|---|---|---|---|
| v2-reg | TabPFNRegressor v2（`create_default_for_version(V2)`，MPS） | 回归 | 主力。商用许可安全的权重（同前轮） |
| v2-clf | TabPFNClassifier v2 | 分类 | 与 miniPFN 直接可比 |
| mini-cells | miniPFN `checkpoints/minipfn_cells.pt`（MNAR 增强训练） | 分类 | 核心考察对象 |
| mini-vanilla | miniPFN `checkpoints/minipfn_vanilla.pt`（无增强） | 分类 | cells vs vanilla = MNAR 增强在真实数据上的增益 |
| hgbt | HistGradientBoosting（原生 NaN 支持） | 回归+分类 | 经典强基线，树模型的 native-NaN 对照 |
| linear | Ridge / LogisticRegression（mean-impute + 标准化） | 回归+分类 | 线性基线 |
| depth | 仅 depth_m 的线性回归 / 分箱多数类 | 回归+分类 | 必须超越的门槛（in-sample R²≈0.63） |
| dummy | 全局均值 / 多数类 | 回归+分类 | 零信息下界 |

### miniPFN 的上下文协议（3.5k 行 ≫ 训练分布 60–160 行）

- **主协议**：每折做 E=16 次集成——每次从训练折按类分层随机抽
  128 行作上下文，对全部测试行前向，平均 softmax 概率。
  （128 行在训练分布内；分层保证 4 类都在上下文中出现。）
- **上下文规模扫描**（附加）：ctx ∈ {128×16, 512×4, 2048×1}，
  观察行数外推能力。测试行分块（≤512/块）控制注意力内存。

## 4. Imputation 对比（Q3）

固定特征集 **LCSG**（缺失最多、最能区分策略），策略族：

| 策略 | 做法 | 检验什么 |
|---|---|---|
| native | NaN 直接进模型（v2 / miniPFN / hgbt 原生支持） | 缺失模式作为信息 |
| mean | 训练折列均值填补，缺失 flag 对模型隐藏 | 合成实验"处处劣于 native"是否在真实数据成立 |
| knn | KNNImputer(k=5)（训练折拟合） | 强相关填补器在真实弱相关数据上的表现 |
| mean+ind | mean 填补 + 0/1 缺失指示列拼接 | 待做队列 #1：v2 能否借指示列弥补 MNAR 缺口。注意本数据指示列在孔级近乎常数块 |
| physics-e | e 用公式 Gs(1+W/100)/ρt−1 重建（≈无损），粒径保持 NaN | 确定性填补上界；与 native 的差值 = "e 的缺失模式信息" vs "e 的数值信息" |

应用对象：v2-reg、v2-clf、mini-cells、linear（hgbt 只跑 native + mean 做参照）。
mean+ind 对 miniPFN 意味着特征数 14+14=28，远超训练范围——只跑 v2/linear。

## 5. 运行计划与预算

| 阶段 | 内容 | 估算 |
|---|---|---|
| P0 | 单折单 arm 冒烟 + v2 一次 fit/predict 计时 | ~5 min |
| P1 | 特征集消融（5 特征集 × {v2-reg, v2-clf, mini-cells, mini-vanilla, hgbt, linear, depth, dummy} × 5 折，native NaN） | v2 约 50 fits，≲30 min |
| P2 | Imputation（5 策略 × {v2-reg, v2-clf, mini-cells, linear} × 5 折，LCSG） | v2 约 45 fits，≲25 min |
| P3 | miniPFN 上下文规模扫描 + Su-cap 敏感性复算 | ~10 min |
| P4 | 汇总 JSON → 分析报告 | — |

- 结果落盘：`results/haneda/<experiment>.json`（每 arm 每折指标 +
  完整配置回显）；per-row 预测存 `results/haneda/predictions/*.csv`
  （不入库，重跑可再生）。
- 若 v2 单次 fit/predict 在 MPS 上 >60 s，降 `n_estimators`（记录在配置里）。

## 6. 预期读法（预注册的解读框架，防事后合理化）

- **Q1**：v2-clf 与 mini 的差距若与 wine/breast_cancer 上的差距
  （~8 pt）同量级，则"综合能力差距"结论迁移成立。
  mini-cells > mini-vanilla 且差值 > 折间噪声 → MNAR 增强兑现；
  两者打平 → 真实缺失的信息量（§5.3 EDA：控制深度后几 kN/m²）
  不足以体现训练分布差异，核心发现需限定适用条件。
- **Q2**：LCS/LCSG vs LC 的增量为地质贡献。注意粒径存在时本身与
  更"努力的勘察批次"相关（confound），解读时结合 LSG。
- **Q3**：若 native ≈ mean（差 < 折间 sem），则合成实验中
  "不要预填补"的效应量在真实弱相关数据上收缩——这本身就是
  重要结论；physics-e vs native 分离数值信息与缺失模式信息。
- 所有配对比较在**折内配对**（同折同测试行），报告配对差值的
  mean ± sem，而非独立比较两个边缘均值。

## 7. 实现

- 新模块 `src/geo_pfn/haneda/`：
  - `data.py` — 加载校验、特征集构建、soil 编码、分箱、填补策略、
    钻孔 GroupKFold；
  - `runners.py` — miniPFN 上下文集成预测、v2 工厂、sklearn 基线工厂、
    指标计算；
  - `run.py` — CLI：`uv run python -m geo_pfn.haneda.run
    --experiments ablation,imputation,context --device auto
    --out results/haneda`；
  - colocated `data_test.py` / `runners_test.py`（合成小表测试，
    不依赖真实 CSV、不下载 v2 权重）。
- `data/` 与 `results/haneda/predictions/` 加入 `.gitignore`
  （真实数据未经确认不入库；聚合 JSON 体积小，入库）。
- 分支 `feat/haneda-real-data-eval` + PR（遵循仓库分支规范）。
