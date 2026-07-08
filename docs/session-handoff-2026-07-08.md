# Session 交接文档：TabPFN 缺失特征研究（截至 2026-07-08）

> 供后续 session/协作者快速接手。深度细节见
> [`tabpfn-missing-features-analysis.md`](tabpfn-missing-features-analysis.md)（分析与全部数字）
> 与 [`minipfn-report.html`](minipfn-report.html)（图文版，浏览器直接打开）。

## 0. 背景与目标

用户（liuchang）的核心问题：TabPFN 要求 X_train/X_test 特征一致，能否支持
**X_test 缺失特征**——后澄清为**问卷式逐行缺失**（每行/每份问卷缺不同的题，
train 和 test 都不完整，缺失可能与标签相关）。长期目标是 geo_pfn 项目的
领域专用 PFN。工作全部在 `~/projects/geo_pfn`（uv 管理，Python 3.14，
`tab_pfn_src/` 为 vendored 只读 TabPFN 源码）。

## 1. 已完成的三轮实验

**共同设置**：三个结构完全相同的 1.6M 参数 mini-TabPFN（per-cell token、
特征轴+样本轴双注意力、`[z值, missing_flag]` cell 编码，复刻 v2 布局），
同一随机 MLP/SCM 合成先验（内建 ≤10% cell MCAR），各训练 12k 步 × batch 32
（Apple Silicon MPS 约 1 小时/个）。评估固定 seed=1234 → 400 个配对合成任务
+ breast_cancer/wine；所有策略吃完全相同的缺失掩码。

| 轮次 | 内容 | 训练产物（未入库，.gitignore） |
|---|---|---|
| 1 | 整列缺失场景（test 行整列 NaN，0/25/50%） | `checkpoints/minipfn.pt`（列 dropout 增强）、`minipfn_vanilla.pt`（无增强） |
| 2 | 问卷式逐行缺失：MCAR 扫描 0/20/40/60%（全部行腐蚀）+ MNAR@30%（缺失率随类别缩放） | `checkpoints/minipfn_cells.pt`（每行独立缺失率 ≤60% + 30% 任务 MNAR 增强） |
| 3 | 原版 TabPFN v2 真实基线：重放同一随机序列（配对性由 logreg 行逐位一致验证），native + mean-impute 两策略，4000 次 fit/predict ≈ 111 min | 无（v2 权重自动下载，无需登录） |

## 2. 核心结论（数字见分析文档 §5）

1. **缺失特征是 TabPFN 原生能力**：特征数一致性检查只在 sklearn 校验层；
   缺失填 NaN 即为分布内输入（v2 先验模拟过 MCAR + 指示通道编码）。
2. **不要预填补**：mean/KNN 填补处处劣于 native NaN，且销毁 MNAR 信号。
3. **MCAR 鲁棒性从低比例自动泛化到高比例**（≤10% 训练 → 60% 评估打平），
   高缺失率增强不必要。
4. **MNAR 利用能力 = 训练分布，与结构/规模无关**（核心发现）：
   同结构 cells 0.706 vs vanilla 0.620；v2 native 0.635 ≈ 自身 mean-impute
   0.634（完全不利用缺失模式），被 1.6M 模型超 7.1pt；0.706 > 零缺失基准
   0.669——缺失模式本身成为特征。
5. **综合能力 v2 明显更强**（真实数据，wine 60% 缺失 0.823 vs 0.746）；
   mini 在合成任务的领先是主场优势（任务采自其自身先验），不可引用。

## 3. 被排除的方向（勿重复踩）

- **整列缺失作为主框架**——用户澄清为逐行缺失；drop-context 对逐行不适用。
- **结构级改造**（缺失 cell 不生成 token 等）——ragged batch 复杂、丢失
  缺失位置信号；消融证明瓶颈在训练分布不在结构。
- **高缺失率/列 dropout 增强作为必要手段**——MCAR 下收益在噪声内。
- **用 TabPFN 2.5/2.6/3 生成数据或蒸馏**——许可禁止（非商用 + 禁训竞争
  模型）。**商用相关只能用 v2 权重**（Prior Labs License：归属 +
  发布模型需 “TabPFN” 前缀命名）。
- **checkpoint 入库**——体积大且 1 小时可复训。

## 4. 待做队列（按性价比排序，均未执行）

1. **v2 + 手动拼接 0/1 缺失指示列**（零训练成本，可能部分弥补 MNAR 缺口）；
2. **在用户真实问卷数据上跑 v2**（NaN vs 指示列）判定缺失是否信息性；
3. 方案 C：官方 `tab_pfn_src` 的 `finetuning/` 模块 + MNAR 模拟微调 v2；
4. MNAR 机制错配泛化测试（一族机制训练、另一族评估）；
5. TabPFN v2 自回归生成数据（tabpfn-extensions unsupervised）混入自研先验；
6. 回归头（bar-distribution）、更宽先验（>10 特征、>4 类、类别变量）、
   geo 领域专用先验（最终目标）。

## 5. 代码与复现

```bash
cd ~/projects/geo_pfn
uv run pytest src/geo_pfn/minipfn/            # 29 个测试
# 训练（--augmentation cells|columns|none）
uv run python -m geo_pfn.minipfn.train --steps 12000 --augmentation cells --out checkpoints/minipfn_cells.pt
# 评估（--scenario cells|columns）
uv run python -m geo_pfn.minipfn.eval --checkpoint checkpoints/minipfn_cells.pt --scenario cells
# v2 真实基线（重放同一随机序列；MPS ~2h）
uv run python -m geo_pfn.minipfn.eval_tabpfn
```

模块：`src/geo_pfn/minipfn/{config,prior,model,train,eval,eval_tabpfn}.py`
+ colocated `*_test.py`。配对机制：评估按固定顺序消耗同一个
`torch.Generator(seed)`——**改动 `evaluate_synthetic_cells` 的随机数消耗顺序
会破坏与历史数字/v2 基线的配对**，改前先读 `eval_tabpfn.py` 的 docstring。

## 6. 历史记录

- PR：#1 demo + 分析文档；#2 HTML 报告 charset 修复；#3 TabPFN v2 基线；
  #4 总体评价章节。全部 squash merge 到 `main`。
- 图文报告 Artifact（与 `docs/minipfn-report.html` 同内容）：
  https://claude.ai/code/artifact/bf1f25f8-2a5b-447e-ae53-ec999f8a8013
- 论文事实速查（详见分析文档 §1/§3）：v1 = 9.2M 合成数据集 / 25.8M 参数；
  v2 = ~130M / 7M（Nature 2025，先验含 MCAR）；v3 = >8T token / 53M，
  三段式行压缩架构。先验生成代码未随仓库发布。
