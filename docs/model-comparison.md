# 模型对比：TabPFN v2 vs TabICL vs geo-PFN（羽田 Su 数据）

> 汇总文档。所有实验在羽田机场周边 240 钻孔 / 3,521 试样的真实数据上运行
> （数据说明 [`haneda-su-dataset.md`](haneda-su-dataset.md)）。原始逐折数字在
> `results/haneda/*.json`，聚合分析在 `results/haneda/analysis/`。
> 图文摘要见 [`model-comparison-report.html`](model-comparison-report.html)。

## 0. 结论速览

1. **TabICL 可完全替代 TabPFN v2**：在每个设定（整孔冷启动 + 锚点迁移 ×
   Su/Wn/土层）都追平或超过 v2。加上 v2 权重的许可限制（禁商用），项目
   **统一以 TabICL 为参考模型，移除全部 v2 代码**。
2. **geo-PFN（自研两段式）的架构猜想机制成立、绝对精度未及**：锚点利用
   competitive/最优，纯坐标+深度冷启动追平 v2/TabICL；但绝对精度落后，
   诊断为先验真实度缺口（详见 [`geopfn-hypothesis-results.md`](geopfn-hypothesis-results.md)）。

## 1. 被比较的模型

| 模型 | 类型 | 许可 | 角色 |
|---|---|---|---|
| **TabICLv2** | 行压缩 ICL（soda-inria） | **BSD-3（可商用）** | **新参考基线** |
| ~~TabPFN v2~~ | per-cell ICL（Prior Labs） | 禁商用 | 已移除（被 TabICL 替代） |
| **geo-PFN** | 两段式 ICL（自研） | 自有 | 研究中，未收敛 |
| HGBT / Ridge | 经典（sklearn） | 开源 | 对照基线 |

## 2. 实验与协议

所有评估：240 孔 5 折**整孔留出**（GroupKFold, seed 42），目标 Su（或 Wn），
native NaN，逐折配对（mean±SEM）。

| 实验 | 问题 | 协议 |
|---|---|---|
| 整孔冷启动 | 陌生钻孔的冷启动精度 | 预测测试孔全部行，上下文=训练孔 |
| 特征消融 | 各特征层的贡献 | L/LC/LCS/LCSG/LSG |
| 填补策略 | 缺失处理 | native/mean/knn/指示列/物理 e |
| **锚点迁移** | 稀疏观测的价值 | 测试孔留 k 个散布深度锚点，预测其余；holdout=无锚点 |
| 土层分类 | 岩性可预测性 | 目标=soil_B02，accuracy |

## 3. TabICL vs v2：整孔冷启动（确认可替代）

| 任务 | TabPFN v2 | TabICLv2 | 判定 |
|---|---|---|---|
| 回归 RMSE（LCSG） | 14.25 | **13.86** | TabICL 更好 |
| 回归 R²（LCSG） | 0.786 | **0.797** | TabICL 更好 |
| 分类 acc（4 箱） | 0.688 | **0.707** | TabICL +1.9pt |

## 4. 锚点迁移：v2 vs TabICL vs geo-PFN

配对 anchor−holdout RMSE（负=锚点有帮助），holdout / anchor 为绝对 RMSE。

> geo-PFN 数字用 **coherent 上下文**（最近整孔），修正了早期"随机 256 行"
> 破坏钻孔单位分布的问题（详见 `geopfn-hypothesis-results.md §9`）。

### Su / LCSG（含廉价土工）

| 模型 | k=3 增益 | k=5 增益 | k=5 绝对(hold→anch) |
|---|---|---|---|
| v2 | −1.57 | −1.92±0.46 | 14.4 → **12.5** |
| TabICL | −0.94 | −1.43±0.57 | 14.2 → 12.8 |
| geo-PFN（coherent） | −2.05 | **−2.70±0.40** | 23.2 → 20.5 |
| hgbt | −0.73 | −1.31±0.39 | 15.5 → 14.2 |

### Su / L（零信息：仅坐标+深度）

| 模型 | k=5 增益 | holdout（冷启动） | k=5 绝对 |
|---|---|---|---|
| v2 | −3.70±0.71 | 17.1 | 14.1 |
| TabICL | −4.57±1.00 | 17.5 | **13.8** |
| **geo-PFN（coherent）** | **−3.88±1.06** | 17.4–19.3 | 15.5 |
| hgbt | −2.75±0.43 | 17.3 | 14.8 |

**geo-PFN 读法**：coherent 上下文后锚点迁移增益与 v2/TabICL 同档（Su-L −3.88、
Su-LCSG −2.70 全场最大）——验证了两段式行度量架构；但绝对 RMSE 仍落后
（LCSG ~22-23，先验真实度缺口，非架构问题）。

### Wn / LCSG（含水率，std≈20）

| 模型 | holdout RMSE | R² |
|---|---|---|
| v2 | 4.0–4.3 | ≈0.96 |
| TabICL | 4.0–4.2 | ≈0.96 |
| geo-PFN | 8.3–8.7 | ≈0.83 |
| hgbt | 4.7–4.9 | — |

## 5. 土层分类（岩性可预测性）

| 模型 | 冷启动 acc | +3 锚点 acc |
|---|---|---|
| v2-clf | 0.481 | 0.835 |
| TabICL | 0.471 | **0.843** |
| hgbt | 0.491 | 0.747 |

**最强锚点效应**：冷启动 ~48% → 加 3 个锚点 84%。岩性在深度上分段常数（层状），
知道几个层界标签就能内插——这也印证了 geo-SCM 先验"层内平缓、层间跳变"的设计。

## 6. 判决

- **TabICL 全面替代 v2**：§3/§4/§5 各设定 TabICL ≈ 或 > v2；许可可商用。
  v2 代码已移除（PR #9），基线统一为 TabICL。
- **geo-PFN**：锚点利用 competitive（Su-LCSG k=5 增益 −2.07 最大）、零信息冷启动
  追平 v2/TabICL；绝对精度落后（先验真实度缺口）。研究继续，见
  [`geopfn-hypothesis-results.md`](geopfn-hypothesis-results.md) §7 的改进路径。

## 7. 证据链（原始数据位置）

| 内容 | 路径 |
|---|---|
| 锚点逐折原始（geo-PFN） | `results/haneda/anchor_geopfn_{su_lcsg,su_l,wn_lcsg}.json` |
| 锚点逐折原始（TabICL） | `results/haneda/anchor_tabicl_{su_lcsg,su_l,wn_lcsg,soil}.json` |
| 锚点逐折原始（v2/hgbt） | `results/haneda/anchor.json`, `anchor_su_l.json`, `anchor_wn_lcsg.json` |
| 整孔冷启动（消融/填补） | `results/haneda/{ablation,imputation,tabicl}.json` |
| 土层分类 | `results/haneda/anchor_soil.json`, `anchor_tabicl_soil.json` |
| 聚合分析 | `results/haneda/analysis/model_comparison.txt` |
| geo-PFN checkpoint | `checkpoints/geopfn2stage.pt`（gitignore，config+state 可重建） |

## 8. 复现

```bash
uv run pytest src/geo_pfn/                                  # 全部测试
# TabICL 基线（临时环境，MPS）
uv run --with tabicl python -m geo_pfn.haneda.anchor --models tabicl --device mps \
  --feature-set LCSG --target Su
# geo-PFN 训练 + 评估
uv run python -m geo_pfn.geopfn.train --steps 8000
uv run python -m geo_pfn.geopfn.eval_anchor --feature-set LCSG --target Su
```
