# Kaggle_Loan_Payback
_Kaggle_Loan_Payback competition_


# Kaggle Tabular Binary Prediction — Final Solution

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![kaggle](https://img.shields.io/badge/Kaggle-Tabular%20Binary%20Classification-20BEFF?logo=kaggle&logoColor=white)
![model](https://img.shields.io/badge/Model-CatBoost%20%2B%20LightGBM-orange)
![status](https://img.shields.io/badge/Status-Final%20Submission-brightgreen)
![score](https://img.shields.io/badge/Score-AUC%200.91781-success)

## 🎓 Introduction

本仓库（Notebook）提供一个**通用且稳健的 Kaggle 表格二分类**最终提交方案：自动定位 `train.csv / test.csv / sample_submission.csv`，并以 `loan_paid_back`（若不存在则回退为 `target`）作为目标列进行建模，输出可直接提交的 `submission.csv`。

整体思路以 **“泛化优先 + 抗分布偏移 + 概率可用”** 为目标：在严格交叉验证下同时训练 **CatBoost（原生类别特征）** 与 **LightGBM（单调约束）**，再进行基于 OOF AUC 的自适应融合与 Isotonic 概率校准，并使用高置信伪标签作为兜底增强，提升鲁棒性与线上表现稳定性。

---

## 🧠 Methodological Framework

### 1) 数据对齐与清洗（Leakage-safe）
- 自动识别数值/类别列：**低基数整数数值列**会被转为类别特征以提升鲁棒性  
- 类别缺失统一填充 `__MISSING__`，数值特征进行 **1%–99% 分位裁剪**抑制异常值

### 2) 自适应金融风格特征（存在则生成）
若数据中出现 `income/salary、loan_amount/amount、interest_rate/rate、debt...、credit_score/score` 等字段，会自动构造：
- `loan_to_income`、`log_income`、`log_amount`
-（若利率存在）组合比值与对数变换等衍生特征

### 3) Adversarial Validation → 样本权重（处理 Train/Test Shift）
训练一个“区分 train vs test”的模型得到样本属于 train 的概率 `p_train`，并用**逆倾向权重**对训练样本加权：
- `w ∝ (1 - p_train) / p_train`（并做截断与归一化），缓解分布偏移导致的过拟合

### 4) 单调约束（LightGBM）
对每个数值特征计算其与目标的 Spearman 方向，得到 `{-1, 0, +1}` 的单调约束向量：
- 数值特征按相关方向施加单调性
- 类别特征不施加约束（置 0）

### 5) 交叉验证训练（CB + LGB）
- **CatBoost**：原生支持类别特征 + 样本权重
- **LightGBM**：类别特征 + 单调约束 + early stopping
- 记录 OOF 预测并计算 AUC

### 6) 融合 + Isotonic 概率校准
- 融合权重按 `w_model ∝ OOF_AUC` 自适应分配  
- 对融合后的 OOF 概率做 **IsotonicRegression** 校准，提升概率可解释性/可用性

### 7) 高置信伪标签兜底增强（Pseudo-labeling）
- 选取 `pred ≥ 0.99` 或 `pred ≤ 0.01` 的测试样本（不足则放宽到 `0.98/0.02`）
- 将伪标签并入训练集，使用更轻量的 LGBM 再训练
- 最终预测：`final_pred = 0.6 * calibrated_blend + 0.4 * aug_lgbm`

