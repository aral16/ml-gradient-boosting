# Online Purchase Intent Prediction using Gradient Boosting

## Problem
E-commerce platforms aim to identify browsing sessions that are likely to convert into purchases.  
This project predicts whether an online shopping session will generate revenue using behavioral features.

## Dataset
Source: UCI Online Shoppers Purchasing Intention  

- 12,330 sessions  
- Mix of behavioral and categorical features (pages visited, bounce rate, month, visitor type, traffic type, etc.)  
- Target:
  - Revenue = True → purchase occurred
  - Revenue = False → no purchase  

The dataset is highly imbalanced (~15% buyers).

## Why Gradient Boosting?
Random Forest provides strong baseline performance, but Gradient Boosting is designed to further improve:

- Ranking quality (PR-AUC / ROC-AUC)
- Minority class detection
- Learning from hard-to-classify sessions

Gradient Boosting trains trees sequentially, where each new tree focuses on correcting errors made by previous ones, making it particularly effective for imbalanced behavioral data.

## Model
LightGBMClassifier (Gradient Boosting)

- n_estimators = 500  
- learning_rate = 0.05  
- max_depth = 6  
- num_leaves = 31  
- min_child_samples = 30  
- subsample = 0.8  
- colsample_bytree = 0.8  
- class_weight = balanced  
- Stratified 5-fold Cross-Validation  

## Threshold Strategy
Threshold selected using CV out-of-fold predictions (no test leakage).

Goal:
- Maintain Recall ≥ 80% (detect most buyers)
- Maximize Precision under that constraint

Chosen threshold: 0.227


---

## Cross-Validation Performance

| Metric | Mean |
|------|------|
Accuracy | 0.894 |
Precision | 0.649 |
Recall | 0.691 |
F1 | 0.669 |
ROC-AUC | 0.923 |
PR-AUC | 0.725 |

---

## Test Performance (Frozen Threshold)

| Metric | Value |
|------|------|
Accuracy | 0.853 |
Precision | 0.516 |
Recall | 0.781 |
F1 | 0.621 |
ROC-AUC | 0.915 |
PR-AUC | 0.697 |

Confusion Matrix:

[[2707, 420],
[125, 447]]


---

## Comparison vs Random Forest

| Metric | Random Forest | Gradient Boosting |
|------|---------------|------------------|
Accuracy | **0.872** | 0.853 |
Precision | **0.564** | 0.516 |
Recall | 0.752 | **0.781** |
F1 | **0.644** | 0.621 |
ROC-AUC | 0.915 | 0.915 |
PR-AUC | 0.680 | **0.697** |

Gradient Boosting improves ranking quality and recall, while Random Forest provides more balanced precision.

---

## Key Insights
- Gradient Boosting excels at ranking rare purchasing sessions.
- Lower decision thresholds improve recall at the cost of precision.
- PR-AUC improvements indicate better buyer prioritization.
- Model choice depends on business objective: coverage vs cost.

## Limitations
- No temporal session modeling
- Class imbalance remains challenging
- No direct profit-based optimization

## Conclusion
Gradient Boosting provides stronger ranking performance and improved buyer detection compared to Random Forest.  
This project highlights how boosting models trade precision for recall and why PR-AUC is the appropriate metric for imbalanced e-commerce prediction tasks.

---

## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
