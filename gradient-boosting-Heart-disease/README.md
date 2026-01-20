# Heart Disease Prediction using Gradient Boosting

## Problem
Early identification of heart disease is critical in clinical decision-making.  
This project predicts whether a patient presents heart disease using structured clinical features and compares three tree-based models:

- Decision Tree  
- Random Forest  
- Gradient Boosting  

## Dataset
Source: UCI Heart Disease (Cleveland)

- 303 patients  
- 13 clinical features (demographics, ECG, exercise, blood markers)  
- Original labels (0–4) converted to binary:
  - 0 → No disease
  - 1–4 → Disease present  

## Why Gradient Boosting?
Random Forest reduces variance by averaging many trees, but all trees are built independently.  
Gradient Boosting improves performance by **training trees sequentially**, where each new tree corrects the mistakes of the previous ensemble.

This makes Gradient Boosting particularly effective when:
- Feature interactions are complex
- Minority-class detection matters
- Probability ranking quality (PR-AUC, ROC-AUC) is important

## Model
GradientBoostingClassifier

- n_estimators = 200  
- learning_rate = 0.05  
- max_depth = 6  
- subsample = 0.8  
- Stratified 5-fold Cross-Validation  

## Threshold Strategy
Threshold selected using **CV out-of-fold predictions** (no test leakage).

Goal:
- Maintain Recall ≥ 80% (minimize false negatives)
- Maximize Precision under that constraint

Chosen threshold: 0.178


---

## Cross-Validation Performance

| Metric | Mean |
|--------|------|
Accuracy | 0.783 |
Precision | 0.770 |
Recall | 0.754 |
F1 | 0.759 |
ROC-AUC | 0.890 |
PR-AUC | 0.898 |

---

## Test Performance (Frozen Threshold)

| Metric | Value |
|--------|-------|
Accuracy | 0.802 |
Precision | 0.740 |
Recall | 0.881 |
F1 | 0.804 |
ROC-AUC | 0.901 |
PR-AUC | 0.888 |

Confusion Matrix:

[[36, 13],
[5, 37]]


---

## Model Comparison (Test Set)

| Metric | Decision Tree | Random Forest | Gradient Boosting |
|------|---------------|---------------|------------------|
Accuracy | 0.736 | 0.780 | **0.802** |
Precision | 0.667 | 0.712 | **0.740** |
Recall | 0.857 | 0.881 | **0.881** |
F1 | 0.750 | 0.787 | **0.804** |
ROC-AUC | 0.836 | **0.928** | 0.901 |
PR-AUC | 0.772 | **0.926** | 0.888 |

### Interpretation
- **Decision Tree**: interpretable but unstable and overfits  
- **Random Forest**: best ranking metrics (ROC/PR-AUC)  
- **Gradient Boosting**: best **decision-quality metrics** (Accuracy, Precision, F1) at fixed recall  

Gradient Boosting produces **more confident probability estimates**, allowing a lower threshold while maintaining high recall.

---

## Key Insights
- Gradient Boosting improves bias by sequentially correcting errors.
- Lower threshold reflects stronger probability calibration.
- Random Forest ranks examples slightly better, but Gradient Boosting converts ranking into better final decisions.
- Threshold tuning is critical for clinical tradeoffs.

## Limitations
- Small dataset size
- No external clinical validation
- Not intended for real medical diagnosis

## Conclusion
Gradient Boosting provides the strongest overall predictive performance for heart disease classification when decision thresholds matter.  
This project demonstrates the full progression from single trees to ensemble methods and highlights why Gradient Boosting is the industry standard for tabular clinical data.

---

## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
