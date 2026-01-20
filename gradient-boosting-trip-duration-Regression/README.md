# Gradient Boosting Regression — NYC Taxi Trip Duration

## Problem
Predict the duration of taxi trips in New York City using spatio-temporal and trip-related features.

Accurate trip duration estimation is critical for:
- ETA prediction
- Pricing and routing
- Fleet optimization

This dataset is particularly challenging due to strong spatial effects, traffic variability, and extreme outliers.

---

## Dataset
Source: Kaggle — **NYC Taxi Trip Duration**

Target:
- `trip_duration` (seconds)

Features include:
- Pickup and dropoff latitude/longitude  
- Vendor ID  
- Passenger count  
- Trip datetime  

The target distribution is highly skewed, with many short trips and a long tail of very long trips.

---

## ML Task
Gradient Boosting Regression (LightGBM)

---

## Approach

1. Train / Validation / Test split  
2. Log-transform target: y = log1p(trip_duration)
3. Train LightGBM regressor on log space  
4. Evaluate:
- Validation in log space
- Test in original scale (seconds)
5. Residual diagnostics

---

## Model
LightGBMRegressor

- n_estimators = 800  
- learning_rate = 0.05  
- max_depth = 6  
- num_leaves = 31  
- min_child_samples = 40  
- subsample = 0.8  
- colsample_bytree = 0.8  

This configuration prioritizes stability and generalization for heavy-tailed regression targets.

---

## Validation Results (log space)

| Metric | Value |
|------|------|
MAE | 0.325 |
RMSE | 0.474 |
R² | 0.644 |

The model explains ~64% of the variance in log-transformed trip duration.

---

## Test Results (original scale)

| Metric | Value |
|------|------|
MAE | 355 seconds |
RMSE | 3149 seconds |
R² | 0.028 |

The model outperforms a Random Forest baseline and simple mean prediction, but still exhibits substantial error on long trips.

---

## Comparison vs Random Forest

| Metric (Test) | Random Forest | Gradient Boosting |
|-------------|---------------|------------------|
MAE | 477 s | **355 s** |
RMSE | 3375 s | **3150 s** |
R² | −0.12 | **+0.03** |

Gradient Boosting clearly improves regression performance by reducing bias and handling skewed targets more effectively.

---

## Residual Analysis

- Error variance increases with trip duration
- Long trips are still underpredicted
- Residuals remain heavy-tailed

This indicates that **feature engineering**, not model complexity, is now the main limitation.

---

## Key Insights
- Random Forest fails on skewed geospatial regression problems without strong features.
- Log-transforming the target is essential for stability.
- Gradient Boosting significantly improves performance by learning residual structure.
- Boosting reduces bias where bagging fails.

---

## Limitations
- No explicit distance or route features
- No traffic or temporal encoding
- High noise inherent to taxi data
- Model not suitable for production ETA use

---

## Conclusion
Gradient Boosting provides a substantial improvement over Random Forest for trip duration regression, particularly when combined with log-transformed targets.  
This project demonstrates why boosting methods are preferred for complex, skewed regression problems and highlights the critical role of feature engineering in geospatial modeling.

---

## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
