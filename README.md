# Gradient Boosting Projects (Classification & Regression)

This repository groups multiple **Gradient Boosting projects** built to understand and demonstrate how boosting models behave across **different problem types** (binary classification, multi-feature behavioral prediction, and regression).

The goal is not just performance, but to develop **modeling intuition**, **proper evaluation**, and **decision-oriented thinking**.

---

## Why Gradient Boosting?

Gradient Boosting models (XGBoost / LightGBM / GradientBoosting) are:

- Industry-standard for tabular data
- Strong at modeling non-linear feature interactions
- Robust to mixed feature types
- Frequently used in finance, healthcare, marketing, and operations

These projects explore **when and why gradient boosting works well**, and where its limitations appear.

---

## 📌 Projects Overview

### 1️⃣ Gradient Boosting — Heart Disease (Classification)

**Problem**  
Predict whether a patient has heart disease based on clinical measurements.

**ML Task**  
Binary classification

**Why this project matters**
- Medical features interact non-linearly (age × cholesterol × heart rate)
- Interpretability and overfitting control are critical
- Strong example of risk prediction under uncertainty

**Key Focus**
- Comparing Gradient Boosting to simpler models
- Precision / recall tradeoffs
- Overfitting control via depth and learning rate
- Feature importance analysis

**Key Insight**
> Gradient Boosting captures complex clinical interactions but requires careful regularization to avoid overfitting, especially on small medical datasets.

---

### 2️⃣ Gradient Boosting — Shoppers Intention (Classification)

**Problem**  
Predict whether an online shopper will make a purchase based on browsing behavior.

**ML Task**  
Binary classification (purchase vs no purchase)

**Why this project matters**
- Behavioral data is noisy and highly non-linear
- Strong class imbalance
- Real-world marketing decision context

**Key Focus**
- Behavioral feature interactions
- Precision vs recall tradeoff (avoiding false positives)
- Threshold selection for business decisions
- Comparison with baseline models

**Key Insight**
> Gradient Boosting significantly improves detection of high-intent users but must be evaluated using business-oriented metrics rather than accuracy.

---

### 3️⃣ Gradient Boosting — Trip Duration (Regression)

**Problem**  
Predict trip duration based on trip and contextual features.

**ML Task**  
Regression

**Why this project matters**
- Continuous target prediction
- Non-linear temporal and spatial patterns
- Demonstrates boosting beyond classification

**Key Focus**
- Regression error analysis (MAE vs RMSE)
- Handling skewed target distributions
- Residual diagnostics
- Comparing linear vs boosted models

**Key Insight**
> Gradient Boosting outperforms linear models by capturing non-linear structure, but residual analysis reveals systematic failure modes on extreme trip durations.

---

## 🧠 What These Projects Demonstrate

Across these three projects, the following ML competencies are covered:

- End-to-end ML workflow (EDA → baseline → model → evaluation)
- Proper metric selection for classification and regression
- Handling non-linearity and feature interactions
- Overfitting detection and control
- Decision-oriented model evaluation
- Clear interpretation of model behavior

---

## 📂 Repository Structure (per project)

Each project follows a consistent structure:

