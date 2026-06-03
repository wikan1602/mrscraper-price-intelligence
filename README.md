# MrScraper Price Intelligence & Anomaly Detection Pipeline

This repository contains a robust, production-grade price reconstruction pipeline built for the **MrScraper AI Engineer Take-Home Test**. The system is designed to recover missing e-commerce product prices on days when automated scraping was unavailable, utilizing rich historical context and a minimal anchor set of 100 manual price samples.

---

## 🧠 Approach & Methodology

Our solution tackles the missing price imputation problem through a three-stage architecture:

1. **Robust Feature Engineering** — Extracting cyclical temporal patterns (hour/day), managing skewed distributions via log transformations ($\log_{1p}$), and engineering relative pricing boundaries (`item_price_min`/`max` ratios) to capture inherent item value.

2. **Bayesian Hyperparameter Optimization** — Utilizing Optuna to dynamically tune LightGBM parameters (e.g., heavily restricting `min_child_samples` to prevent overfitting on long-tail, low-frequency shops).

3. **Dynamic Anchor Calibration (Post-Processing)** — Leveraging the 100 surviving anchor samples on the outage day to calculate a real-time market shift multiplier, effectively bridging historical model knowledge with live macro-economic realities.

---

## 🚀 Quick Start & Execution Guide

### 1. Environment Setup

Ensure you have Python 3.9+ installed, then set up the environment and install dependencies:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# Install required packages
pip install -r requirements.txt
```

### 2. Run Training Pipeline

To clean raw historical data, engineer features with zero-division protections, and fit both the Tier 1 (Global) and Tier 2 (Entity-Specific) LightGBM models:

```bash
python run_pipeline.py --mode train
```

> This will generate `global_model.pkl`, `entity_model.pkl`, and `entity_stats.pkl` inside the `models/` directory.

### 3. Run Inference Pipeline (Predicting Missing Data)

The inference engine automatically isolates the 100 anchor samples present in the test file, calculates live market shifts, and imputes the missing prices.

**Approach 1 — Global Marketplace Model (RECOMMENDED):**

```bash
python run_pipeline.py --mode infer \
  --input data/raw/ecommerce_price_prediction-test-3-days.csv \
  --output submission_global.csv \
  --approach global
```

**Approach 2 — Shop/Product Level Model (for audit/comparison):**

```bash
python run_pipeline.py --mode infer \
  --input data/raw/ecommerce_price_prediction-test-3-days.csv \
  --output submission_entity.csv \
  --approach entity
```

---

## 📊 Validation Framework & Key Findings

We implemented a strict **Time-Based Validation Split** (isolating the final scraping date of the training set as a simulated outage day) to rigorously test both architectures before evaluating them on hidden test data.

### Performance Summary

| Approach | Description | Evaluated MAPE | Evaluated MAE |
|----------|-------------|----------------|---------------|
| Approach 1 | Global Marketplace Model + Macro Calibration | **0.71% ✅** | ~181,917 IDR |
| Approach 2 | Shop/Product Level Model + Granular Calibration | 0.86% | ~335,265 IDR |

### Critical Engineering Trade-offs & Production Decision

Based on empirical validation, **Approach 1 (Global Model)** is explicitly recommended for final production deployment.

1. **The Complexity & Small-Sample Trap** — While Approach 2 attempts a highly personalized shop-level prediction using historical priors (`historical_shop_price_mean`), it suffers from extreme data sparsity. Distributing the 100 anchor samples across hundreds of unique shops leaves most entities with 0–1 reference points. Calculating a calibration multiplier from a single observation introduces severe variance, causing the Tier 2 model to overfit to micro-level pricing noise.

2. **Stable Historical Corridors** — Our EDA revealed that global pricing boundaries (`item_price_min` and `item_price_max`) serve as exceptionally strong linear anchors. The Global LightGBM model utilizes these boundaries effectively without introducing the noise of sparse, low-frequency shops.

3. **Production Safeguards Implemented:**
   - **Numerical Instability Fix** — Items with single variants create a flat denominator boundary (`max - min == 0`), which natively causes standard scaling formulas to explode. Our pipeline applies a safe vectorized clamp ($+1 \times 10^{-5}$) to ensure mathematical stability.
   - **Multilevel Cold-Start Defense** — For Approach 2, any shop or product encountered during test-time with insufficient history is automatically protected via a fallback hierarchy (`.fillna(global_mean)`), ensuring the pipeline never breaks in production.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `notebooks/` | Contains `01_eda.ipynb` and `02_validation.ipynb` detailing research, Optuna tuning, and hypothesis testing. |
| `src/data_loader.py` | Handles file ingestion, datetime casting, and basic formatting. |
| `src/features.py` | Executes temporal cyclical encoding, pricing boundary indexes, and zero-division scaling protections. |
| `src/train.py` | Fits and serializes both the baseline and entity-conditioned LightGBM models. |
| `src/inference.py` | Handles dynamic anchor-set isolation, computes multi-level calibration ratios, and writes final CSV payloads. |
| `run_pipeline.py` | Unified CLI for executing training and inference operations. |