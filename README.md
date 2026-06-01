# MrScraper Price Intelligence & Anomaly Detection Pipeline

This repository contains a robust, production-grade price reconstruction pipeline built for the **MrScraper AI Engineer Take-Home Test**. The system is designed to recover missing e-commerce product prices on days when automated scraping was unavailable, utilizing rich historical context and a minimal anchor set of 100 manual price samples.

---

## 🚀 Quick Start & Execution Guide

### 1. Environment Setup

Ensure you have Python 3.9+ installed, then set up the environment and install dependencies:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run Training Pipeline (Fits Both Approaches)

To clean raw historical data, engineer features with zero-division protections, and train both the Tier 1 Global Model and Tier 2 Entity Model:

```bash
python run_pipeline.py --mode train
```

### 3. Run Inference Pipeline

The inference engine automatically detects the 100 anchor samples inside the test file to calculate live market shifts, then calibrates and populates the missing prices.

- **Run Approach 1** (Global Marketplace Model - Recommended):

  ```bash
  python run_pipeline.py --mode infer \
    --input data/raw/ecommerce_price_prediction-train.csv \
    --output data/processed/submission_global.csv \
    --approach global
  ```

- **Run Approach 2** (Shop/Product Level Model):

  ```bash
  python run_pipeline.py --mode infer \
    --input data/raw/ecommerce_price_prediction-train.csv \
    --output data/processed/submission_entity.csv \
    --approach entity
  ```

---

## 📊 Validation Framework & Key Findings

We implemented a strict **Time-Based Validation Split** (isolating the final scraping date, `2025-03-22`, as a simulated outage day) to rigorously test both architectures before evaluating hidden test data.

### Performance Summary

| Approach | Description | MAPE |
|---|---|---|
| **Approach 1** | Global Marketplace Model + Macro Calibration | **0.97%** ✅ |
| **Approach 2** | Shop/Product Level Model + Granular Calibration | 1.43% |

### Critical Engineering Trade-offs & Retrospective

**1. Why the Global Model Outperformed the Entity Model:**

- **The Small-Sample Calibration Trap:** While Approach 2 attempts a highly personalized shop-level calibration, fragmenting the 100 anchor samples across 219 unique shops leaves most entities with only 1–2 reference points. Calculating a median shift factor from such a sparse sample introduces severe variance, causing the model to overfit to micro-level pricing noise on the outage day.

- **Stable Historical Corridors:** Our Exploratory Data Analysis (EDA) revealed that `item_price_min` and `item_price_max` serve as exceptionally strong linear anchors for price predictability. The Global LightGBM model utilizes these boundaries effectively without introducing entity-specific data sparsity noise.

**2. Production Safeguards Implemented:**

- **Numerical Instability Fix:** Items with single variants create a flat denominator boundary (`max - min == 0`), which natively causes standard scaling formulas to blow up to numerical scales of $10^{14}$. Our pipeline applies a safe vectorized clamp to ensure stability.

- **Cold-Start Defense:** For Approach 2, any shop or product encountered during test-time with insufficient history is automatically protected via a fallback hierarchy that routes its reference metrics back to macro global priors.

---

## 📁 Repository Structure

| File | Description |
|---|---|
| `src/data_loader.py` | Handles file ingestion, basic formatting, and high-sparsity column pruning. |
| `src/features.py` | Executes temporal cyclical encoding, pricing boundary indexes, and strict LightGBM categorical casting. |
| `src/train.py` | Fits and serializes both model candidates. |
| `src/inference.py` | Isolates anchor samples dynamically, computes calibration ratios, and fills missing pricing payloads. |
| `run_pipeline.py` | Unified Command-Line Interface (CLI) for production operations. |