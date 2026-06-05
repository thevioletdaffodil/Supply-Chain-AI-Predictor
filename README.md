# Dual-Band AI Supply Chain Lead Time Predictor

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-orange.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

## The Business Problem
In modern manufacturing, supply chain delays result in catastrophic assembly line stockouts, while overly cautious planning results in bloated warehouses and wasted capital. Traditional ERP systems rely on static, human-assigned lead times that fail to account for the erratic, volatile nature of global logistics. 

## The Solution
I engineered an automated machine learning pipeline that discards human schedules entirely. Instead, it generates a **Dual-Band AI Forecast** for bulk Purchase Orders, providing operations teams with two critical calendar dates:
1. **The Expected Date:** The statistically most likely arrival day.
2. **The 95th Percentile Safety Date:** An impenetrable buffer designed to cover extreme logistical volatility without holding excess inventory.

*(Note: To comply with strict data confidentiality, this repository utilizes a 10,000-row synthetic dataset. It perfectly mirrors the mathematical shape, high-cardinality constraints, and noise of the proprietary ERP environment this architecture was originally built for).*

---

## Technical Architecture

### 1. High-Cardinality Volatility Encoding
The dataset contains 351 unique vendors and 125 unique cities. Standard One-Hot Encoding would bloat the dimensionality to unmanageable levels and fail to capture variance. 

This pipeline utilizes dynamic **Target Encoding with Volatility metrics**, assigning both a `mean` (average delay) and a `std` (standard deviation) to every Vendor, City, and Item. The AI mathematically understands the difference between a vendor who is consistently slow and a vendor who is wildly unpredictable. Rare vendors (`count < 3`) are automatically mapped to an 'Other' baseline to prevent overfitting.

### 2. Dual-Band Gradient Boosting Regressors
Instead of a brittle classifier, this pipeline uses parallel regressors to balance accuracy with risk management:
* **Model A (Accuracy Focus):** `GradientBoostingRegressor(loss='squared_error')`. Learns the true average timeline of the logistics chain.
* **Model B (Safety Buffer Focus):** `GradientBoostingRegressor(loss='quantile', alpha=0.95)`. A highly aggressive risk-management model that purposefully targets the 95th percentile of historical disaster scenarios to eliminate stockouts.

### 3. Automated Data-Leakage Protection
The production inference script (`predict_batch_orders.py`) utilizes robust MLOps practices. It automatically intercepts messy, multi-column ERP exports and uses `.reindex(fill_value=0)` to silently strip out human-assigned schedules and irrelevant noise. This guarantees the AI predicts strictly on raw physical parameters (Vendor, Item, Quantity, Date).

---

## 📊 Proof of Value (A/B Test)
By testing the Quantile Model at `alpha=0.85` vs `alpha=0.95`, the architecture successfully isolates risk tuning. 
* For highly unpredictable vendors (e.g., *Quantum Components*), moving the safety dial to 95% triggers a massive inventory buffer to protect the assembly line. 
* For slow-but-consistent vendors (e.g., *Strata Global*), the AI recognizes that adding safety buffer is a waste of warehouse space, keeping inventory holding times exceptionally lean.

## Quick Start
To run the automated batch inference on a raw spreadsheet of unscheduled Purchase Orders:

```python
from predict_batch_orders import predict_batch_orders

# Ingests raw ERP output, drops noise, applies encoding, and calculates calendar dates
predicted_df = predict_batch_orders(
    new_csv_path="unscheduled_orders.csv", 
    output_csv_path="AI_Scheduled_Orders_Output.csv"
)
