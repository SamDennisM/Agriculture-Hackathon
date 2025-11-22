# 🌾 **Farm Input Optimization Console — High-Fidelity Prototype**
> **Data-Driven Recommendations • Budget-aware • Environmental Constraints**

<p align="center">
  <img src="https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-Ridge%20Regression-16a34a" />
  <img src="https://img.shields.io/badge/Focus-Farm%20Input%20Optimization-blueviolet" />
  <img src="https://img.shields.io/badge/Output-CSV%20Recommendations-orange" />
</p>

---
## **TeamMates**

- Sam Dennis M 2448544 
- Jewel Joshy 2448520  
- Samyuktaa 2448545 
- Vyshnavi Katherine 2448558

---

## ✨ **Overview**

**Farm Input Optimization Console** is a polished Streamlit app that generates **data-driven recommendations** for fertilizer, irrigation, and pesticide use while balancing **predicted yield**, **per-hectare budget**, and **environmental impact**.

The pipeline covers:
- Data ingestion & cleaning  
- Statistical generation of synthetic inputs (fertilizer_kg_per_ha, irrigation_mm, pesticide_l)  
- Model training (Ridge regression) and evaluation  
- Explainability (coefficients, optional SHAP)  
- Rule-based optimization (grid search of multipliers under budget & environmental constraints)  
- Downloadable CSV of recommended input plans

This prototype is ideal for hackathons, pilots, or as a reference for production implementations.

---

## 🔗 **Project Resources**

### 🧾 Streamlit app (local file)
`/mnt/data/streamlit_app.py`

> Use the local file above to run or inspect the app. (If you deployed the app, replace with the public URL.)

---

## 🚀 **Key Features**

- 🧪 **Synthetic Input Generation**  
  Automatically derives fertilizer, irrigation, and pesticide proxies from dataset heuristics (soil, weather, env_score).

- 🧰 **Preprocessing & Modeling**  
  Median imputation, winsorization, seasonal features (month sin/cos), standardized numeric features, Ridge regression for robust baseline modeling.

- 🔍 **Explainability**  
  Top coefficients & PDP-like plots; optional SHAP integration if `shap` is installed.

- ⚖️ **Rule-based Optimization**  
  Grid search over multiplier sets (e.g., 0.8, 1.0, 1.2) per-record, enforcing budget and environmental limits; produces before/after plans.

- 📥 **Downloadable Recommendations**  
  CSV export with baseline vs recommended inputs, estimated costs, and predicted yield improvement.

---

## 🎯 **Design Principles Applied**

### **1️⃣ Interpretability**
Recommendations are rule-based (transparent multipliers) and accompanied by feature-coefficient explanations.

### **2️⃣ Business-signal Driven**
Budget and environmental constraints are first-class — the optimizer respects per-hectare budgets and avoids raising inputs where environmental_score is already high.

### **3️⃣ Practical & Lightweight**
Avoids heavy solvers by default; pragmatic grid search is fast and explainable. Option to swap in LP/nonlinear solvers for higher-fidelity optimization.

---

## 📱 **Core Screens / Sections (Streamlit)**

- Dataset preview & column list  
- Synthetic inputs snapshot (fertilizer / irrigation / pesticide / cost)  
- Data quality & missingness report  
- Model training & performance (R², RMSE)  
- Explainability (coefficients, optional SHAP, PDP-style plots)  
- Optimization & Recommendations (generate CSV & metrics)

---

## 🔢 **Optimization Logic (Summary)**

- Start from synthetic inputs for each record.  
- Try small multiplier grid (e.g., `[0.8, 1.0, 1.2]`) on fertilizer/irrigation/pesticide.  
- Recompute per-record cost using unit prices (example: ₹15/kg fertilizer, ₹10/mm irrigation, ₹500/L pesticide).  
- Keep only combos that meet `budget_limit` and `env_threshold` constraints.  
- Choose the combo that maximizes model-predicted yield (or baseline if none feasible).

---

## 📊 **Key Visuals to Include in Presentations**

- **Feature Importance / Coefficient Bar Chart** — highlights which inputs/features drive yield.  
- **PDP-like curves** — show marginal effect of fertilizer/irrigation/pesticide.  
- **Before vs After Optimization** — metrics (avg baseline yield, avg recommended yield, avg improvement).  
- **Top Improvements Chart** — per-record predicted yield improvement sorted descending.

---


## ⚙️ **How to Run (Local)**

```bash
# 1. (Optional) create a virtual env
python -m venv venv
# mac / linux
source venv/bin/activate
# windows (powershell)
venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install streamlit pandas numpy scikit-learn altair

# 3. Run the Streamlit app (local file)
streamlit run /mnt/data/streamlit_app.py


