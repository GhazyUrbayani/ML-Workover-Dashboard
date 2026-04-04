<div align="center">

  <h1>🛢️ ML Workover Optimization Dashboard</h1>

  <p>
    <img src="https://img.shields.io/badge/🏆_Award-1st_Place_International_Hackathon-gold?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Competition-IPFEST_2026-blue?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Domain-Petroleum_Engineering-orange?style=for-the-badge" />
  </p>

  <p>
    <img src="https://img.shields.io/badge/Model-LightGBM-lightgreen?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/ROC--AUC-81.9%25-success?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Wells_Analyzed-300-informational?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Cost_Savings-40.9%25_($10.6M)-brightgreen?style=for-the-badge" />
  </p>

  <p>
    <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" />
    <img src="https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white" />
    <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
    <img src="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white" />
  </p>

</div>

---

## 📖 About the Project

An **ML-powered interactive dashboard** for petroleum workover optimization, built for the **IPFEST 2026 competition** — and awarded **1st Place International Hackathon** in the Petroleum Competition track.

The dashboard uses a **LightGBM classifier** to predict which oil wells require workover intervention, enabling engineers to prioritize operations, reduce costs, and increase production uplift. The Power BI–style UI is built entirely in vanilla HTML + Chart.js, deployed on Vercel.

> 🔗 **Live Demo:** [ml-workover-dashboard.vercel.app](https://ml-workover-dashboard.vercel.app)
> 
> 📓 **Kaggle Notebook:** [kaggle.com/code/ghazyachmed/ipfestproto](https://www.kaggle.com/code/ghazyachmed/ipfestproto)

---

## ✨ Key Features

- **🤖 ML Prediction Engine** — LightGBM classifier trained on 300 oil wells (2015–2017 field data)
- **📊 4-Panel Interactive Dashboard** — Power BI–style layout with real-time interactivity
- **🔍 Dynamic Filters (Slicer Style)** — Filter by Heterogeneity Index, Advisory type, and Success Rate
- **💰 Cost Optimization Analysis** — Baseline vs. ML-Optimized cost breakdown with savings highlight
- **🔬 Heterogeneity Index Quadrant** — 4-quadrant well classification (Oil vs. Water production)
- **📈 Production Lifecycle Chart** — Before & After Workover trend visualization (28 time periods)
- **✅ Model Performance Metrics** — Live display of Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **🗂️ Confusion Matrix Visualization** — Horizontal bar chart with TN/FP/FN/TP breakdown
- **💡 Power BI Tooltips** — Hover any card/quadrant for detailed contextual insights
- **📤 JSON Export** — Export full dashboard state as structured JSON
- **📁 CSV/Parquet Upload** — Run live predictions on custom well datasets

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| **ROC-AUC** | 81.9% |
| **Accuracy** | 78.3% |
| **Precision** | 76.7% |
| **Recall** | 76.7% |
| **F1-Score** | 76.7% |
| **Model** | LightGBM |
| **Train/Test Split** | 80% / 20% (stratified) |
| **Total Wells** | 300 |
| **Test Wells** | 60 |

---

## 💰 Business Impact

| Cost Metric | Value |
|---|---|
| **Baseline Total Cost** | $26.0M |
| **ML-Optimized Cost** | $15.4M |
| **Total Savings** | **$10.6M (40.9%)** |
| **Savings per Well** | ~$35.5K |
| **Est. ROI Period** | 8 months |
| **Production Uplift** | +21% (2850 → 3450 BOPD) |

---

## 🔬 Heterogeneity Index (4-Quadrant Classification)

| Quadrant | Classification | Wells | Success Rate | Action |
|---|---|---|---|---|
| **Q4** | 🟢 High Oil – Low Water | 37 | 81% | Best candidates |
| **Q3** | 🟡 High Oil – High Water | 113 | 79% | Monitor water cut |
| **Q2** | 🔵 Low Oil – Low Water | 113 | 77% | Stimulate production |
| **Q1** | 🔴 Low Oil – High Water | 37 | 73% | High risk – review |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **ML Model** | LightGBM (Python) |
| **Dashboard UI** | Vanilla HTML5 + CSS Grid |
| **Charts** | Chart.js 4.4.1 |
| **Backend API** | Flask (Python) |
| **Deployment** | Vercel |
| **Data Format** | CSV / Parquet |
| **Export** | JSON |

---

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.8
pip install flask lightgbm pandas numpy
```

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/GhazyUrbayani/ML-Workover-Dashboard.git
cd ML-Workover-Dashboard

# 2. Open dashboard directly (static mode)
open index.html

# 3. Or run with Flask backend (live prediction mode)
python app.py
# Then visit http://localhost:5000
```

### Using Live Prediction
1. Open the dashboard
2. Upload a `.csv` or `.parquet` file of well data
3. Click **Run Prediction** — results update in real-time
4. Use **Filters** to drill down by quadrant, advisory, or success rate
5. Click **Export JSON** to download the full dashboard state

---

## 📁 Project Structure
├── index.html # Full dashboard UI (HTML + JS + CSS) <br>
├── vercel.json # Vercel deployment config <br>
└── .gitignore


---

## 🏆 Competition

- **Event:** IPFEST 2026 — International Petroleum Competition  
- **Track:** Machine Learning / Data Science  
- **Result:** 🥇 **1st Place (International)**  
- **Kaggle Notebook:** [ipfestproto](https://www.kaggle.com/code/ghazyachmed/ipfestproto)

---

<div align="center">
  <p>Made with ❤️ by <a href="https://github.com/GhazyUrbayani">GhazyUrbayani</a></p>
  <a href="https://www.linkedin.com/in/ghazyurbayani">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
  <a href="https://kaggle.com/ghazyachmed">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
  </a>
</div>
