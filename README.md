# 🛢️ ML Workover Optimization Dashboard

## IPFEST 2026 - Hengker Berkelas Team

Dashboard prediksi keberhasilan workover sumur minyak menggunakan Machine Learning (LightGBM).

---

## 📁 Struktur File

```
Downloads/
├── index.html                      # Dashboard frontend
├── main.py                         # Flask API server (lokal)
├── requirements.txt                # Python dependencies
├── vercel.json                     # Vercel deployment config
├── api/
│   └── index.py                    # Vercel serverless function
├── workover_model_pipeline.joblib  # Model ML (dari notebook)
├── preprocessor.joblib             # Preprocessor (dari notebook)
├── dashboard_data.json             # Data dashboard (dari notebook)
└── ipfestproto NEW.ipynb           # Jupyter notebook
```

---

## Link URL web App

https://ml-workover-dashboard.onrender.com/

## 🚀 Cara Test di Lokal

### Step 1: Install Dependencies

```bash
cd C:\Users\asus\Downloads
pip install -r requirements.txt
```

### Step 2: Jalankan Notebook untuk Generate Model

1. Buka `ipfestproto NEW.ipynb` di Jupyter/VS Code
2. Run semua cell sampai cell terakhir (Export Model)
3. Pastikan file berikut terbuat:
   - `workover_model_pipeline.joblib`
   - `preprocessor.joblib`
   - `dashboard_data.json`

### Step 3: Jalankan Flask Server

```bash
python main.py
```
