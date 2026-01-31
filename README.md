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

Output yang diharapkan:

```
============================================================
🛢️  ML WORKOVER OPTIMIZATION API SERVER
============================================================
✅ Model loaded: workover_model_pipeline.joblib
✅ Preprocessor loaded: preprocessor.joblib
✅ Dashboard data loaded: dashboard_data.json

📡 Starting server...
   Local:   http://127.0.0.1:5000
   Network: http://0.0.0.0:5000

🌐 Open index.html in browser or visit http://127.0.0.1:5000
============================================================
```

### Step 4: Buka Dashboard

Pilih salah satu:

- **Opsi A**: Buka `http://127.0.0.1:5000` di browser
- **Opsi B**: Double-click `index.html` (API status akan offline, tapi dashboard tetap jalan dengan sample data)

---

## 🌐 Deploy ke Vercel

### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

### Step 2: Login ke Vercel

```bash
vercel login
```

### Step 3: Deploy

```bash
cd C:\Users\asus\Downloads
vercel --prod
```

### Step 4: Akses Dashboard

Vercel akan memberikan URL seperti:

```
https://your-project.vercel.app
```

---

## 📡 API Endpoints

| Endpoint              | Method | Description            |
| --------------------- | ------ | ---------------------- |
| `/api/health`         | GET    | Cek status API & model |
| `/api/dashboard-data` | GET    | Ambil data dashboard   |
| `/api/predict`        | POST   | Prediksi dari file CSV |

### Contoh Request Predict

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "file=@well_data.csv"
```

Response:

```json
{
  "success": true,
  "summary": {
    "total_wells": 100,
    "strongly_recommend": 45,
    "review_engineer": 30,
    "low_priority": 25
  },
  "predictions": [
    {
      "rank": 1,
      "well_name": "WELL_006",
      "success_prob": 0.989,
      "advisory": "Strongly Recommend"
    }
  ]
}
```

---

## 📋 Format CSV untuk Prediksi

File CSV harus memiliki kolom-kolom berikut (minimal):

| Column              | Description               | Required |
| ------------------- | ------------------------- | -------- |
| `WELL_NAME`         | Nama sumur                | ✅ Yes   |
| `CUM_OIL`           | Kumulatif produksi minyak | ✅ Yes   |
| `CUM_WATER`         | Kumulatif produksi air    | ✅ Yes   |
| `CUM_GAS`           | Kumulatif produksi gas    | Optional |
| `WATER_CUT_MEAN`    | Rata-rata water cut       | Optional |
| `POROSITY_MEAN`     | Rata-rata porositas       | Optional |
| `WELL_TYPE`         | Tipe sumur                | Optional |
| `RESERVOIR_QUALITY` | Kualitas reservoir        | Optional |

> Kolom yang tidak ada akan di-fill dengan nilai default.

---

## 🎯 Fitur Dashboard

1. **Workover Activities Overview** - Statistik total sumur
2. **Model Performance** - ROC AUC, Accuracy, Precision, Recall
3. **Cost Optimization** - Baseline vs Optimized cost
4. **Production Decline Curve** - Grafik decline → intervention → recovery
5. **Heterogeneity Index 4-Quadrant** - Klasifikasi sumur berdasarkan Oil/Water
6. **ROC Curve** - Visualisasi performa model
7. **Confusion Matrix** - TP, TN, FP, FN
8. **Advisory Distribution** - Strongly Recommend / Review / Low Priority
9. **Top 15 Recommendations** - Tabel rekomendasi sumur prioritas
10. **CSV Upload** - Upload file baru untuk prediksi realtime

---

## 🔧 Troubleshooting

### Model tidak terload

```
⚠️ Model not found: workover_model_pipeline.joblib
```

**Solusi**: Jalankan notebook sampai cell terakhir untuk generate file `.joblib`

### CORS Error di browser

```
Access-Control-Allow-Origin error
```

**Solusi**: Pastikan Flask server running dan buka via `http://127.0.0.1:5000`

### Vercel deploy gagal

```
Error: Cannot find module
```

**Solusi**: Pastikan `requirements.txt` ada di root folder

---

## 👥 Tim Hengker Berkelas

IPFEST 2026 Hackathon - Well Intervention Optimization

---

## 📄 License

MIT License © 2026
