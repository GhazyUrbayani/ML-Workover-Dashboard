import os
import sys
import warnings
import json
import logging
import threading
import uuid
import time
import io

#Suppress all warnings
os.environ['LIGHTGBM_SUPPRESS_WARNINGS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.CRITICAL)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.columns] = np.log1p(X[self.columns].clip(lower=0))
        return X
    
    def set_output(self, transform=None):
        return self
    
    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.columns

app = Flask(__name__, static_folder='.')
# Configure max content length (50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# ============== ASYNC JOB QUEUE ==============
# Store job results in memory (will reset on server restart)
JOBS = {}

def cleanup_old_jobs():
    """Remove jobs older than 10 minutes"""
    now = time.time()
    old_jobs = [jid for jid, job in JOBS.items() if now - job.get('created_at', now) > 600]
    for jid in old_jobs:
        del JOBS[jid]
        print(f"🗑️ Cleaned up old job: {jid}")

# Log all incoming requests
@app.before_request
def log_request():
    print(f"📥 {request.method} {request.path} - Content-Length: {request.content_length}")
    sys.stdout.flush()
    cleanup_old_jobs()  # Cleanup on each request

#LOAD MODEL & PREPROCESSOR

MODEL_PATH = 'workover_model_pipeline.joblib'
PREPROCESSOR_PATH = 'preprocessor.joblib'
DASHBOARD_DATA_PATH = 'dashboard_data.json'

model = None
preprocessor = None
dashboard_data = None

def load_model():
    #Load the trained model pipeline#
    global model, preprocessor, dashboard_data
    
    # Load model with error handling for version mismatch
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded: {MODEL_PATH}")
        except (AttributeError, ModuleNotFoundError) as e:
            print(f"⚠️ Model version mismatch: {e}")
            print("   Model was trained with different scikit-learn version.")
            print("   Dashboard will run in STATIC MODE (no predictions).")
            model = None
    else:
        print(f"⚠️ Model not found: {MODEL_PATH}")
        print("   Run the notebook first to generate the model!")
    
    # Load preprocessor with error handling
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print(f"✅ Preprocessor loaded: {PREPROCESSOR_PATH}")
        except (AttributeError, ModuleNotFoundError) as e:
            print(f"⚠️ Preprocessor version mismatch: {e}")
            preprocessor = None
    
    # Load dashboard data (JSON - no version issues)
    if os.path.exists(DASHBOARD_DATA_PATH):
        with open(DASHBOARD_DATA_PATH, 'r') as f:
            dashboard_data = json.load(f)
        print(f"✅ Dashboard data loaded: {DASHBOARD_DATA_PATH}")

#INTERVENTION COST MAPPING

INTERVENTION_COSTS = {
    "REPERFORASI": 100000,
    "WATER_SHUTOFF": 140000,
    "STIMULASI": 200000,
    "ARTIFICIAL_LIFT": 275000,
    "WELLBORE_CLEANOUT": 70000,
    "ZONAL_ISOLATION": 235000,
    "NONE": 0,
}

#API ROUTES


@app.route('/')
def serve_index():
    #Serve the main dashboard HTML#
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    #Serve static files#
    return send_from_directory('.', filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    #Health check endpoint#
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "dashboard_data_loaded": dashboard_data is not None,
        "mode": "FULL" if model is not None else "STATIC (no predictions)"
    })

@app.route('/api/dashboard-data', methods=['GET'])
def get_dashboard_data():
    #Return pre-computed dashboard data from JSON#
    if dashboard_data:
        return jsonify(dashboard_data)
    else:
        return jsonify({"error": "Dashboard data not found. Run notebook first!"}), 404

# ============== BACKGROUND PREDICTION WORKER ==============
def _run_prediction_bg(job_id, csv_content, filename):
    """Background worker to run prediction without blocking HTTP request"""
    global JOBS
    
    try:
        print(f"🔄 Job {job_id}: Starting prediction...")
        JOBS[job_id]['status'] = 'processing'
        JOBS[job_id]['progress'] = 10
        sys.stdout.flush()
        
        # Read CSV from bytes
        df = pd.read_csv(io.StringIO(csv_content))
        print(f"📊 Job {job_id}: Received CSV: {len(df)} rows, {len(df.columns)} columns")
        JOBS[job_id]['progress'] = 20
        sys.stdout.flush()
        
        # Validate required columns
        required_cols = ['CUM_OIL', 'CUM_WATER', 'WELL_NAME']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = f"Missing columns: {missing}"
            return
        
        JOBS[job_id]['progress'] = 30
        
        # Calculate Heterogeneity Index if not present
        if 'HETERO_INDEX' not in df.columns:
            oil_median = df['CUM_OIL'].median()
            water_median = df['CUM_WATER'].median()
            
            def calc_hetero(row):
                high_oil = row['CUM_OIL'] >= oil_median
                high_water = row['CUM_WATER'] >= water_median
                if high_oil and not high_water:
                    return 1  # High Oil - Low Water
                elif high_oil and high_water:
                    return 2  # High Oil - High Water
                elif not high_oil and not high_water:
                    return 3  # Low Oil - Low Water
                else:
                    return 4  # Low Oil - High Water
            
            df['HETERO_INDEX'] = df.apply(calc_hetero, axis=1)
        
        JOBS[job_id]['progress'] = 40
        
        # Prepare features for prediction
        well_names = df['WELL_NAME'].values
        
        # Get feature columns (same as training)
        numeric_log_cols = ['CUM_OIL', 'CUM_WATER', 'CUM_GAS', 'CUM_LIQUID', 'PERM_LOG_MEAN', 'INTERVENTION_COST_MEAN']
        ratio_cols = ['WATER_CUT_MEAN', 'POROSITY_MEAN', 'PAYFLAG_RATIO', 'RESFLAG_RATIO']
        phys_cols = ['FBHP_MEAN', 'FTHP_MEAN', 'FBHT_MEAN', 'FTHT_MEAN', 'NET_PAY_FROM_LOG']
        ma_diff_cols = [
            'OIL_PROD_MA7_MEAN', 'OIL_PROD_MA90_MEAN', 'WATER_CUT_MA7_MEAN', 'WATER_CUT_MA90_MEAN',
            'OIL_PROD_DIFF7_MEAN', 'WATER_CUT_DIFF7_MEAN', 'GOR_DIFF7_MEAN', 'FBHP_DIFF7_MEAN',
            'OIL_DECLINE_RATE_MEAN', 'N_SHUT_IN', 'N_INTERVENTION', 'PHIE_MEAN', 'SW_MEAN', 'VCLGR_MEAN'
        ]
        cat_cols = ['WELL_TYPE', 'RESERVOIR_QUALITY', 'HETERO_INDEX']
        
        all_cols = numeric_log_cols + ratio_cols + phys_cols + ma_diff_cols + cat_cols
        
        JOBS[job_id]['progress'] = 50
        
        # Fill missing columns with default values
        for col in all_cols:
            if col not in df.columns:
                if col in cat_cols:
                    df[col] = 'UNKNOWN'
                else:
                    df[col] = 0.0
        
        # Convert categorical columns to string
        for col in cat_cols:
            df[col] = df[col].astype(str)
        
        # Handle infinities and NaN
        X = df[all_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        JOBS[job_id]['progress'] = 60
        print(f"🤖 Job {job_id}: Running ML prediction...")
        sys.stdout.flush()
        
        # Predict (this is the slow part)
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        JOBS[job_id]['progress'] = 80
        print(f"✅ Job {job_id}: Prediction complete, building results...")
        sys.stdout.flush()
        
        # Create results
        results = []
        for i, (well, prob, pred, hetero) in enumerate(zip(well_names, y_prob, y_pred, df['HETERO_INDEX'])):
            # Advisory rule
            if prob >= 0.8:
                advisory = "Strongly Recommend"
            elif prob >= 0.6:
                advisory = "Review by Engineer"
            else:
                advisory = "Low Priority"
            
            # Hetero label
            hetero_labels = {
                '1': "High Oil - Low Water",
                '2': "High Oil - High Water",
                '3': "Low Oil - Low Water",
                '4': "Low Oil - High Water"
            }
            
            results.append({
                "rank": i + 1,
                "well_name": str(well),
                "success_prob": round(float(prob), 4),
                "success_class": int(pred),
                "hetero_index": int(hetero) if str(hetero).isdigit() else 0,
                "hetero_label": hetero_labels.get(str(hetero), "Unknown"),
                "advisory": advisory,
                "estimated_roi": round(float(prob) * 150000 - 50000, 0)
            })
        
        JOBS[job_id]['progress'] = 90
        
        # Sort by probability
        results = sorted(results, key=lambda x: x['success_prob'], reverse=True)
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        # Calculate summary stats
        summary = {
            "total_wells": len(results),
            "strongly_recommend": sum(1 for r in results if r['advisory'] == "Strongly Recommend"),
            "review_engineer": sum(1 for r in results if r['advisory'] == "Review by Engineer"),
            "low_priority": sum(1 for r in results if r['advisory'] == "Low Priority"),
            "avg_success_prob": round(float(np.mean(y_prob)), 3),
            "hetero_distribution": {
                "q1": sum(1 for r in results if r['hetero_index'] == 1),
                "q2": sum(1 for r in results if r['hetero_index'] == 2),
                "q3": sum(1 for r in results if r['hetero_index'] == 3),
                "q4": sum(1 for r in results if r['hetero_index'] == 4),
            }
        }
        
        # Save result
        JOBS[job_id]['status'] = 'complete'
        JOBS[job_id]['progress'] = 100
        JOBS[job_id]['result'] = {
            "success": True,
            "summary": summary,
            "predictions": results
        }
        print(f"🎉 Job {job_id}: COMPLETE - {len(results)} wells processed")
        sys.stdout.flush()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        JOBS[job_id]['status'] = 'error'
        JOBS[job_id]['error'] = str(e)
        print(f"❌ Job {job_id}: ERROR - {str(e)}")
        sys.stdout.flush()


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Submit prediction job - returns immediately with job_id"""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        print("✅ OPTIONS preflight for /api/predict")
        sys.stdout.flush()
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    print("="*50)
    print("📥 POST /api/predict - ASYNC JOB SUBMISSION")
    print(f"   Content-Type: {request.content_type}")
    print(f"   Content-Length: {request.content_length}")
    sys.stdout.flush()
    
    if model is None:
        print("❌ Model is None - not loaded")
        sys.stdout.flush()
        return jsonify({"error": "Model not loaded. Server running in STATIC MODE due to scikit-learn version mismatch."}), 503
    
    try:
        # Check if file uploaded
        if 'file' not in request.files:
            print(f"❌ No 'file' in request.files. Keys: {list(request.files.keys())}")
            sys.stdout.flush()
            return jsonify({"error": "No file uploaded. Send CSV with 'file' field."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        print(f"📄 File received: {file.filename}")
        
        # Read file content into memory (so we can close request quickly)
        csv_content = file.read().decode('utf-8')
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())[:8]
        
        # Create job entry
        JOBS[job_id] = {
            'status': 'queued',
            'progress': 0,
            'filename': file.filename,
            'created_at': time.time(),
            'result': None,
            'error': None
        }
        
        # Start background thread
        thread = threading.Thread(
            target=_run_prediction_bg,
            args=(job_id, csv_content, file.filename),
            daemon=True
        )
        thread.start()
        
        print(f"✅ Job {job_id} submitted - returning immediately")
        sys.stdout.flush()
        
        # Return immediately (within 1 second!) 
        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "message": "Prediction job submitted. Poll /api/predict/{job_id} for status."
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/<job_id>', methods=['GET', 'OPTIONS'])
def get_job_status(job_id):
    """Poll job status - returns progress or final result"""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response, 200
    
    if job_id not in JOBS:
        return jsonify({
            "error": f"Job {job_id} not found. It may have expired."
        }), 404
    
    job = JOBS[job_id]
    
    response_data = {
        "job_id": job_id,
        "status": job['status'],
        "progress": job['progress'],
        "filename": job.get('filename', 'unknown')
    }
    
    if job['status'] == 'complete':
        response_data['result'] = job['result']
        # Keep job for 5 more minutes then cleanup will remove it
        
    elif job['status'] == 'error':
        response_data['error'] = job['error']
    
    return jsonify(response_data)

@app.route('/api/predict-single', methods=['POST'])
def predict_single():
    #Predict for a single well from JSON data#
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        return jsonify({
            "success": True,
            "message": "Use /api/predict with CSV for full predictions"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#MAIN ENTRY POINT

# Load model at module level (required for gunicorn)
load_model()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛢️  ML WORKOVER OPTIMIZATION API SERVER")
    print("="*60)
    
    print("\n📡 Starting server...")
    print("   Local:   http://127.0.0.1:5000")
    print("   Network: http://0.0.0.0:5000")
    print("\n🌐 Open index.html in browser or visit http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)
