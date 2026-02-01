import os
import sys
import warnings
import json
import logging
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
CORS(app)  # Enable CORS for all routes

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

@app.route('/api/predict', methods=['POST'])
def predict():
    print("📥 Received POST /api/predict request")
    
    if model is None:
        print("❌ Model is None - not loaded")
        return jsonify({"error": "Model not loaded. Server running in STATIC MODE due to scikit-learn version mismatch."}), 503
    
    try:
        # Check if file uploaded
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({"error": "No file uploaded. Send CSV with 'file' field."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        print(f"📄 File received: {file.filename}")
        
        # Read CSV
        df = pd.read_csv(file)
        print(f"📊 Received CSV: {len(df)} rows, {len(df.columns)} columns")
        
        # Validate required columns
        required_cols = ['CUM_OIL', 'CUM_WATER', 'WELL_NAME']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400
        
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
        
        # Predict
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
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
            "avg_success_prob": round(np.mean(y_prob), 3),
            "hetero_distribution": {
                "q1": sum(1 for r in results if r['hetero_index'] == 1),
                "q2": sum(1 for r in results if r['hetero_index'] == 2),
                "q3": sum(1 for r in results if r['hetero_index'] == 3),
                "q4": sum(1 for r in results if r['hetero_index'] == 4),
            }
        }
        
        return jsonify({
            "success": True,
            "summary": summary,
            "predictions": results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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
