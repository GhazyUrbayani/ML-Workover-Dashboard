import os
import sys
import warnings
import json
import logging
import io
import gc
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
# Configure max content length (500MB to support large parquet/CSV files)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# Handle 413 Request Entity Too Large gracefully (return JSON instead of HTML)
from werkzeug.exceptions import RequestEntityTooLarge

@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        "error": "File too large. Maximum allowed size is 500MB.",
        "max_size_mb": 500
    }), 413

# Log all incoming requests
@app.before_request
def log_request():
    print(f"📥 {request.method} {request.path} - Content-Length: {request.content_length}")
    sys.stdout.flush()

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

def process_timeseries_data(df):
    """
    Process time-series production data into well-level features.
    This handles both old CSV format and new Parquet format.
    Generates MA/Diff features required by the model.
    """
    print("   🔄 Standardizing column names...")
    
    # Map alternative column names to standard names
    col_mapping = {
        'OILPROD': 'OIL_PROD', 'WATERPROD': 'WATER_PROD', 'GASPROD': 'GAS_PROD',
        'WATERCUT': 'WATER_CUT', 'DATETIME': 'DATE_TIME',
        'INTVTYPE': 'INTV_TYPE', 'EVALINTVTYPE': 'EVAL_INTV_TYPE',
        'INTERVENTIONFLAG': 'INTERVENTION_FLAG', 'INTERVENTIONSUCCESS': 'INTERVENTION_SUCCESS',
        'INTERVENTIONCOST': 'INTERVENTION_COST', 'SHUTIN': 'SHUT_IN',
        'WELLNAME': 'WELL_NAME', 'WELLTYPE': 'WELL_TYPE',
        'RESERVOIRQUALITY': 'RESERVOIR_QUALITY'
    }
    
    for old_name, new_name in col_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    # Ensure datetime
    if 'DATE_TIME' in df.columns:
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        df = df.sort_values(['WELL_NAME', 'DATE_TIME'])
    
    # Calculate GOR if missing
    if 'GOR' not in df.columns and 'GAS_PROD' in df.columns and 'OIL_PROD' in df.columns:
        df['GOR'] = df['GAS_PROD'] / (df['OIL_PROD'] + 1e-6)
    
    print("   🔄 Generating MA & Diff features...")
    
    # Generate MA and Diff features
    for col in ['OIL_PROD', 'WATER_CUT', 'GOR', 'FBHP']:
        if col in df.columns:
            df[f'{col}_MA7'] = df.groupby('WELL_NAME')[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            df[f'{col}_MA90'] = df.groupby('WELL_NAME')[col].transform(
                lambda x: x.rolling(90, min_periods=1).mean()
            )
            df[f'{col}_DIFF7'] = df.groupby('WELL_NAME')[col].transform(
                lambda x: x.diff(7)
            )
    
    # Calculate cumulative production if missing
    if 'CUM_OIL' not in df.columns and 'OIL_PROD' in df.columns:
        df['CUM_OIL'] = df.groupby('WELL_NAME')['OIL_PROD'].cumsum()
    if 'CUM_WATER' not in df.columns and 'WATER_PROD' in df.columns:
        df['CUM_WATER'] = df.groupby('WELL_NAME')['WATER_PROD'].cumsum()
    if 'CUM_GAS' not in df.columns and 'GAS_PROD' in df.columns:
        df['CUM_GAS'] = df.groupby('WELL_NAME')['GAS_PROD'].cumsum()
    if 'CUM_LIQUID' not in df.columns:
        df['CUM_LIQUID'] = df.get('CUM_OIL', 0) + df.get('CUM_WATER', 0)
    
    # Oil decline rate
    df['OIL_DECLINE_RATE'] = df.groupby('WELL_NAME')['OIL_PROD'].transform(
        lambda x: x.pct_change().rolling(30, min_periods=1).mean()
    ) if 'OIL_PROD' in df.columns else 0
    
    print("   🔄 Aggregating to well-level features...")
    
    def mode_non_none(x):
        x = x[x != "NONE"]
        return x.mode().iat[0] if len(x.mode()) > 0 else "NONE"
    
    def safe_mode(x):
        return x.mode().iat[0] if len(x.mode()) > 0 else "UNKNOWN"
    
    # Build aggregation dict
    agg_dict = {'CUM_OIL': 'max', 'CUM_WATER': 'max', 'CUM_GAS': 'max', 'CUM_LIQUID': 'max'}
    
    # Add optional columns
    optional_aggs = {
        'WATER_CUT': ('WATER_CUT_MEAN', 'mean'),
        'FBHP': ('FBHP_MEAN', 'mean'),
        'FTHP': ('FTHP_MEAN', 'mean'),
        'FBHT': ('FBHT_MEAN', 'mean'),
        'FTHT': ('FTHT_MEAN', 'mean'),
        'POROSITY': ('POROSITY_MEAN', 'mean'),
        'SHUT_IN': ('N_SHUT_IN', 'sum'),
        'INTERVENTION_FLAG': ('N_INTERVENTION', 'sum'),
        'INTERVENTION_COST': ('INTERVENTION_COST_MEAN', 'mean'),
        'INTERVENTION_SUCCESS': ('INTERVENTION_SUCCESS', 'max'),
        'OIL_PROD_MA7': ('OIL_PROD_MA7_MEAN', 'mean'),
        'OIL_PROD_MA90': ('OIL_PROD_MA90_MEAN', 'mean'),
        'WATER_CUT_MA7': ('WATER_CUT_MA7_MEAN', 'mean'),
        'WATER_CUT_MA90': ('WATER_CUT_MA90_MEAN', 'mean'),
        'OIL_PROD_DIFF7': ('OIL_PROD_DIFF7_MEAN', 'mean'),
        'WATER_CUT_DIFF7': ('WATER_CUT_DIFF7_MEAN', 'mean'),
        'GOR_DIFF7': ('GOR_DIFF7_MEAN', 'mean'),
        'FBHP_DIFF7': ('FBHP_DIFF7_MEAN', 'mean'),
        'OIL_DECLINE_RATE': ('OIL_DECLINE_RATE_MEAN', 'mean'),
    }
    
    for col, (new_name, func) in optional_aggs.items():
        if col in df.columns:
            agg_dict[col] = func
    
    well_df = df.groupby('WELL_NAME').agg(agg_dict).reset_index()
    
    # Rename aggregated columns
    rename_map = {col: new_name for col, (new_name, _) in optional_aggs.items() if col in well_df.columns}
    well_df = well_df.rename(columns=rename_map)
    
    # Add categorical columns (mode)
    for cat_col in ['WELL_TYPE', 'RESERVOIR_QUALITY']:
        if cat_col in df.columns:
            cat_mode = df.groupby('WELL_NAME')[cat_col].agg(safe_mode).reset_index()
            well_df = well_df.merge(cat_mode, on='WELL_NAME', how='left')
    
    # Add intervention types
    for intv_col in ['INTV_TYPE', 'EVAL_INTV_TYPE']:
        if intv_col in df.columns:
            intv_mode = df.groupby('WELL_NAME')[intv_col].agg(mode_non_none).reset_index()
            well_df = well_df.merge(intv_mode, on='WELL_NAME', how='left')
    
    # Fill missing numeric columns with defaults
    default_values = {
        'WATER_CUT_MEAN': 0.3, 'FBHP_MEAN': 1500, 'FTHP_MEAN': 200, 'FBHT_MEAN': 180, 'FTHT_MEAN': 100,
        'POROSITY_MEAN': 0.15, 'N_SHUT_IN': 0, 'N_INTERVENTION': 0, 'INTERVENTION_COST_MEAN': 0,
        'OIL_PROD_MA7_MEAN': 0, 'OIL_PROD_MA90_MEAN': 0, 'WATER_CUT_MA7_MEAN': 0, 'WATER_CUT_MA90_MEAN': 0,
        'OIL_PROD_DIFF7_MEAN': 0, 'WATER_CUT_DIFF7_MEAN': 0, 'GOR_DIFF7_MEAN': 0, 'FBHP_DIFF7_MEAN': 0,
        'OIL_DECLINE_RATE_MEAN': 0, 'PERM_LOG_MEAN': 100, 'NET_PAY_FROM_LOG': 10,
        'PHIE_MEAN': 0.15, 'SW_MEAN': 0.3, 'VCLGR_MEAN': 0.2, 'PAYFLAG_RATIO': 0.5, 'RESFLAG_RATIO': 0.5
    }
    
    for col, default in default_values.items():
        if col not in well_df.columns:
            well_df[col] = default
    
    # Fill missing categorical
    if 'WELL_TYPE' not in well_df.columns:
        well_df['WELL_TYPE'] = 'PRODUCER'
    if 'RESERVOIR_QUALITY' not in well_df.columns:
        well_df['RESERVOIR_QUALITY'] = 'MEDIUM'
    
    return well_df

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

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
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
    print("📥 POST /api/predict - START")
    print(f"   Content-Type: {request.content_type}")
    print(f"   Content-Length: {request.content_length}")
    sys.stdout.flush()
    
    if model is None:
        print("❌ Model is None - not loaded")
        sys.stdout.flush()
        return jsonify({"error": "Model not loaded. Server running in STATIC MODE due to scikit-learn version mismatch."}), 503
    
    try:
        # Check if file uploaded
        print("📋 Checking for file in request.files...")
        sys.stdout.flush()
        
        try:
            has_file = 'file' in request.files
        except RequestEntityTooLarge:
            print("❌ File too large for server to process")
            sys.stdout.flush()
            return jsonify({"error": "File too large. Maximum allowed size is 500MB. Try reducing the file or splitting into smaller files."}), 413
        
        if not has_file:
            print(f"❌ No 'file' in request.files. Keys: {list(request.files.keys())}")
            sys.stdout.flush()
            return jsonify({"error": "No file uploaded. Send CSV with 'file' field."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        print(f"📄 File received: {file.filename}")
        
        # Read file based on extension (CSV or Parquet)
        filename = file.filename.lower()
        if filename.endswith('.parquet'):
            # Read Parquet file - memory efficient: read into bytes then free upload
            raw_bytes = file.read()
            file_bytes = io.BytesIO(raw_bytes)
            del raw_bytes
            gc.collect()
            df = pd.read_parquet(file_bytes, engine='pyarrow')
            del file_bytes
            gc.collect()
            print(f"📊 Received Parquet: {len(df)} rows, {len(df.columns)} columns")
        elif filename.endswith('.csv'):
            # Read CSV file with low_memory mode
            df = pd.read_csv(file, low_memory=True)
            print(f"📊 Received CSV: {len(df)} rows, {len(df.columns)} columns")
        else:
            return jsonify({"error": "Unsupported file format. Use .csv or .parquet"}), 400
        
        # Memory check: log approximate size
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"💾 DataFrame memory: {mem_mb:.1f} MB")
        sys.stdout.flush()
        
        # If DataFrame is very large, downsample numeric columns to float32
        if mem_mb > 100:
            print("   ⚡ Optimizing memory: downcasting float64 → float32")
            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = df[float_cols].astype(np.float32)
            gc.collect()
            mem_mb_after = df.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"   💾 After optimization: {mem_mb_after:.1f} MB")
        
        # Standardize column names (handle both formats)
        df.columns = df.columns.str.upper().str.replace(' ', '_')
        print(f"📋 Columns after standardization: {list(df.columns)[:10]}...")
        
        # Check if this is time-series data (needs aggregation) or well-level data
        is_timeseries = 'DATE_TIME' in df.columns or 'DATETIME' in df.columns
        
        if is_timeseries:
            print("📈 Detected time-series data - performing feature engineering...")
            df = process_timeseries_data(df)
            print(f"✅ Aggregated to {len(df)} wells")
        
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
        
        # Free large objects before building response
        del X, y_prob, y_pred, df
        gc.collect()
        
        response_data = {
            "success": True,
            "summary": summary,
            "predictions": results
        }
        print(f"✅ Prediction complete: {len(results)} wells")
        sys.stdout.flush()
        return jsonify(response_data)
        
    except RequestEntityTooLarge:
        return jsonify({"error": "File too large. Maximum allowed size is 500MB."}), 413
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
