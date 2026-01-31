"""
Vercel Serverless Function for ML Workover Optimization
Deploy to Vercel: vercel --prod
"""

import os
import sys
import json
import warnings

# Suppress warnings
os.environ['LIGHTGBM_SUPPRESS_WARNINGS'] = '1'
warnings.filterwarnings('ignore')

from http.server import BaseHTTPRequestHandler
import pandas as pd
import numpy as np

# Try to import joblib and load model
try:
    import joblib
    MODEL_LOADED = True
except ImportError:
    MODEL_LOADED = False

# Global model (loaded once per cold start)
model = None
dashboard_data = None

def load_resources():
    """Load model and dashboard data"""
    global model, dashboard_data
    
    # Try multiple paths for Vercel deployment
    model_paths = [
        'workover_model_pipeline.joblib',
        '../workover_model_pipeline.joblib',
        '/var/task/workover_model_pipeline.joblib'
    ]
    
    dashboard_paths = [
        'dashboard_data.json',
        '../dashboard_data.json',
        '/var/task/dashboard_data.json'
    ]
    
    # Load model
    if MODEL_LOADED:
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = joblib.load(path)
                    print(f"Model loaded from: {path}")
                    break
                except Exception as e:
                    print(f"Error loading model from {path}: {e}")
    
    # Load dashboard data
    for path in dashboard_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    dashboard_data = json.load(f)
                print(f"Dashboard data loaded from: {path}")
                break
            except Exception as e:
                print(f"Error loading dashboard data from {path}: {e}")

# Load resources on cold start
load_resources()

class handler(BaseHTTPRequestHandler):
    """Vercel Serverless Function Handler"""
    
    def do_GET(self):
        """Handle GET requests"""
        
        # Parse path
        path = self.path.split('?')[0]
        
        if path == '/api' or path == '/api/':
            self.send_json_response({
                "status": "ok",
                "message": "ML Workover Optimization API",
                "endpoints": {
                    "GET /api": "This help message",
                    "GET /api/health": "Health check",
                    "GET /api/dashboard-data": "Get dashboard data",
                    "POST /api/predict": "Predict from CSV file"
                }
            })
        
        elif path == '/api/health':
            self.send_json_response({
                "status": "ok",
                "model_loaded": model is not None,
                "dashboard_data_loaded": dashboard_data is not None,
                "python_version": sys.version
            })
        
        elif path == '/api/dashboard-data':
            if dashboard_data:
                self.send_json_response(dashboard_data)
            else:
                # Return sample data if no file
                self.send_json_response(self.get_sample_dashboard_data())
        
        else:
            self.send_json_response({"error": "Not found"}, status=404)
    
    def do_POST(self):
        """Handle POST requests"""
        
        path = self.path.split('?')[0]
        
        if path == '/api/predict':
            self.handle_predict()
        else:
            self.send_json_response({"error": "Not found"}, status=404)
    
    def handle_predict(self):
        """Handle prediction request"""
        
        if model is None:
            self.send_json_response({
                "error": "Model not loaded. Deploy with model file or use sample data.",
                "sample_data": True
            }, status=500)
            return
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            # Try to parse as JSON
            try:
                data = json.loads(body.decode('utf-8'))
                df = pd.DataFrame(data if isinstance(data, list) else [data])
            except:
                self.send_json_response({
                    "error": "Invalid JSON. Send array of well data."
                }, status=400)
                return
            
            # Validate
            if 'WELL_NAME' not in df.columns:
                df['WELL_NAME'] = [f"WELL_{i}" for i in range(len(df))]
            
            # Predict
            X = df.drop(columns=['WELL_NAME'], errors='ignore')
            X = X.fillna(0)
            
            y_prob = model.predict_proba(X)[:, 1]
            y_pred = model.predict(X)
            
            results = []
            for i, (well, prob, pred) in enumerate(zip(df['WELL_NAME'], y_prob, y_pred)):
                advisory = "Strongly Recommend" if prob >= 0.8 else "Review by Engineer" if prob >= 0.6 else "Low Priority"
                results.append({
                    "rank": i + 1,
                    "well_name": str(well),
                    "success_prob": round(float(prob), 4),
                    "success_class": int(pred),
                    "advisory": advisory
                })
            
            results = sorted(results, key=lambda x: x['success_prob'], reverse=True)
            
            self.send_json_response({
                "success": True,
                "predictions": results
            })
            
        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)
    
    def send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def get_sample_dashboard_data(self):
        """Return sample dashboard data"""
        return {
            "kpis": {
                "totalWells": 625,
                "testWells": 300,
                "rocAuc": 0.773,
                "accuracy": 0.827,
                "precision": 0.817,
                "recall": 0.917,
                "totalCostBaseline": 167740000,
                "totalCostOptimized": 96355000,
                "costSaving": 71385000,
                "stronglyRecommend": 165,
                "reviewEngineer": 78,
                "lowPriority": 57
            },
            "confusionMatrix": {"tn": 83, "fp": 37, "fn": 15, "tp": 165},
            "rocCurve": {
                "fpr": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.31, 0.4, 0.5, 0.7, 1.0],
                "tpr": [0.0, 0.45, 0.62, 0.72, 0.78, 0.83, 0.87, 0.92, 0.96, 0.99, 1.0]
            },
            "heteroIndex": [
                {"index": 1, "label": "High Oil - Low Water", "count": 156, "success_rate": 0.72},
                {"index": 2, "label": "High Oil - High Water", "count": 157, "success_rate": 0.58},
                {"index": 3, "label": "Low Oil - Low Water", "count": 156, "success_rate": 0.42},
                {"index": 4, "label": "Low Oil - High Water", "count": 156, "success_rate": 0.28}
            ],
            "topWells": [
                {"WELL_NAME": "WELL_006", "SUCCESS_PROB": 0.989, "HETERO_INDEX": 1, "ADVISORY": "Strongly Recommend"},
                {"WELL_NAME": "WELL_885", "SUCCESS_PROB": 0.985, "HETERO_INDEX": 1, "ADVISORY": "Strongly Recommend"},
                {"WELL_NAME": "WELL_946", "SUCCESS_PROB": 0.978, "HETERO_INDEX": 1, "ADVISORY": "Strongly Recommend"},
                {"WELL_NAME": "WELL_741", "SUCCESS_PROB": 0.972, "HETERO_INDEX": 2, "ADVISORY": "Strongly Recommend"},
                {"WELL_NAME": "WELL_093", "SUCCESS_PROB": 0.968, "HETERO_INDEX": 1, "ADVISORY": "Strongly Recommend"}
            ],
            "productionData": {
                "beforeIntervention": [1000, 920, 850, 785, 725, 670, 620, 575, 532, 492, 455, 420],
                "duringWorkover": [420, 50, 20, 10],
                "afterIntervention": [10, 650, 780, 820, 845, 860, 870, 875, 878, 880, 881, 882]
            }
        }
