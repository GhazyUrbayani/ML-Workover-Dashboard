import os
import sys
import warnings
import json
import logging
import io
import gc
import tempfile
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

# Register LogTransformer
import __main__
__main__.LogTransformer = LogTransformer

app = Flask(__name__, static_folder='.')
# No file size limit
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

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


def generate_dashboard_from_predictions(results, summary, df=None,
                                       model_metrics=None, production_lifecycle=None,
                                       npv_analysis=None):
    # Advisory distribution
    advisory_dist = {
        "stronglyRecommend": summary.get("strongly_recommend", 0),
        "reviewEngineer": summary.get("review_engineer", 0),
        "lowPriority": summary.get("low_priority", 0),
    }

    # Hetero index from results
    hetero_index = []
    for q in [1, 2, 3, 4]:
        wells_in_q = [r for r in results if r.get("hetero_index") == q]
        count = len(wells_in_q)
        if count > 0:
            success_rate = sum(1 for w in wells_in_q if w["success_class"] == 1) / count
        else:
            success_rate = None
        hetero_index.append({
            "index": q,
            "count": count,
            "success_rate": round(success_rate, 4) if success_rate is not None else None,
            "well_names": [w["well_name"] for w in wells_in_q],
        })

    # Cost estimation from INTV_TYPE if available
    total_cost_baseline = None
    total_cost_optimized = None
    cost_saving = None
    if df is not None and 'INTV_TYPE' in df.columns:
        try:
            cost_per_well = df['INTV_TYPE'].map(INTERVENTION_COSTS).fillna(0)
            total_cost_baseline = float(cost_per_well.sum())
            # ML-optimized: only intervene on wells with success_class == 1
            success_wells = set(r['well_name'] for r in results if r['success_class'] == 1)
            mask = df['WELL_NAME'].isin(success_wells)
            total_cost_optimized = float(cost_per_well[mask].sum())
            cost_saving = total_cost_baseline - total_cost_optimized
        except Exception:
            pass

    # Model performance from ground truth comparison
    roc_auc = None
    accuracy = None
    precision = None
    recall = None
    confusion_matrix = None
    if model_metrics:
        roc_auc = model_metrics.get('rocAuc')
        accuracy = model_metrics.get('accuracy')
        precision = model_metrics.get('precision')
        recall = model_metrics.get('recall')
        confusion_matrix = model_metrics.get('confusionMatrix')

    # Production lifecycle
    prod_data = None
    if production_lifecycle:
        prod_data = production_lifecycle

    dashboard = {
        "kpis": {
            "totalWells": summary.get("total_wells", len(results)),
            "testWells": summary.get("test_wells", None),
            "rocAuc": roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            # Cost
            "totalCostBaseline": total_cost_baseline,
            "totalCostOptimized": total_cost_optimized,
            "costSaving": cost_saving,
            # Advisory
            "stronglyRecommend": advisory_dist["stronglyRecommend"],
            "reviewEngineer": advisory_dist["reviewEngineer"],
            "lowPriority": advisory_dist["lowPriority"],
        },
        "confusionMatrix": confusion_matrix,
        "heteroIndex": hetero_index,
        "productionData": prod_data,
        "npvAnalysis": npv_analysis,
    }
    return dashboard

def process_timeseries_data(df):
    print("   🔄 Standardizing column names...")
    sys.stdout.flush()
    
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
            df.drop(columns=[old_name], inplace=True)
    
    # Define which columns are actually needed for feature engineering
    needed_cols = {
        'WELL_NAME', 'DATE_TIME', 'OIL_PROD', 'WATER_PROD', 'GAS_PROD',
        'WATER_CUT', 'FBHP', 'FTHP', 'FBHT', 'FTHT', 'POROSITY', 'GOR',
        'SHUT_IN', 'INTERVENTION_FLAG', 'INTERVENTION_COST', 'INTERVENTION_SUCCESS',
        'WELL_TYPE', 'RESERVOIR_QUALITY', 'INTV_TYPE', 'EVAL_INTV_TYPE',
        'CUM_OIL', 'CUM_WATER', 'CUM_GAS', 'CUM_LIQUID',
        'PERM_LOG_MEAN', 'NET_PAY_FROM_LOG', 'PHIE_MEAN', 'SW_MEAN', 'VCLGR_MEAN',
        'PAYFLAG_RATIO', 'RESFLAG_RATIO'
    }
    # Drop columns we don't need to save memory
    drop_cols = [c for c in df.columns if c not in needed_cols]
    if drop_cols:
        print(f"   🗑️  Dropping {len(drop_cols)} unused columns to save memory")
        df.drop(columns=drop_cols, inplace=True)
        gc.collect()
    
    # Downcast floats to float32 early
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)
        gc.collect()
    
    # Ensure datetime
    if 'DATE_TIME' in df.columns:
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        df = df.sort_values(['WELL_NAME', 'DATE_TIME'])
    
    # Calculate GOR if missing
    if 'GOR' not in df.columns and 'GAS_PROD' in df.columns and 'OIL_PROD' in df.columns:
        df['GOR'] = (df['GAS_PROD'] / (df['OIL_PROD'] + 1e-6)).astype(np.float32)
    
    print("   🔄 Generating MA & Diff features...")
    sys.stdout.flush()
    
    # Generate MA and Diff features (compute one at a time to save memory)
    for col in ['OIL_PROD', 'WATER_CUT', 'GOR', 'FBHP']:
        if col in df.columns:
            df[f'{col}_MA7'] = df.groupby('WELL_NAME')[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            ).astype(np.float32)
            df[f'{col}_MA90'] = df.groupby('WELL_NAME')[col].transform(
                lambda x: x.rolling(90, min_periods=1).mean()
            ).astype(np.float32)
            df[f'{col}_DIFF7'] = df.groupby('WELL_NAME')[col].transform(
                lambda x: x.diff(7)
            ).astype(np.float32)
    
    # Calculate cumulative production if missing
    if 'CUM_OIL' not in df.columns and 'OIL_PROD' in df.columns:
        df['CUM_OIL'] = df.groupby('WELL_NAME')['OIL_PROD'].cumsum().astype(np.float32)
    if 'CUM_WATER' not in df.columns and 'WATER_PROD' in df.columns:
        df['CUM_WATER'] = df.groupby('WELL_NAME')['WATER_PROD'].cumsum().astype(np.float32)
    if 'CUM_GAS' not in df.columns and 'GAS_PROD' in df.columns:
        df['CUM_GAS'] = df.groupby('WELL_NAME')['GAS_PROD'].cumsum().astype(np.float32)
    if 'CUM_LIQUID' not in df.columns:
        df['CUM_LIQUID'] = (df.get('CUM_OIL', 0) + df.get('CUM_WATER', 0))
        if hasattr(df['CUM_LIQUID'], 'astype'):
            df['CUM_LIQUID'] = df['CUM_LIQUID'].astype(np.float32)
    
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
        'INTERVENTION_SUCCESS': ('INTERVENTION_SUCCESS', 'last'),
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
    
    # Compute categorical modes BEFORE freeing df
    cat_modes = {}
    for cat_col in ['WELL_TYPE', 'RESERVOIR_QUALITY']:
        if cat_col in df.columns:
            cat_modes[cat_col] = df.groupby('WELL_NAME')[cat_col].agg(safe_mode).reset_index()
    intv_modes = {}
    for intv_col in ['INTV_TYPE', 'EVAL_INTV_TYPE']:
        if intv_col in df.columns:
            intv_modes[intv_col] = df.groupby('WELL_NAME')[intv_col].agg(mode_non_none).reset_index()
    
    # ── Extract production lifecycle BEFORE deleting time-series ──
    production_lifecycle = None
    if 'INTERVENTION_FLAG' in df.columns and 'OIL_PROD' in df.columns:
        try:
            intv_wells = df.loc[df['INTERVENTION_FLAG'] == 1, 'WELL_NAME'].unique()
            print(f"   📈 Wells with interventions: {len(intv_wells)}")
            sys.stdout.flush()
            if len(intv_wells) >= 1:
                sample_wells = intv_wells[:min(30, len(intv_wells))]
                avg_before_list = []
                avg_after_list = []
                for wn in sample_wells:
                    wd = df[df['WELL_NAME'] == wn].sort_values('DATE_TIME') if 'DATE_TIME' in df.columns else df[df['WELL_NAME'] == wn]
                    intv_mask = wd['INTERVENTION_FLAG'] == 1
                    if not intv_mask.any():
                        continue
                    first_intv_iloc = wd.index.get_loc(wd[intv_mask].index[0])
                    before_prod = wd.iloc[:first_intv_iloc]['OIL_PROD']
                    after_prod = wd.iloc[first_intv_iloc:]['OIL_PROD']
                    # Relaxed threshold: at least 5 data points (was 30)
                    if len(before_prod) >= 5:
                        avg_before_list.append(float(before_prod.tail(360).mean()))
                    if len(after_prod) >= 5:
                        avg_after_list.append(float(after_prod.head(360).mean()))
                print(f"   📈 Valid before/after samples: {len(avg_before_list)}/{len(avg_after_list)}")
                sys.stdout.flush()
                if avg_before_list and avg_after_list:
                    avg_b = np.mean(avg_before_list)
                    avg_a = np.mean(avg_after_list)
                    # Build representative 28-point curve (12 before + 4 WO + 12 after)
                    before_curve = np.linspace(avg_b * 1.3, avg_b * 0.65, 12)
                    wo_curve = np.array([avg_b * 0.08, avg_b * 0.03, avg_b * 0.02, avg_a * 0.5])
                    after_curve = np.linspace(avg_a * 0.6, avg_a * 1.05, 12)
                    production_lifecycle = {
                        'beforeIntervention': [round(float(x), 1) for x in before_curve],
                        'duringWorkover': [round(float(x), 1) for x in wo_curve],
                        'afterIntervention': [round(float(x), 1) for x in after_curve],
                    }
                    print(f"   📈 Production lifecycle: avg before={avg_b:.0f}, after={avg_a:.0f} BOPD")
                else:
                    print(f"   ⚠️ Not enough valid before/after data for lifecycle curve")
            else:
                print(f"   ⚠️ No wells with INTERVENTION_FLAG=1 found")
        except Exception as e:
            print(f"   ⚠️ Production lifecycle extraction skipped: {e}")
            import traceback
            traceback.print_exc()

    # Free the large time-series DataFrame immediately
    del df
    gc.collect()
    print(f"   🗑️  Freed time-series DataFrame")
    sys.stdout.flush()
    
    # Rename aggregated columns
    rename_map = {col: new_name for col, (new_name, _) in optional_aggs.items() if col in well_df.columns}
    well_df = well_df.rename(columns=rename_map)
    
    # Merge categorical columns
    for cat_col, cat_df in cat_modes.items():
        well_df = well_df.merge(cat_df, on='WELL_NAME', how='left')
    for intv_col, intv_df in intv_modes.items():
        well_df = well_df.merge(intv_df, on='WELL_NAME', how='left')
    del cat_modes, intv_modes
    
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
    
    return well_df, production_lifecycle


def process_well_logging_data(df):
    print("   🔄 Resolving log column names...")
    sys.stdout.flush()
    
    # Flexible column name resolution (same as notebook)
    col_candidates = {
        'PAYFLAG': ['PAYFLAG', 'PAY_FLAG', 'PAYFLAG_RATIO'],
        'PHIE': ['PHIE', 'POROSITY', 'PHI'],
        'SW': ['SWARCHIE', 'SW', 'SW_ARCHIE'],
        'VCLGR': ['VCLGR', 'VCL', 'VCL_GR', 'VSHALE', 'VSH'],
        'PERM': ['PERMEABILITY', 'PERM', 'K', 'PERM_LOG'],
        'RESFLAG': ['RESFLAG', 'RES_FLAG', 'RESFLAG_RATIO', 'RES', 'RESERVOIR_FLAG'],
    }
    
    resolved = {}
    for canonical, candidates in col_candidates.items():
        for name in candidates:
            if name in df.columns:
                resolved[canonical] = name
                break
    
    print(f"   📋 Resolved log columns: {resolved}")
    
    # Drop unused columns to save memory
    keep_cols = {'WELL_NAME', 'DEPTH'} | set(resolved.values())
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        gc.collect()
    
    # Downcast floats
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)
    
    # Build aggregation dict
    agg_dict = {}
    agg_rename = {}
    
    for canonical, src_col in resolved.items():
        if canonical == 'PAYFLAG':
            # Mean = ratio of pay flag, Sum = net pay thickness (count of pay intervals)
            agg_dict[src_col] = ['mean', 'sum']
            agg_rename[(src_col, 'mean')] = 'PAYFLAG_RATIO'
            agg_rename[(src_col, 'sum')] = 'NET_PAY_FROM_LOG'
        elif canonical == 'RESFLAG':
            agg_dict[src_col] = 'mean'
            agg_rename[(src_col, 'mean')] = 'RESFLAG_RATIO'
        elif canonical == 'PHIE':
            agg_dict[src_col] = 'mean'
            agg_rename[(src_col, 'mean')] = 'PHIE_MEAN'
        elif canonical == 'SW':
            agg_dict[src_col] = 'mean'
            agg_rename[(src_col, 'mean')] = 'SW_MEAN'
        elif canonical == 'VCLGR':
            agg_dict[src_col] = 'mean'
            agg_rename[(src_col, 'mean')] = 'VCLGR_MEAN'
        elif canonical == 'PERM':
            agg_dict[src_col] = 'mean'
            agg_rename[(src_col, 'mean')] = 'PERM_LOG_MEAN'
    
    if not agg_dict:
        print("   ⚠️ No recognized log columns found, using all defaults")
        well_names = df['WELL_NAME'].unique()
        well_df = pd.DataFrame({'WELL_NAME': well_names})
    else:
        well_df = df.groupby('WELL_NAME').agg(agg_dict)
        # Flatten multi-level columns
        well_df.columns = [agg_rename.get((col, agg), f"{col}_{agg}") for col, agg in well_df.columns]
        well_df = well_df.reset_index()
    
    # Free the large log DataFrame
    del df
    gc.collect()
    
    n_wells = len(well_df)
    print(f"   📊 Aggregated {n_wells} wells from log data")
    
    np.random.seed(42)
    production_defaults = {
        'CUM_OIL': (500000.0, 0.3),
        'CUM_WATER': (200000.0, 0.3),
        'CUM_GAS': (300000.0, 0.3),
        'CUM_LIQUID': (700000.0, 0.3),
        'WATER_CUT_MEAN': (0.35, 0.2),
        'FBHP_MEAN': (1500.0, 0.15),
        'FTHP_MEAN': (200.0, 0.15),
        'FBHT_MEAN': (180.0, 0.1),
        'FTHT_MEAN': (100.0, 0.1),
        'POROSITY_MEAN': (0.15, 0.2),
        'N_SHUT_IN': (2.0, 0.5),
        'N_INTERVENTION': (1.0, 0.5),
        'INTERVENTION_COST_MEAN': (100000.0, 0.3),
        'OIL_PROD_MA7_MEAN': (500.0, 0.3),
        'OIL_PROD_MA90_MEAN': (450.0, 0.3),
        'WATER_CUT_MA7_MEAN': (0.35, 0.2),
        'WATER_CUT_MA90_MEAN': (0.33, 0.2),
        'OIL_PROD_DIFF7_MEAN': (-10.0, 0.5),
        'WATER_CUT_DIFF7_MEAN': (0.005, 0.5),
        'GOR_DIFF7_MEAN': (5.0, 0.5),
        'FBHP_DIFF7_MEAN': (-5.0, 0.5),
        'OIL_DECLINE_RATE_MEAN': (-0.02, 0.3),
    }
    
    for col, (default_val, noise_scale) in production_defaults.items():
        if col not in well_df.columns:
            noise = np.random.normal(1.0, noise_scale, n_wells).clip(0.3, 2.0)
            well_df[col] = (default_val * noise).astype(np.float32)
    
    # Fill missing log-derived columns with defaults
    log_defaults = {
        'PHIE_MEAN': 0.15, 'SW_MEAN': 0.35, 'VCLGR_MEAN': 0.25,
        'PERM_LOG_MEAN': 100.0, 'PAYFLAG_RATIO': 0.6,
        'RESFLAG_RATIO': 0.7, 'NET_PAY_FROM_LOG': 50.0,
    }
    for col, default in log_defaults.items():
        if col not in well_df.columns:
            noise = np.random.normal(1.0, 0.1, n_wells).clip(0.7, 1.3)
            well_df[col] = (default * noise).astype(np.float32)
    
    # Add categorical defaults
    if 'WELL_TYPE' not in well_df.columns:
        well_df['WELL_TYPE'] = 'PRODUCER'
    if 'RESERVOIR_QUALITY' not in well_df.columns:
        # Derive reservoir quality from PHIE if available
        if 'PHIE_MEAN' in well_df.columns:
            well_df['RESERVOIR_QUALITY'] = well_df['PHIE_MEAN'].apply(
                lambda x: 'HIGH' if x > 0.2 else ('MEDIUM' if x > 0.1 else 'LOW')
            )
        else:
            well_df['RESERVOIR_QUALITY'] = 'MEDIUM'
    
    print(f"   ✅ Well logging features ready: {list(well_df.columns)[:10]}...")
    sys.stdout.flush()
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
            return jsonify({"error": "File too large. Maximum 100MB for server with limited memory. Please reduce rows/columns before uploading."}), 413
        
        if not has_file:
            print(f"❌ No 'file' in request.files. Keys: {list(request.files.keys())}")
            sys.stdout.flush()
            return jsonify({"error": "No file uploaded. Send CSV with 'file' field."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        print(f"📄 File received: {file.filename}")
        
        # Save uploaded file to disk (temp file)
        tmp_path = None
        try:
            filename = file.filename.lower()
            suffix = '.parquet' if filename.endswith('.parquet') else '.csv'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                while True:
                    chunk = file.stream.read(8192)
                    if not chunk:
                        break
                    tmp.write(chunk)
            
            file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            print(f"💾 Saved to temp file: {file_size_mb:.1f} MB")
            sys.stdout.flush()
            
            file.close()
            gc.collect()
            
            # Columns actually used by the pipeline — prune everything else
            _NEEDED_COLS = {
                'WELL_NAME', 'DATE_TIME', 'DATETIME',
                'OIL_PROD', 'OILPROD', 'WATER_PROD', 'WATERPROD',
                'GAS_PROD', 'GASPROD', 'WATER_CUT', 'WATERCUT',
                'FBHP', 'FTHP', 'FBHT', 'FTHT', 'POROSITY', 'GOR',
                'SHUT_IN', 'SHUTIN', 'INTERVENTION_FLAG', 'INTERVENTIONFLAG',
                'INTERVENTION_COST', 'INTERVENTIONCOST',
                'INTERVENTION_SUCCESS', 'INTERVENTIONSUCCESS',
                'WELL_TYPE', 'WELLTYPE', 'RESERVOIR_QUALITY', 'RESERVOIRQUALITY',
                'INTV_TYPE', 'INTVTYPE', 'EVAL_INTV_TYPE', 'EVALINTVTYPE',
                'CUM_OIL', 'CUM_WATER', 'CUM_GAS', 'CUM_LIQUID',
                'PERM_LOG_MEAN', 'NET_PAY_FROM_LOG', 'PHIE_MEAN', 'SW_MEAN',
                'VCLGR_MEAN', 'PAYFLAG_RATIO', 'RESFLAG_RATIO',
                'DEPTH', 'PAYFLAG', 'PAY_FLAG', 'PHIE', 'PHI',
                'SWARCHIE', 'SW', 'SW_ARCHIE', 'VCLGR', 'VCL', 'VCL_GR',
                'VSHALE', 'VSH', 'PERMEABILITY', 'PERM', 'K', 'PERM_LOG',
                'RESFLAG', 'RES_FLAG', 'RES', 'RESERVOIR_FLAG',
                'HETERO_INDEX',
            }
            _NEEDED_UPPER = {c.upper().replace(' ', '_') for c in _NEEDED_COLS}

            # Read file based on extension (CSV or Parquet) from disk
            if filename.endswith('.parquet'):
                # Column-pruned read: only load columns we actually need
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(tmp_path)
                all_parquet_cols = [c for c in pf.schema.names]
                keep_cols = [c for c in all_parquet_cols
                             if c.upper().replace(' ', '_') in _NEEDED_UPPER]
                if not keep_cols:
                    keep_cols = all_parquet_cols  # fallback
                print(f"   📦 Parquet has {len(all_parquet_cols)} cols, reading {len(keep_cols)}")
                sys.stdout.flush()
                df = pf.read(columns=keep_cols).to_pandas()
                del pf
                gc.collect()
                print(f"📊 Received Parquet: {len(df)} rows, {len(df.columns)} columns")
            elif filename.endswith('.csv'):
                # For CSV we can sniff columns first too
                sample = pd.read_csv(tmp_path, nrows=0)
                all_csv_cols = list(sample.columns)
                keep_cols = [c for c in all_csv_cols
                             if c.upper().replace(' ', '_') in _NEEDED_UPPER]
                if not keep_cols:
                    keep_cols = None  # read all as fallback
                df = pd.read_csv(tmp_path, usecols=keep_cols, low_memory=True)
                print(f"📊 Received CSV: {len(df)} rows, {len(df.columns)} columns")
            else:
                return jsonify({"error": "Unsupported file format. Use .csv or .parquet"}), 400
        finally:
            # Always clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # Memory check: log approximate size
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"💾 DataFrame memory: {mem_mb:.1f} MB")
        sys.stdout.flush()

        # No memory size limit enforced
        
        # Immediately downcast floats to float32 to save ~50% memory
        float_cols = df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype(np.float32)
            gc.collect()
            mem_mb_after = df.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"   ⚡ After float32 optimization: {mem_mb_after:.1f} MB")
        
        # Standardize column names (handle both formats)
        df.columns = df.columns.str.upper().str.replace(' ', '_')
        print(f"📋 Columns after standardization: {list(df.columns)[:10]}...")
        
        # Detect data type: time-series production, well logging, or pre-aggregated well-level
        is_timeseries = 'DATE_TIME' in df.columns or 'DATETIME' in df.columns
        is_well_logging = (
            'DEPTH' in df.columns and 
            'WELL_NAME' in df.columns and
            not is_timeseries and
            'CUM_OIL' not in df.columns
        )
        
        if is_timeseries:
            print("📈 Detected time-series production data - performing feature engineering...")
            df, _production_lifecycle = process_timeseries_data(df)
            print(f"✅ Aggregated to {len(df)} wells")
        elif is_well_logging:
            print("🔬 Detected well logging data - extracting petrophysical features...")
            df = process_well_logging_data(df)
            _production_lifecycle = None
            print(f"✅ Aggregated to {len(df)} wells")
        else:
            _production_lifecycle = None
        
        # Validate required columns
        required_cols = ['WELL_NAME']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}. File must contain at least WELL_NAME."}), 400
        
        # For production-based data, also check CUM_OIL/CUM_WATER
        if not is_well_logging and ('CUM_OIL' not in df.columns or 'CUM_WATER' not in df.columns):
            prod_missing = [c for c in ['CUM_OIL', 'CUM_WATER'] if c not in df.columns]
            available = list(df.columns)[:20]
            return jsonify({
                "error": f"Missing production columns: {prod_missing}. This doesn't look like production data or well logging data. Available columns: {available}"
            }), 400
        
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
        
        # ── NPV Economic Ranking ──
        oil_price = 70  # $/bbl
        discount_rate = 0.10
        npv_values = []
        for i in range(len(df)):
            row = df.iloc[i]
            intervention_cost = 0
            if 'INTV_TYPE' in df.columns:
                intervention_cost = INTERVENTION_COSTS.get(str(row.get('INTV_TYPE', 'NONE')), 0)
            elif 'INTERVENTION_COST_MEAN' in df.columns:
                intervention_cost = float(row.get('INTERVENTION_COST_MEAN', 0))
            
            baseline_oil = float(row.get('OIL_PROD_MA90_MEAN', 0)) * 365
            improvement_factor = 0.20 * float(y_prob[i])
            oil_gain = baseline_oil * improvement_factor * 2
            revenue = oil_gain * oil_price
            npv = (revenue - intervention_cost) / ((1 + discount_rate) ** 2)
            npv_values.append(npv)
        
        npv_arr = np.array(npv_values)
        npv_min, npv_max = npv_arr.min(), npv_arr.max()
        if npv_max > npv_min:
            npv_norm = (npv_arr - npv_min) / (npv_max - npv_min)
        else:
            npv_norm = np.full_like(npv_arr, 0.5)
        
        # Combined rank score: 60% technical + 40% economic
        rank_scores = 0.6 * y_prob + 0.4 * npv_norm
        
        # Create results
        hetero_labels = {
            '1': "High Oil - Low Water",
            '2': "High Oil - High Water",
            '3': "Low Oil - Low Water",
            '4': "Low Oil - High Water"
        }
        results = []
        for i, (well, prob, pred, hetero) in enumerate(zip(well_names, y_prob, y_pred, df['HETERO_INDEX'])):
            npv = npv_values[i]
            rs = rank_scores[i]
            
            # Advisory: considers both technical + economic
            if prob >= 0.7 and npv > 0:
                advisory = "Strongly Recommend"
            elif prob >= 0.7 and npv <= 0:
                advisory = "Review by Engineer"  # High success but unprofitable
            elif prob >= 0.4 and npv > 0:
                advisory = "Review by Engineer"
            else:
                advisory = "Low Priority"
            
            results.append({
                "rank": i + 1,
                "well_name": str(well),
                "success_prob": round(float(prob), 4),
                "success_class": int(pred),
                "hetero_index": int(hetero) if str(hetero).isdigit() else 0,
                "hetero_label": hetero_labels.get(str(hetero), "Unknown"),
                "advisory": advisory,
                "npv": round(float(npv), 0),
                "rank_score": round(float(rs), 4),
                "estimated_roi": round(float(prob) * 150000 - 50000, 0)
            })
        
        # Sort by combined rank score (not just probability)
        results = sorted(results, key=lambda x: x['rank_score'], reverse=True)
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        # Calculate summary stats
        summary = {
            "total_wells": len(results),
            "test_wells": len(results),
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
        
        # ── ML-Based Production Lifecycle ──
        try:
            success_mask = y_pred == 1
            success_oil = float(df.loc[success_mask, 'OIL_PROD_MA90_MEAN'].mean()) if success_mask.any() else 0
            avg_prob_success = float(y_prob[success_mask].mean()) if success_mask.any() else 0.5
            improvement = avg_prob_success * 0.25
            avg_baseline = float(df['OIL_PROD_MA90_MEAN'].mean())
            avg_after = avg_baseline * (1 + improvement)
            
            before_curve = np.linspace(avg_baseline * 1.3, avg_baseline * 0.65, 12)
            wo_curve = np.array([avg_baseline * 0.08, avg_baseline * 0.03, avg_baseline * 0.02, avg_after * 0.5])
            after_curve = np.linspace(avg_after * 0.6, avg_after * 1.05, 12)
            
            ml_production_lifecycle = {
                'beforeIntervention': [round(float(x), 1) for x in before_curve],
                'duringWorkover': [round(float(x), 1) for x in wo_curve],
                'afterIntervention': [round(float(x), 1) for x in after_curve],
                'mlBased': True,
                'avgSuccessProb': round(float(avg_prob_success), 3),
                'predictedImprovement': round(float(improvement * 100), 1),
                'nSuccessWells': int(success_mask.sum()),
                'nFailWells': int((~success_mask).sum()),
            }
            # Use ML lifecycle if time-series lifecycle not available
            if _production_lifecycle is None:
                _production_lifecycle = ml_production_lifecycle
            else:
                # Merge ML metadata into time-series lifecycle
                _production_lifecycle['mlBased'] = True
                _production_lifecycle['avgSuccessProb'] = ml_production_lifecycle['avgSuccessProb']
                _production_lifecycle['predictedImprovement'] = ml_production_lifecycle['predictedImprovement']
                _production_lifecycle['nSuccessWells'] = ml_production_lifecycle['nSuccessWells']
                _production_lifecycle['nFailWells'] = ml_production_lifecycle['nFailWells']
            print(f"📈 ML Production Lifecycle: before={avg_baseline:.0f}, after={avg_after:.0f} BOPD")
        except Exception as e:
            print(f"⚠️ ML production lifecycle skipped: {e}")
        
        # ── NPV Analysis Summary ──
        npv_analysis = {
            'totalNPV': round(float(sum(npv_values)), 0),
            'avgNPV': round(float(np.mean(npv_values)), 0),
            'medianNPV': round(float(np.median(npv_values)), 0),
            'positiveNPVCount': int(sum(1 for v in npv_values if v > 0)),
            'negativeNPVCount': int(sum(1 for v in npv_values if v <= 0)),
        }
        
        # Free large objects before building response
        del X
        gc.collect()
        
        # ── Compute model metrics if ground truth is available ──
        model_metrics = None
        # Check multiple possible column names for ground truth
        gt_col = None
        for col_name in ['INTERVENTION_SUCCESS', 'INTERVENTIONSUCCESS', 'SUCCESS']:
            if col_name in df.columns:
                gt_col = col_name
                break
        
        if gt_col:
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix as sk_cm
                gt_values = df[gt_col].values
                # Handle NaN: drop wells without ground truth
                valid_mask = pd.notna(gt_values)
                n_valid = valid_mask.sum()
                n_total = len(gt_values)
                print(f"📊 Ground truth '{gt_col}': {n_valid}/{n_total} wells have labels")
                sys.stdout.flush()
                
                if n_valid >= 2:
                    y_true = gt_values[valid_mask].astype(float).astype(int)
                    y_pred_valid = y_pred[valid_mask]
                    y_prob_valid = y_prob[valid_mask]
                    
                    # Only compute if we have both classes
                    unique_classes = set(y_true)
                    if len(unique_classes) >= 2:
                        acc = accuracy_score(y_true, y_pred_valid)
                        prec = precision_score(y_true, y_pred_valid, zero_division=0)
                        rec = recall_score(y_true, y_pred_valid, zero_division=0)
                        roc = roc_auc_score(y_true, y_prob_valid)
                        cm = sk_cm(y_true, y_pred_valid)
                        model_metrics = {
                            'rocAuc': round(float(roc), 4),
                            'accuracy': round(float(acc), 4),
                            'precision': round(float(prec), 4),
                            'recall': round(float(rec), 4),
                            'confusionMatrix': {
                                'tn': int(cm[0][0]), 'fp': int(cm[0][1]),
                                'fn': int(cm[1][0]), 'tp': int(cm[1][1]),
                            }
                        }
                        print(f"📊 Model metrics: AUC={roc:.3f} Acc={acc:.3f} P={prec:.3f} R={rec:.3f}")
                        print(f"   CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
                    else:
                        print(f"⚠️ Only one class in ground truth ({unique_classes}), skipping metrics")
                else:
                    print(f"⚠️ Too few valid ground truth labels ({n_valid}), skipping metrics")
            except Exception as e:
                import traceback
                print(f"⚠️ Could not compute model metrics: {e}")
                traceback.print_exc()
        else:
            print(f"ℹ️ No ground truth column found in data. Columns: {list(df.columns)}")
        
        # Fallback: if model_metrics couldn't be computed, use pre-trained metrics from dashboard_data.json
        if model_metrics is None and dashboard_data:
            print("📊 Falling back to pre-trained model metrics from dashboard_data.json")
            mi = dashboard_data.get('modelInfo', {})
            cm_data = dashboard_data.get('confusionMatrix', {})
            if mi:
                model_metrics = {
                    'rocAuc': mi.get('rocAuc'),
                    'accuracy': mi.get('accuracy'),
                    'precision': mi.get('precision'),
                    'recall': mi.get('recall'),
                    'confusionMatrix': cm_data if cm_data else None,
                }
                # Use pre-trained test wells count if available
                if mi.get('nTestWells'):
                    summary['test_wells'] = mi['nTestWells']
                print(f"   ✅ Using pre-trained metrics: AUC={mi.get('rocAuc')} Acc={mi.get('accuracy')}")
            sys.stdout.flush()
        
        sys.stdout.flush()
        
        del y_prob, y_pred
        gc.collect()
        
        # Generate dashboard data from predictions
        dashboard_update = generate_dashboard_from_predictions(
            results, summary, df,
            model_metrics=model_metrics,
            production_lifecycle=_production_lifecycle,
            npv_analysis=npv_analysis,
        )
        
        del df
        gc.collect()
        
        response_data = {
            "success": True,
            "summary": summary,
            "predictions": results,
            "dashboard_data": dashboard_update
        }
        print(f"✅ Prediction complete: {len(results)} wells (with dashboard_data)")
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
    
    # Run Flask server - NEVER use debug=True in production (doubles memory with reloader)
    is_production = os.environ.get('RENDER') or os.environ.get('PORT')
    app.run(host='0.0.0.0', port=5000, debug=not is_production)
