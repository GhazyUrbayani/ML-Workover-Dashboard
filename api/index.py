"""
Vercel Serverless Function for ML Workover Optimization
Deploy to Vercel: vercel --prod

Ported from main.py (Flask) to Vercel's BaseHTTPRequestHandler.
Includes LogTransformer, preprocessor, intervention costs, NPV ranking,
time-series aggregation, and well-logging feature extraction.
"""

import os
import sys
import json
import warnings
import gc

# Suppress warnings
os.environ['LIGHTGBM_SUPPRESS_WARNINGS'] = '1'
warnings.filterwarnings('ignore')

from http.server import BaseHTTPRequestHandler
import pandas as pd
import numpy as np

# ── Custom transformer required by the joblib model ──
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    """Log1p transformer used during model training.
    Must be importable when joblib deserialises the pipeline."""
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


# Register LogTransformer in __main__ so joblib can resolve it
import __main__
__main__.LogTransformer = LogTransformer

# ── Try to import joblib ──
try:
    import joblib
    MODEL_LOADED = True
except ImportError:
    MODEL_LOADED = False

# ── Intervention cost mapping (same as main.py) ──
INTERVENTION_COSTS = {
    "REPERFORASI": 100_000,
    "WATER_SHUTOFF": 140_000,
    "STIMULASI": 200_000,
    "ARTIFICIAL_LIFT": 275_000,
    "WELLBORE_CLEANOUT": 70_000,
    "ZONAL_ISOLATION": 235_000,
    "NONE": 0,
}

# Global resources (loaded once per cold start)
model = None
preprocessor = None
dashboard_data = None
model_error = None
openmp_status = None


def _resolve_path(filename):
    """Try several paths Vercel may place bundled files at."""
    candidates = [
        filename,
        f'../{filename}',
        f'/var/task/{filename}',
        os.path.join(os.path.dirname(__file__), '..', filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _load_openmp():
    """Pre-load libgomp.so.1 with RTLD_GLOBAL for LightGBM on serverless Linux."""
    global openmp_status
    import ctypes
    candidates = [
        os.path.join(os.path.dirname(__file__), 'lib', 'libgomp.so.1'),
        'api/lib/libgomp.so.1',
        'lib/libgomp.so.1',
        'libgomp.so.1',
        os.path.join(os.path.dirname(__file__), '..', 'lib', 'libgomp.so.1'),
    ]
    for candidate in candidates:
        p = candidate if os.path.exists(candidate) else _resolve_path(candidate)
        if p and os.path.exists(p):
            try:
                lib_dir = os.path.dirname(os.path.abspath(p))
                os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                ctypes.CDLL(os.path.abspath(p), mode=ctypes.RTLD_GLOBAL)
                openmp_status = f"Loaded from {p}"
                print(f"✅ Pre-loaded OpenMP runtime from: {p}")
                return True
            except Exception as e:
                openmp_status = f"Failed loading OpenMP from {p}: {e}"
                print(f"⚠️ Failed loading OpenMP from {p}: {e}")
    if not openmp_status:
        openmp_status = "OpenMP libgomp.so.1 not found"
    return False


def load_resources():
    """Load model, preprocessor, and dashboard data."""
    global model, preprocessor, dashboard_data, model_error

    # Pre-load OpenMP for LightGBM before deserializing pipeline
    _load_openmp()

    # Load model
    if MODEL_LOADED:
        path = _resolve_path('workover_model_pipeline.joblib')
        if path:
            try:
                model = joblib.load(path)
                print(f"Model loaded from: {path}")
            except Exception as e:
                model_error = f"Error loading model from {path}: {str(e)}"
                print(model_error)
        else:
            model_error = "Model file workover_model_pipeline.joblib not found"
    else:
        model_error = "joblib module could not be imported"

    # Load preprocessor
    if MODEL_LOADED:
        path = _resolve_path('preprocessor.joblib')
        if path:
            try:
                preprocessor = joblib.load(path)
                print(f"Preprocessor loaded from: {path}")
            except Exception as e:
                print(f"Error loading preprocessor from {path}: {e}")

    # Load dashboard data
    path = _resolve_path('dashboard_data.json')
    if path:
        try:
            with open(path, 'r') as f:
                dashboard_data = json.load(f)
            print(f"Dashboard data loaded from: {path}")
        except Exception as e:
            print(f"Error loading dashboard data from {path}: {e}")


# Load resources on cold start
load_resources()


# ── Feature columns (same as main.py training pipeline) ──
NUMERIC_LOG_COLS = ['CUM_OIL', 'CUM_WATER', 'CUM_GAS', 'CUM_LIQUID',
                    'PERM_LOG_MEAN', 'INTERVENTION_COST_MEAN']
RATIO_COLS = ['WATER_CUT_MEAN', 'POROSITY_MEAN', 'PAYFLAG_RATIO',
              'RESFLAG_RATIO']
PHYS_COLS = ['FBHP_MEAN', 'FTHP_MEAN', 'FBHT_MEAN', 'FTHT_MEAN',
             'NET_PAY_FROM_LOG']
MA_DIFF_COLS = [
    'OIL_PROD_MA7_MEAN', 'OIL_PROD_MA90_MEAN',
    'WATER_CUT_MA7_MEAN', 'WATER_CUT_MA90_MEAN',
    'OIL_PROD_DIFF7_MEAN', 'WATER_CUT_DIFF7_MEAN',
    'GOR_DIFF7_MEAN', 'FBHP_DIFF7_MEAN',
    'OIL_DECLINE_RATE_MEAN', 'N_SHUT_IN', 'N_INTERVENTION',
    'PHIE_MEAN', 'SW_MEAN', 'VCLGR_MEAN',
]
CAT_COLS = ['WELL_TYPE', 'RESERVOIR_QUALITY', 'HETERO_INDEX']
ALL_FEATURE_COLS = NUMERIC_LOG_COLS + RATIO_COLS + PHYS_COLS + MA_DIFF_COLS + CAT_COLS

HETERO_LABELS = {
    '1': "High Oil - Low Water",
    '2': "High Oil - High Water",
    '3': "Low Oil - Low Water",
    '4': "Low Oil - High Water",
}


def process_timeseries_data(df):
    """Aggregate daily time-series production data to well-level features."""
    col_mapping = {
        'OILPROD': 'OIL_PROD', 'WATERPROD': 'WATER_PROD', 'GASPROD': 'GAS_PROD',
        'WATERCUT': 'WATER_CUT', 'DATETIME': 'DATE_TIME',
        'INTVTYPE': 'INTV_TYPE', 'EVALINTVTYPE': 'EVAL_INTV_TYPE',
        'INTERVENTIONFLAG': 'INTERVENTION_FLAG', 'INTERVENTIONSUCCESS': 'INTERVENTION_SUCCESS',
        'INTERVENTIONCOST': 'INTERVENTION_COST', 'SHUTIN': 'SHUT_IN',
        'WELLNAME': 'WELL_NAME', 'WELLTYPE': 'WELL_TYPE',
        'RESERVOIRQUALITY': 'RESERVOIR_QUALITY',
    }
    for old, new in col_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
            df.drop(columns=[old], inplace=True)

    # Datetime sort
    if 'DATE_TIME' in df.columns:
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        df = df.sort_values(['WELL_NAME', 'DATE_TIME'])

    # GOR
    if 'GOR' not in df.columns and 'GAS_PROD' in df.columns and 'OIL_PROD' in df.columns:
        df['GOR'] = (df['GAS_PROD'] / (df['OIL_PROD'] + 1e-6)).astype(np.float32)

    # MA & Diff features
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

    # Cumulative production
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
    if 'OIL_PROD' in df.columns:
        df['OIL_DECLINE_RATE'] = df.groupby('WELL_NAME')['OIL_PROD'].transform(
            lambda x: x.pct_change().rolling(30, min_periods=1).mean()
        )

    # Aggregation
    def safe_mode(x):
        return x.mode().iat[0] if len(x.mode()) > 0 else "UNKNOWN"

    def mode_non_none(x):
        x = x[x != "NONE"]
        return x.mode().iat[0] if len(x.mode()) > 0 else "NONE"

    agg_dict = {'CUM_OIL': 'max', 'CUM_WATER': 'max', 'CUM_GAS': 'max', 'CUM_LIQUID': 'max'}
    optional_aggs = {
        'WATER_CUT': ('WATER_CUT_MEAN', 'mean'), 'FBHP': ('FBHP_MEAN', 'mean'),
        'FTHP': ('FTHP_MEAN', 'mean'), 'FBHT': ('FBHT_MEAN', 'mean'),
        'FTHT': ('FTHT_MEAN', 'mean'), 'POROSITY': ('POROSITY_MEAN', 'mean'),
        'SHUT_IN': ('N_SHUT_IN', 'sum'), 'INTERVENTION_FLAG': ('N_INTERVENTION', 'sum'),
        'INTERVENTION_COST': ('INTERVENTION_COST_MEAN', 'mean'),
        'INTERVENTION_SUCCESS': ('INTERVENTION_SUCCESS', 'last'),
        'OIL_PROD_MA7': ('OIL_PROD_MA7_MEAN', 'mean'), 'OIL_PROD_MA90': ('OIL_PROD_MA90_MEAN', 'mean'),
        'WATER_CUT_MA7': ('WATER_CUT_MA7_MEAN', 'mean'), 'WATER_CUT_MA90': ('WATER_CUT_MA90_MEAN', 'mean'),
        'OIL_PROD_DIFF7': ('OIL_PROD_DIFF7_MEAN', 'mean'), 'WATER_CUT_DIFF7': ('WATER_CUT_DIFF7_MEAN', 'mean'),
        'GOR_DIFF7': ('GOR_DIFF7_MEAN', 'mean'), 'FBHP_DIFF7': ('FBHP_DIFF7_MEAN', 'mean'),
        'OIL_DECLINE_RATE': ('OIL_DECLINE_RATE_MEAN', 'mean'),
    }
    for col, (_, func) in optional_aggs.items():
        if col in df.columns:
            agg_dict[col] = func

    well_df = df.groupby('WELL_NAME').agg(agg_dict).reset_index()

    # Categorical modes
    for cat_col in ['WELL_TYPE', 'RESERVOIR_QUALITY']:
        if cat_col in df.columns:
            cat_df = df.groupby('WELL_NAME')[cat_col].agg(safe_mode).reset_index()
            well_df = well_df.merge(cat_df, on='WELL_NAME', how='left')
    for intv_col in ['INTV_TYPE', 'EVAL_INTV_TYPE']:
        if intv_col in df.columns:
            intv_df = df.groupby('WELL_NAME')[intv_col].agg(mode_non_none).reset_index()
            well_df = well_df.merge(intv_df, on='WELL_NAME', how='left')

    # Production lifecycle extraction
    production_lifecycle = None
    if 'INTERVENTION_FLAG' in df.columns and 'OIL_PROD' in df.columns:
        try:
            intv_wells = df.loc[df['INTERVENTION_FLAG'] == 1, 'WELL_NAME'].unique()
            if len(intv_wells) >= 1:
                sample_wells = intv_wells[:min(30, len(intv_wells))]
                avg_before_list, avg_after_list = [], []
                for wn in sample_wells:
                    wd = df[df['WELL_NAME'] == wn].sort_values('DATE_TIME') if 'DATE_TIME' in df.columns else df[df['WELL_NAME'] == wn]
                    intv_mask = wd['INTERVENTION_FLAG'] == 1
                    if not intv_mask.any():
                        continue
                    first_intv_iloc = wd.index.get_loc(wd[intv_mask].index[0])
                    before_prod = wd.iloc[:first_intv_iloc]['OIL_PROD']
                    after_prod = wd.iloc[first_intv_iloc:]['OIL_PROD']
                    if len(before_prod) >= 5:
                        avg_before_list.append(float(before_prod.tail(360).mean()))
                    if len(after_prod) >= 5:
                        avg_after_list.append(float(after_prod.head(360).mean()))
                if avg_before_list and avg_after_list:
                    avg_b = np.mean(avg_before_list)
                    avg_a = np.mean(avg_after_list)
                    production_lifecycle = {
                        'beforeIntervention': [round(float(x), 1) for x in np.linspace(avg_b * 1.3, avg_b * 0.65, 12)],
                        'duringWorkover': [round(float(x), 1) for x in [avg_b * 0.08, avg_b * 0.03, avg_b * 0.02, avg_a * 0.5]],
                        'afterIntervention': [round(float(x), 1) for x in np.linspace(avg_a * 0.6, avg_a * 1.05, 12)],
                    }
        except Exception:
            pass

    del df
    gc.collect()

    # Rename aggregated columns
    rename_map = {col: new_name for col, (new_name, _) in optional_aggs.items() if col in well_df.columns}
    well_df = well_df.rename(columns=rename_map)

    # Fill missing columns with defaults
    default_values = {
        'WATER_CUT_MEAN': 0.3, 'FBHP_MEAN': 1500, 'FTHP_MEAN': 200, 'FBHT_MEAN': 180, 'FTHT_MEAN': 100,
        'POROSITY_MEAN': 0.15, 'N_SHUT_IN': 0, 'N_INTERVENTION': 0, 'INTERVENTION_COST_MEAN': 0,
        'OIL_PROD_MA7_MEAN': 0, 'OIL_PROD_MA90_MEAN': 0, 'WATER_CUT_MA7_MEAN': 0, 'WATER_CUT_MA90_MEAN': 0,
        'OIL_PROD_DIFF7_MEAN': 0, 'WATER_CUT_DIFF7_MEAN': 0, 'GOR_DIFF7_MEAN': 0, 'FBHP_DIFF7_MEAN': 0,
        'OIL_DECLINE_RATE_MEAN': 0, 'PERM_LOG_MEAN': 100, 'NET_PAY_FROM_LOG': 10,
        'PHIE_MEAN': 0.15, 'SW_MEAN': 0.3, 'VCLGR_MEAN': 0.2, 'PAYFLAG_RATIO': 0.5, 'RESFLAG_RATIO': 0.5,
    }
    for col, default in default_values.items():
        if col not in well_df.columns:
            well_df[col] = default

    if 'WELL_TYPE' not in well_df.columns:
        well_df['WELL_TYPE'] = 'PRODUCER'
    if 'RESERVOIR_QUALITY' not in well_df.columns:
        well_df['RESERVOIR_QUALITY'] = 'MEDIUM'

    return well_df, production_lifecycle


def process_well_logging_data(df):
    """Extract petrophysical features from well-log data."""
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

    agg_dict = {}
    agg_rename = {}
    for canonical, src_col in resolved.items():
        if canonical == 'PAYFLAG':
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
        well_names = df['WELL_NAME'].unique()
        well_df = pd.DataFrame({'WELL_NAME': well_names})
    else:
        well_df = df.groupby('WELL_NAME').agg(agg_dict)
        well_df.columns = [agg_rename.get((col, agg), f"{col}_{agg}") for col, agg in well_df.columns]
        well_df = well_df.reset_index()

    del df
    gc.collect()

    n_wells = len(well_df)
    np.random.seed(42)

    production_defaults = {
        'CUM_OIL': (500000.0, 0.3), 'CUM_WATER': (200000.0, 0.3),
        'CUM_GAS': (300000.0, 0.3), 'CUM_LIQUID': (700000.0, 0.3),
        'WATER_CUT_MEAN': (0.35, 0.2), 'FBHP_MEAN': (1500.0, 0.15),
        'FTHP_MEAN': (200.0, 0.15), 'FBHT_MEAN': (180.0, 0.1),
        'FTHT_MEAN': (100.0, 0.1), 'POROSITY_MEAN': (0.15, 0.2),
        'N_SHUT_IN': (2.0, 0.5), 'N_INTERVENTION': (1.0, 0.5),
        'INTERVENTION_COST_MEAN': (100000.0, 0.3),
        'OIL_PROD_MA7_MEAN': (500.0, 0.3), 'OIL_PROD_MA90_MEAN': (450.0, 0.3),
        'WATER_CUT_MA7_MEAN': (0.35, 0.2), 'WATER_CUT_MA90_MEAN': (0.33, 0.2),
        'OIL_PROD_DIFF7_MEAN': (-10.0, 0.5), 'WATER_CUT_DIFF7_MEAN': (0.005, 0.5),
        'GOR_DIFF7_MEAN': (5.0, 0.5), 'FBHP_DIFF7_MEAN': (-5.0, 0.5),
        'OIL_DECLINE_RATE_MEAN': (-0.02, 0.3),
    }
    for col, (default_val, noise_scale) in production_defaults.items():
        if col not in well_df.columns:
            noise = np.random.normal(1.0, noise_scale, n_wells).clip(0.3, 2.0)
            well_df[col] = (default_val * noise).astype(np.float32)

    log_defaults = {
        'PHIE_MEAN': 0.15, 'SW_MEAN': 0.35, 'VCLGR_MEAN': 0.25,
        'PERM_LOG_MEAN': 100.0, 'PAYFLAG_RATIO': 0.6,
        'RESFLAG_RATIO': 0.7, 'NET_PAY_FROM_LOG': 50.0,
    }
    for col, default in log_defaults.items():
        if col not in well_df.columns:
            noise = np.random.normal(1.0, 0.1, n_wells).clip(0.7, 1.3)
            well_df[col] = (default * noise).astype(np.float32)

    if 'WELL_TYPE' not in well_df.columns:
        well_df['WELL_TYPE'] = 'PRODUCER'
    if 'RESERVOIR_QUALITY' not in well_df.columns:
        if 'PHIE_MEAN' in well_df.columns:
            well_df['RESERVOIR_QUALITY'] = well_df['PHIE_MEAN'].apply(
                lambda x: 'HIGH' if x > 0.2 else ('MEDIUM' if x > 0.1 else 'LOW')
            )
        else:
            well_df['RESERVOIR_QUALITY'] = 'MEDIUM'

    return well_df


def generate_dashboard_from_predictions(results, summary, df=None,
                                        model_metrics=None, production_lifecycle=None,
                                        npv_analysis=None):
    """Build dashboard JSON payload from prediction results."""
    advisory_dist = {
        "stronglyRecommend": summary.get("strongly_recommend", 0),
        "reviewEngineer": summary.get("review_engineer", 0),
        "lowPriority": summary.get("low_priority", 0),
    }
    hetero_index = []
    for q in [1, 2, 3, 4]:
        wells_in_q = [r for r in results if r.get("hetero_index") == q]
        count = len(wells_in_q)
        success_rate = (sum(1 for w in wells_in_q if w["success_class"] == 1) / count) if count > 0 else None
        hetero_index.append({
            "index": q, "count": count,
            "success_rate": round(success_rate, 4) if success_rate is not None else None,
            "well_names": [w["well_name"] for w in wells_in_q],
        })

    total_cost_baseline = total_cost_optimized = cost_saving = None
    if df is not None and 'INTV_TYPE' in df.columns:
        try:
            cost_per_well = df['INTV_TYPE'].map(INTERVENTION_COSTS).fillna(0)
            total_cost_baseline = float(cost_per_well.sum())
            success_wells = set(r['well_name'] for r in results if r['success_class'] == 1)
            mask = df['WELL_NAME'].isin(success_wells)
            total_cost_optimized = float(cost_per_well[mask].sum())
            cost_saving = total_cost_baseline - total_cost_optimized
        except Exception:
            pass

    roc_auc = accuracy = precision = recall = confusion_matrix = None
    if model_metrics:
        roc_auc = model_metrics.get('rocAuc')
        accuracy = model_metrics.get('accuracy')
        precision = model_metrics.get('precision')
        recall = model_metrics.get('recall')
        confusion_matrix = model_metrics.get('confusionMatrix')

    return {
        "kpis": {
            "totalWells": summary.get("total_wells", len(results)),
            "testWells": summary.get("test_wells", None),
            "rocAuc": roc_auc, "accuracy": accuracy,
            "precision": precision, "recall": recall,
            "totalCostBaseline": total_cost_baseline,
            "totalCostOptimized": total_cost_optimized,
            "costSaving": cost_saving,
            "stronglyRecommend": advisory_dist["stronglyRecommend"],
            "reviewEngineer": advisory_dist["reviewEngineer"],
            "lowPriority": advisory_dist["lowPriority"],
        },
        "confusionMatrix": confusion_matrix,
        "heteroIndex": hetero_index,
        "productionData": production_lifecycle,
        "npvAnalysis": npv_analysis,
    }


# ═══════════════════════════════════════════════════════════════
#  Vercel Handler
# ═══════════════════════════════════════════════════════════════

class handler(BaseHTTPRequestHandler):
    """Vercel Serverless Function Handler"""

    # ── Helpers ──────────────────────────────────────────────────

    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _get_request_path(self):
        import urllib.parse
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if 'path' in qs and qs['path'][0]:
            return qs['path'][0]
        for h in ['x-matched-path', 'x-forwarded-uri', 'x-invoke-path']:
            val = self.headers.get(h)
            if val:
                return val.split('?')[0]
        return parsed.path

    # ── GET ──────────────────────────────────────────────────────

    def do_GET(self):
        path = self._get_request_path()

        if path in ('/api', '/api/'):
            self.send_json_response({
                "status": "ok",
                "message": "ML Workover Optimization API (Vercel Serverless)",
                "endpoints": {
                    "GET /api": "This help message",
                    "GET /api/health": "Health check",
                    "GET /api/debug": "Debug system & model status",
                    "GET /api/dashboard-data": "Get dashboard data",
                    "POST /api/predict": "Predict from JSON well data",
                },
            })
        elif path == '/api/health':
            self.send_json_response({
                "status": "ok",
                "model_loaded": model is not None,
                "model_error": model_error,
                "openmp_status": openmp_status,
                "preprocessor_loaded": preprocessor is not None,
                "dashboard_data_loaded": dashboard_data is not None,
                "python_version": sys.version,
            })
        elif path == '/api/debug':
            self.send_json_response({
                "status": "ok",
                "resolved_path": path,
                "raw_path": self.path,
                "model_loaded": model is not None,
                "model_error": model_error,
                "openmp_status": openmp_status,
                "preprocessor_loaded": preprocessor is not None,
                "dashboard_data_loaded": dashboard_data is not None,
                "curdir": os.path.abspath('.'),
                "curdir_files": os.listdir('.') if os.path.exists('.') else [],
                "api_dir_files": os.listdir(os.path.dirname(__file__)) if os.path.exists(os.path.dirname(__file__)) else [],
            })
        elif path == '/api/dashboard-data':
            if dashboard_data:
                self.send_json_response(dashboard_data)
            else:
                self.send_json_response(self._get_sample_dashboard_data())
        else:
            self.send_json_response({"error": "Not found", "path": path, "raw": self.path}, status=404)

    # ── POST ─────────────────────────────────────────────────────

    def do_POST(self):
        path = self._get_request_path()

        if path in ('/api/predict', '/api/predict-single'):
            self._handle_predict()
        else:
            self.send_json_response({"error": "Not found", "path": path}, status=404)

    def _handle_predict(self):
        """Full prediction pipeline matching main.py."""
        if model is None:
            self.send_json_response({
                "error": "Model not loaded. Deploy with model file or use sample data.",
                "sample_data": True,
            }, status=503)
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body.decode('utf-8'))
                df = pd.DataFrame(data if isinstance(data, list) else [data])
            except Exception:
                self.send_json_response({"error": "Invalid JSON. Send array of well data."}, status=400)
                return

            # Standardize column names
            df.columns = df.columns.str.upper().str.replace(' ', '_')

            # Detect data type
            is_timeseries = 'DATE_TIME' in df.columns or 'DATETIME' in df.columns
            is_well_logging = (
                'DEPTH' in df.columns and
                'WELL_NAME' in df.columns and
                not is_timeseries and
                'CUM_OIL' not in df.columns
            )

            _production_lifecycle = None
            if is_timeseries:
                df, _production_lifecycle = process_timeseries_data(df)
            elif is_well_logging:
                df = process_well_logging_data(df)

            if 'WELL_NAME' not in df.columns:
                df['WELL_NAME'] = [f"WELL_{i}" for i in range(len(df))]

            # Heterogeneity Index
            if 'HETERO_INDEX' not in df.columns:
                if 'CUM_OIL' in df.columns and 'CUM_WATER' in df.columns:
                    oil_med = df['CUM_OIL'].median()
                    water_med = df['CUM_WATER'].median()
                    def calc_h(row):
                        ho = row['CUM_OIL'] >= oil_med
                        hw = row['CUM_WATER'] >= water_med
                        if ho and not hw: return 1
                        elif ho and hw: return 2
                        elif not ho and not hw: return 3
                        else: return 4
                    df['HETERO_INDEX'] = df.apply(calc_h, axis=1)
                else:
                    df['HETERO_INDEX'] = 1

            # Fill missing feature columns
            for col in ALL_FEATURE_COLS:
                if col not in df.columns:
                    df[col] = 'UNKNOWN' if col in CAT_COLS else 0.0
            for col in CAT_COLS:
                df[col] = df[col].astype(str)

            X = df[ALL_FEATURE_COLS].copy()
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Predict
            y_prob = model.predict_proba(X)[:, 1]
            y_pred = model.predict(X)

            well_names = df['WELL_NAME'].values

            # ── NPV Economic Ranking ──
            oil_price, discount_rate = 70, 0.10
            npv_values = []
            for i in range(len(df)):
                row = df.iloc[i]
                intv_cost = 0
                if 'INTV_TYPE' in df.columns:
                    intv_cost = INTERVENTION_COSTS.get(str(row.get('INTV_TYPE', 'NONE')), 0)
                elif 'INTERVENTION_COST_MEAN' in df.columns:
                    intv_cost = float(row.get('INTERVENTION_COST_MEAN', 0))
                baseline_oil = float(row.get('OIL_PROD_MA90_MEAN', 0)) * 365
                improvement = 0.20 * float(y_prob[i])
                oil_gain = baseline_oil * improvement * 2
                revenue = oil_gain * oil_price
                npv = (revenue - intv_cost) / ((1 + discount_rate) ** 2)
                npv_values.append(npv)

            npv_arr = np.array(npv_values)
            npv_min, npv_max = npv_arr.min(), npv_arr.max()
            npv_norm = (npv_arr - npv_min) / (npv_max - npv_min) if npv_max > npv_min else np.full_like(npv_arr, 0.5)
            rank_scores = 0.6 * y_prob + 0.4 * npv_norm

            # Build results
            results = []
            for i, (well, prob, pred, hetero) in enumerate(zip(well_names, y_prob, y_pred, df['HETERO_INDEX'])):
                npv = npv_values[i]
                rs = rank_scores[i]
                if prob >= 0.7 and npv > 0:
                    advisory = "Strongly Recommend"
                elif prob >= 0.7 and npv <= 0:
                    advisory = "Review by Engineer"
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
                    "hetero_label": HETERO_LABELS.get(str(hetero), "Unknown"),
                    "advisory": advisory,
                    "npv": round(float(npv), 0),
                    "rank_score": round(float(rs), 4),
                    "estimated_roi": round(float(prob) * 150000 - 50000, 0),
                })

            results = sorted(results, key=lambda x: x['rank_score'], reverse=True)
            for i, r in enumerate(results):
                r['rank'] = i + 1

            summary = {
                "total_wells": len(results),
                "test_wells": len(results),
                "strongly_recommend": sum(1 for r in results if r['advisory'] == "Strongly Recommend"),
                "review_engineer": sum(1 for r in results if r['advisory'] == "Review by Engineer"),
                "low_priority": sum(1 for r in results if r['advisory'] == "Low Priority"),
                "avg_success_prob": round(float(np.mean(y_prob)), 3),
                "hetero_distribution": {
                    f"q{q}": sum(1 for r in results if r['hetero_index'] == q) for q in [1, 2, 3, 4]
                },
            }

            # ML-based production lifecycle
            try:
                success_mask = y_pred == 1
                avg_prob_success = float(y_prob[success_mask].mean()) if success_mask.any() else 0.5
                improvement = avg_prob_success * 0.25
                avg_baseline = float(df['OIL_PROD_MA90_MEAN'].mean()) if 'OIL_PROD_MA90_MEAN' in df.columns else 0
                avg_after = avg_baseline * (1 + improvement)
                ml_lifecycle = {
                    'beforeIntervention': [round(float(x), 1) for x in np.linspace(avg_baseline * 1.3, avg_baseline * 0.65, 12)],
                    'duringWorkover': [round(float(x), 1) for x in [avg_baseline * 0.08, avg_baseline * 0.03, avg_baseline * 0.02, avg_after * 0.5]],
                    'afterIntervention': [round(float(x), 1) for x in np.linspace(avg_after * 0.6, avg_after * 1.05, 12)],
                    'mlBased': True,
                    'avgSuccessProb': round(float(avg_prob_success), 3),
                    'predictedImprovement': round(float(improvement * 100), 1),
                    'nSuccessWells': int(success_mask.sum()),
                    'nFailWells': int((~success_mask).sum()),
                }
                if _production_lifecycle is None:
                    _production_lifecycle = ml_lifecycle
                else:
                    _production_lifecycle.update({
                        'mlBased': True,
                        'avgSuccessProb': ml_lifecycle['avgSuccessProb'],
                        'predictedImprovement': ml_lifecycle['predictedImprovement'],
                        'nSuccessWells': ml_lifecycle['nSuccessWells'],
                        'nFailWells': ml_lifecycle['nFailWells'],
                    })
            except Exception:
                pass

            # NPV analysis summary
            npv_analysis = {
                'totalNPV': round(float(sum(npv_values)), 0),
                'avgNPV': round(float(np.mean(npv_values)), 0),
                'medianNPV': round(float(np.median(npv_values)), 0),
                'positiveNPVCount': int(sum(1 for v in npv_values if v > 0)),
                'negativeNPVCount': int(sum(1 for v in npv_values if v <= 0)),
            }

            # Model metrics from ground truth if available
            model_metrics = None
            gt_col = None
            for col_name in ['INTERVENTION_SUCCESS', 'INTERVENTIONSUCCESS', 'SUCCESS']:
                if col_name in df.columns:
                    gt_col = col_name
                    break

            if gt_col:
                try:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
                    from sklearn.metrics import confusion_matrix as sk_cm
                    gt_values = df[gt_col].values
                    valid_mask = pd.notna(gt_values)
                    n_valid = valid_mask.sum()
                    if n_valid >= 2:
                        y_true = gt_values[valid_mask].astype(float).astype(int)
                        y_pred_valid = y_pred[valid_mask]
                        y_prob_valid = y_prob[valid_mask]
                        if len(set(y_true)) >= 2:
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
                                },
                            }
                except Exception:
                    pass

            # Fallback to pre-trained metrics
            if model_metrics is None and dashboard_data:
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
                    if mi.get('nTestWells'):
                        summary['test_wells'] = mi['nTestWells']

            # Generate dashboard data
            dashboard_update = generate_dashboard_from_predictions(
                results, summary, df,
                model_metrics=model_metrics,
                production_lifecycle=_production_lifecycle,
                npv_analysis=npv_analysis,
            )

            del X, df
            gc.collect()

            self.send_json_response({
                "success": True,
                "summary": summary,
                "predictions": results,
                "dashboard_data": dashboard_update,
            })

        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    # ── Fallback sample data ─────────────────────────────────────

    def _get_sample_dashboard_data(self):
        return {
            "kpis": {
                "totalWells": 625, "testWells": 300,
                "rocAuc": 0.773, "accuracy": 0.827,
                "precision": 0.817, "recall": 0.917,
                "totalCostBaseline": 167740000, "totalCostOptimized": 96355000,
                "costSaving": 71385000,
                "stronglyRecommend": 165, "reviewEngineer": 78, "lowPriority": 57,
            },
            "confusionMatrix": {"tn": 83, "fp": 37, "fn": 15, "tp": 165},
            "rocCurve": {
                "fpr": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.31, 0.4, 0.5, 0.7, 1.0],
                "tpr": [0.0, 0.45, 0.62, 0.84, 0.82, 0.83, 0.87, 0.92, 0.96, 0.99, 1.0],
            },
            "heteroIndex": [
                {"index": 1, "label": "High Oil - Low Water", "count": 156, "success_rate": 0.72},
                {"index": 2, "label": "High Oil - High Water", "count": 157, "success_rate": 0.58},
                {"index": 3, "label": "Low Oil - Low Water", "count": 156, "success_rate": 0.42},
                {"index": 4, "label": "Low Oil - High Water", "count": 156, "success_rate": 0.28},
            ],
            "topWells": [
                {"WELL_NAME": "WELL_006", "SUCCESS_PROB": 0.989, "HETERO_INDEX": 1, "ADVISORY": "Strongly Recommend"},
                {"WELL_NAME": "WELL_885", "SUCCESS_PROB": 0.985, "HETERO_INDEX": 1, "ADVISORY": "Strongly Recommend"},
                {"WELL_NAME": "WELL_946", "SUCCESS_PROB": 0.978, "HETERO_INDEX": 1, "ADVISORY": "Strongly Recommend"},
                {"WELL_NAME": "WELL_741", "SUCCESS_PROB": 0.972, "HETERO_INDEX": 2, "ADVISORY": "Strongly Recommend"},
                {"WELL_NAME": "WELL_093", "SUCCESS_PROB": 0.968, "HETERO_INDEX": 1, "ADVISORY": "Strongly Recommend"},
            ],
            "productionData": {
                "beforeIntervention": [1000, 920, 850, 785, 725, 670, 620, 575, 532, 492, 455, 420],
                "duringWorkover": [420, 50, 20, 10],
                "afterIntervention": [10, 650, 780, 820, 845, 860, 870, 875, 878, 880, 881, 882],
            },
        }
