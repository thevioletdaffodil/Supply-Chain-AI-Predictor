import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# GLOBAL CONFIGURATION & CONSTANTS (SANITIZED FOR GITHUB)
# ---------------------------------------------------------
BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "SYNTHETIC_PO_DATA.csv") # Sanitized local path
RAG_DB_PATH = os.path.join(BASE_DIR, "predicted_lead_times_for_rag.csv")

# Sanitized dummy tracking targets for portfolio presentation
HIGH_RISK_VENDORS = [
    'Strata Global', 'Quantum Components'
]

HIGH_RISK_CITIES = [
    'Shenzhen', 'Mumbai'
]

# ---------------------------------------------------------
# 6. BATCH PRODUCTION SCRIPT AUTOMATION
# ---------------------------------------------------------
def predict_batch_orders(new_csv_path, output_csv_path):
    if not os.path.exists(new_csv_path):
        print(f"Source batch file {new_csv_path} not found.")
        return None

    print(f"Loading new unscheduled orders from {new_csv_path}...")
    df_new = pd.read_csv(new_csv_path)

    BASE_DIR = os.getcwd()
    reg_mean = joblib.load(os.path.join(BASE_DIR, 'stage2_mean_regressor.joblib'))
    reg_quant = joblib.load(os.path.join(BASE_DIR, 'stage2_quant_regressor.joblib'))
    encoding_maps = joblib.load(os.path.join(BASE_DIR, 'target_encoding_maps.joblib'))
    expected_cols = joblib.load(os.path.join(BASE_DIR, 'expected_columns.joblib'))

    print("Formatting dates and extracting time features...")
    df_new['PO_DATE'] = pd.to_datetime(df_new['PO_DATE'], errors='coerce', dayfirst=True)
    df_new['PO_MONTH'] = df_new['PO_DATE'].dt.month
    df_new['PO_YEAR'] = df_new['PO_DATE'].dt.year
    df_new['PO_DAYOFTHEWEEK'] = df_new['PO_DATE'].dt.dayofweek
    df_new['Is_International'] = (df_new.get('COUNTRY', 'Domestic') != 'Domestic').astype(int)

    df_new['Is_High_Risk_Vendor'] = df_new['VENDOR'].isin(HIGH_RISK_VENDORS).astype(int)
    df_new['Is_High_Risk_City'] = df_new['CITY'].isin(HIGH_RISK_CITIES).astype(int)
    df_new['Severe_Risk_Warning'] = ((df_new['Is_High_Risk_Vendor'] == 1) & (df_new['Is_High_Risk_City'] == 1)).astype(
        int)

    print("Applying historical risk profiles to Vendors, Cities, and Items...")
    for col, maps in encoding_maps.items():
        if col in df_new.columns:
            df_new[f"{col}_mean"] = df_new[col].map(maps['mean']).fillna(0)
            df_new[f"{col}_std"] = df_new[col].map(maps['std']).fillna(0)

    # Reindex forces unexpected leakage columns to drop safely
    X_score = df_new.reindex(columns=expected_cols, fill_value=0)

    print("Running Dual-Band AI Predictions...")
    expected_days = np.maximum(0, reg_mean.predict(X_score))
    max_days = np.maximum(expected_days, reg_quant.predict(X_score))

    df_new['AI_Expected_Lead_Time_Days'] = np.round(expected_days).astype(int)
    df_new['AI_Max_Safety_Buffer_Days'] = np.round(max_days).astype(int)

    df_new['AI_Expected_Arrival_Date'] = df_new['PO_DATE'] + pd.to_timedelta(df_new['AI_Expected_Lead_Time_Days'],
                                                                             unit='D')
    df_new['AI_Safest_Schedule_Date'] = df_new['PO_DATE'] + pd.to_timedelta(df_new['AI_Max_Safety_Buffer_Days'],
                                                                            unit='D')

    # Reformat timestamps to human-readable strings
    df_new['AI_Expected_Arrival_Date'] = df_new['AI_Expected_Arrival_Date'].dt.strftime('%d-%m-%Y')
    df_new['AI_Safest_Schedule_Date'] = df_new['AI_Safest_Schedule_Date'].dt.strftime('%d-%m-%Y')

    export_columns = [
        'PO_DATE', 'VENDOR', 'ITEM', 'QUANTITY',
        'AI_Expected_Lead_Time_Days', 'AI_Expected_Arrival_Date',
        'AI_Max_Safety_Buffer_Days', 'AI_Safest_Schedule_Date'
    ]

    final_output = df_new[export_columns]
    final_output.to_csv(output_csv_path, index=False)
    print(f"\nSuccess! Analyzed {len(df_new)} orders. Results saved to: {output_csv_path}")

    return final_output

if __name__ == "__main__":
    predict_batch_orders("unscheduled_orders.csv", "AI_Scheduled_Orders_Output.csv")