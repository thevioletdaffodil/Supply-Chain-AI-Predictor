# -*- coding: utf-8 -*-
"""
Supply Chain Total Lead Time Prediction & Assistant Pipeline
Architecture: Dual-Band Regression (Expected Mean + Safety Quantile) with Volatility Encoding.
"""

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
# GLOBAL CONFIGURATION & CONSTANTS 
# ---------------------------------------------------------
BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "SYNTHETIC_PO_DATA.csv") 
RAG_DB_PATH = os.path.join(BASE_DIR, "predicted_lead_times_for_rag.csv")

HIGH_RISK_VENDORS = [
    'Strata Global', 'Quantum Components'
]

HIGH_RISK_CITIES = [
    'Shenzhen', 'Mumbai'
]

# ---------------------------------------------------------
# 1. DATA PREPROCESSING (Total Lead Time Focus)
# ---------------------------------------------------------
def load_and_preprocess_data(filepath):
    print("Loading historical data...")
    df_raw = pd.read_csv(filepath)

    print("Cleaning dates and handling nulls...")
    df_raw['PO_DATE'] = pd.to_datetime(df_raw['PO_DATE'], errors='coerce', dayfirst=True)
    df_raw['ACTUALDATEOFRECEIPT'] = pd.to_datetime(df_raw['ACTUALDATEOFRECEIPT'], errors='coerce', dayfirst=True)

    df_raw['RECEIPTQUANTITY'] = pd.to_numeric(
        df_raw['RECEIPTQUANTITY'].astype(str).str.replace(',', ''), errors='coerce'
    ).fillna(0)

    print("Compressing split deliveries...")
    df = df_raw.groupby('PO_NO').agg({
        'PO_DATE': 'first', 'VENDOR': 'first', 'CITY': 'first', 'COUNTRY': 'first',
        'PURCHASE_LEAD_TIME': 'first', 'ITEM': 'first', 'QUANTITY': 'first',
        'RECEIPTQUANTITY': 'sum', 'ACTUALDATEOFRECEIPT': 'max'
    }).reset_index()

    print("Applying risk flags...")
    df['Is_High_Risk_Vendor'] = df['VENDOR'].isin(HIGH_RISK_VENDORS).astype(int)
    df['Is_High_Risk_City'] = df['CITY'].isin(HIGH_RISK_CITIES).astype(int)
    df['Severe_Risk_Warning'] = ((df['Is_High_Risk_Vendor'] == 1) & (df['Is_High_Risk_City'] == 1)).astype(int)

    df['CITY'] = df['CITY'].fillna('Unknown')
    df['COUNTRY'] = df['COUNTRY'].fillna('Unknown')
    df['ITEM'] = df['ITEM'].fillna('Unknown')

    # Drop rows where we cannot calculate the actual total lead time
    original_rows = df.shape[0]
    df.dropna(subset=['PO_DATE', 'ACTUALDATEOFRECEIPT'], inplace=True)
    print(f"Dropped {original_rows - df.shape[0]} rows missing crucial timeline dates.")

    # TARGET: Total Lead Time Days (No human schedule dependency)
    df['Total_Lead_Time_Days'] = (df['ACTUALDATEOFRECEIPT'] - df['PO_DATE']).dt.days

    # Filter out anomalous data entry errors
    df = df[df['Total_Lead_Time_Days'] >= 0]

    df['PO_MONTH'] = df['PO_DATE'].dt.month
    df['PO_YEAR'] = df['PO_DATE'].dt.year
    df['PO_DAYOFTHEWEEK'] = df['PO_DATE'].dt.dayofweek
    df['Is_International'] = (df['COUNTRY'] != 'Domestic').astype(int)

    return df

# ---------------------------------------------------------
# 2. MACHINE LEARNING PIPELINE (Dual-Band Lead Time)
# ---------------------------------------------------------
def train_and_evaluate_models(df):
    cols_to_drop = [
        'PO_NO', 'ACTUALDATEOFRECEIPT', 'PO_DATE', 'RECEIPTQUANTITY', 'SCHEDULEDDATEOFDELIVERY'
    ]

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns] + ['Total_Lead_Time_Days'], axis=1)
    y = df['Total_Lead_Time_Days']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Applying Target Encoding with Volatility metrics...")
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
    encoding_maps = {}

    for col in categorical_cols:
        threshold = 3
        counts = X_train[col].value_counts()
        rare_items = counts[counts < threshold].index

        X_train[col] = X_train[col].replace(rare_items, 'Other')
        X_test[col] = X_test[col].replace(rare_items, 'Other')

        grouped = y_train.groupby(X_train[col])
        mean_map = grouped.mean().to_dict()
        std_map = grouped.std().fillna(0).to_dict()

        encoding_maps[col] = {'mean': mean_map, 'std': std_map}

        X_train[f"{col}_mean"] = X_train[col].map(mean_map).fillna(0)
        X_train[f"{col}_std"] = X_train[col].map(std_map).fillna(0)
        X_test[f"{col}_mean"] = X_test[col].map(mean_map).fillna(0)
        X_test[f"{col}_std"] = X_test[col].map(std_map).fillna(0)

        X_train.drop(columns=[col], inplace=True)
        X_test.drop(columns=[col], inplace=True)

    joblib.dump(encoding_maps, os.path.join(BASE_DIR, 'target_encoding_maps.joblib'))
    joblib.dump(X_train.columns.tolist(), os.path.join(BASE_DIR, 'expected_columns.joblib'))

    # MODEL A: EXPECTED LEAD TIME (Mean Estimation)
    print("\nTraining Model A: Expected Total Lead Time (Mean)...")
    reg_mean = GradientBoostingRegressor(loss='squared_error', random_state=42, n_estimators=150, learning_rate=0.05, max_depth=5)
    reg_mean.fit(X_train, y_train)
    joblib.dump(reg_mean, os.path.join(BASE_DIR, 'stage2_mean_regressor.joblib'))

    # MODEL B: MAXIMUM LEAD TIME BUFFER (95th Percentile)
    print("Training Model B: Maximum Lead Time Buffer...")
    reg_quant = GradientBoostingRegressor(loss='quantile', alpha=0.95, random_state=42, n_estimators=150, learning_rate=0.05, max_depth=5)
    reg_quant.fit(X_train, y_train)
    joblib.dump(reg_quant, os.path.join(BASE_DIR, 'stage2_quant_regressor.joblib'))

    preds_mean = np.maximum(0, reg_mean.predict(X_test))
    preds_quant = np.maximum(0, reg_quant.predict(X_test))

    print("\n--- DUAL-BAND EVALUATION (Total Lead Time) ---")
    print("MODEL A (Accuracy Focus):")
    print(f"  Mean Absolute Error: {mean_absolute_error(y_test, preds_mean):.2f} days off on average")
    print(f"  R2 Score: {r2_score(y_test, preds_mean):.3f}")

    print("\nMODEL B (Safety Focus):")
    coverage = (y_test <= preds_quant).sum() / len(y_test) * 100
    print(f"  Business Safety Coverage: {coverage:.1f}% of orders arrived within this buffer limit.")

    return reg_mean, reg_quant

# ---------------------------------------------------------
# 3. RAG KNOWLEDGE BASE GENERATOR
# ---------------------------------------------------------
def generate_rag_database(df):
    print("\nGenerating RAG context database...")
    encoding_maps = joblib.load(os.path.join(BASE_DIR, 'target_encoding_maps.joblib'))

    if 'VENDOR' in encoding_maps:
        df['Vendor_Avg_Time'] = df['VENDOR'].map(encoding_maps['VENDOR']['mean']).fillna(0)
        df['Vendor_Volatility'] = df['VENDOR'].map(encoding_maps['VENDOR']['std']).fillna(0)
    else:
        df['Vendor_Avg_Time'] = 0
        df['Vendor_Volatility'] = 0

    def build_context(row):
        po_date = row['PO_DATE'].date() if pd.notnull(row['PO_DATE']) else "unknown date"

        base = (f"Historical Order PO {row['PO_NO']} was placed on {po_date} with {row['VENDOR']} "
                f"located in {row['CITY']}, {row['COUNTRY']}. "
                f"Ordered item: {row['ITEM']} (Quantity: {row.get('QUANTITY', 'unknown')}). ")

        outcome = f"It took a total of {int(row['Total_Lead_Time_Days'])} days to arrive. "

        analysis = (f"Analytical Context: Contractual purchase lead time was {row.get('PURCHASE_LEAD_TIME', 0)} days. "
                    f"Vendor historical profile: Usually takes {round(row['Vendor_Avg_Time'], 1)} days to fulfill orders, "
                    f"with a volatility (std dev) of {round(row['Vendor_Volatility'], 1)} days.")

        return base + outcome + analysis

    df['RAG_Context'] = df.apply(build_context, axis=1)
    export_cols = ['PO_NO', 'VENDOR', 'CITY', 'ITEM', 'COUNTRY', 'Total_Lead_Time_Days', 'RAG_Context']
    df[export_cols].to_csv(RAG_DB_PATH, index=False)
    print(f"Database saved to {RAG_DB_PATH}")
