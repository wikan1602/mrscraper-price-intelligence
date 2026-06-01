import pandas as pd
import numpy as np

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw e-commerce price dataset from a given CSV path
    and ensures proper datetime parsing.
    """
    print(f"[INFO] Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    df['capturedAt'] = pd.to_datetime(df['capturedAt'])
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial data cleaning based on EDA insights:
    1. Handles missing brand tokens.
    2. Drops hyper-sparse stock columns to prevent model noise.
    """
    df_clean = df.copy()
    
    # Handle missing brand tokens
    if 'brand' in df_clean.columns:
        df_clean['brand'] = df_clean['brand'].fillna('Unknown')
        
    # Drop hyper-sparse stock features (98.7% missingness)
    sparse_cols = ['stock', 'normal_stock', 'promotionId']
    df_clean = df_clean.drop(columns=[col for col in sparse_cols if col in df_clean.columns], errors='ignore')
    
    print(f"[INFO] Data cleaning completed. Current shape: {df_clean.shape}")
    return df_clean