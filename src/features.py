import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Executes the production feature engineering pipeline:
    1. Extracts continuous cyclical temporal signals.
    2. Computes robust boundary price positions with division-by-zero protection.
    3. Encodes categorical identifiers into high-performance pandas categories.
    """
    df_feat = df.copy()
    
    # 1. Cyclical Temporal Engineering
    df_feat['hour'] = df_feat['capturedAt'].dt.hour
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)
    
    # 2. Advanced E-Commerce Pricing Logic & Outlier Protection
    # Prevent division overflow (10^14) when item_price_max == item_price_min
    denominator = df_feat['item_price_max'] - df_feat['item_price_min']
    df_feat['price_position'] = np.where(
        denominator == 0, 
        0.5, # Safe default center anchor for single-variant products
        (df_feat['priceBeforeDiscount'] - df_feat['item_price_min']) / (denominator + 1e-5)
    )
    
    # Prevent division by zero if priceBeforeDiscount is missing or 0
    df_feat['discount_ratio'] = np.where(
        df_feat['priceBeforeDiscount'] == 0,
        0.0,
        df_feat['raw_discount'] / (df_feat['priceBeforeDiscount'] + 1e-5)
    )
    
    # 3. Frequency Encoding Proxy for High-Cardinality Entities
    df_feat['shop_appearance_count'] = df_feat['shopId'].map(df_feat['shopId'].value_counts())
    
    # 4. Strict Type Casting for Native LightGBM Categorical Handling
    categorical_columns = [
        'shopId', 'itemId', 'modelId', 'cat_id', 'brand',
        'is_free_shipping', 'is_pre_order', 'is_official_shop', 
        'is_verified', 'is_preferred_plus_seller'
    ]
    
    for col in categorical_columns:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype('category')
            
    # 5. Drop raw metadata columns that cause data leakage or overfitting
    drop_metadata = ['capturedAt', 'hour']
    df_feat = df_feat.drop(columns=[col for col in drop_metadata if col in df_feat.columns], errors='ignore')
    
    return df_feat