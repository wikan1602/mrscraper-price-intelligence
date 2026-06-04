# FIXED features.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame, shop_freq_map: pd.Series, is_train: bool = True) -> pd.DataFrame:
    """
    Executes the production feature engineering pipeline.
    
    Args:
        df: Input DataFrame (must contain capturedAt column).
        shop_freq_map: pd.Series of shopId value_counts computed from TRAINING data only.
                       Pass the same map for both train and inference to prevent leakage.
        is_train: Reserved for future train-only transformations.
    """
    df_feat = df.copy()
    
    # 1. Cyclical Temporal Engineering
    df_feat['hour']     = df_feat['capturedAt'].dt.hour
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)
    
    # 2. Pricing Boundary Features & Zero-Division Protection
    denominator = df_feat['item_price_max'] - df_feat['item_price_min']
    df_feat['price_position'] = np.where(
        denominator == 0,
        0.5,
        (df_feat['priceBeforeDiscount'] - df_feat['item_price_min']) / (denominator + 1e-5)
    )
    df_feat['discount_ratio'] = np.where(
        df_feat['priceBeforeDiscount'] == 0,
        0.0,
        df_feat['raw_discount'] / (df_feat['priceBeforeDiscount'] + 1e-5)
    )
    
    # 3. Frequency Encoding — always from train map (anti-leakage)
    df_feat['shop_appearance_count'] = df_feat['shopId'].map(shop_freq_map).fillna(0)
    
    # 4. Categorical Type Casting for LightGBM
    categorical_columns = [
        'shopId', 'itemId', 'modelId', 'cat_id', 'brand',
        'is_free_shipping', 'is_pre_order', 'is_official_shop',
        'is_verified', 'is_preferred_plus_seller'
    ]
    for col in categorical_columns:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype('category')
    
    # 5. Drop metadata columns
    drop_cols = ['capturedAt', 'hour', 'scrape_date']
    df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], errors='ignore')
    
    return df_feat