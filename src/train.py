import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from src.data_loader import load_raw_data, clean_data
from src.features import engineer_features

TRAIN_DATA_PATH = os.path.join("data", "raw", "ecommerce_price_prediction-train.csv")
RANDOM_SEED = 42

def run_training():
    print("=== STARTING MULTI-APPROACH MODEL TRAINING PIPELINE ===")
    os.makedirs("models", exist_ok=True)
    
    # 1. Load and clean
    raw_df    = load_raw_data(TRAIN_DATA_PATH)
    cleaned_df = clean_data(raw_df)
    
    # 2. Compute shop frequency map dari train (anti-leakage)
    shop_freq_map = cleaned_df['shopId'].value_counts()
    joblib.dump(shop_freq_map, os.path.join("models", "shop_freq_map.pkl"))
    print("[INFO] Shop frequency map saved.")
    
    # 3. Feature engineering
    print("[INFO] Engineering base features...")
    base_df = engineer_features(cleaned_df, shop_freq_map=shop_freq_map, is_train=True)
    
    # ==========================================
    # APPROACH 1: GLOBAL MARKETPLACE MODEL
    # ==========================================
    print("\n--- Training Approach 1: Global Marketplace Model ---")
    features_a1 = [col for col in base_df.columns if col not in ['price', 'log_price']]
    X_train_a1  = base_df[features_a1]
    y_train_log = np.log1p(base_df['price'])
    
    # Best params dari Optuna (validation notebook)
    model_a1 = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=80,
        max_depth=11,
        min_child_samples=58,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )
    model_a1.fit(X_train_a1, y_train_log)
    joblib.dump(model_a1, os.path.join("models", "global_model.pkl"))
    print("[SUCCESS] Approach 1 model saved.")

    # ==========================================
    # APPROACH 2: ENTITY-CONDITIONED MODEL
    # ==========================================
    print("\n--- Training Approach 2: Entity-Conditioned Model ---")
    
    shop_stats = cleaned_df.groupby('shopId')['price'].agg(['mean', 'std']).reset_index()
    shop_stats.columns = ['shopId', 'historical_shop_price_mean', 'historical_shop_price_std']
    
    item_stats = cleaned_df.groupby('itemId')['price'].agg(['mean']).reset_index()
    item_stats.columns = ['itemId', 'historical_item_price_mean']
    
    entity_stats = {
        'shop_stats' : shop_stats,
        'item_stats' : item_stats,
        'global_mean': cleaned_df['price'].mean()
    }
    joblib.dump(entity_stats, os.path.join("models", "entity_stats.pkl"))
    
    global_mean = entity_stats['global_mean']
    base_df_a2  = base_df.merge(shop_stats, on='shopId', how='left')
    base_df_a2  = base_df_a2.merge(item_stats, on='itemId', how='left')
    base_df_a2['historical_shop_price_mean'] = base_df_a2['historical_shop_price_mean'].fillna(global_mean)
    base_df_a2['historical_shop_price_std']  = base_df_a2['historical_shop_price_std'].fillna(0)
    base_df_a2['historical_item_price_mean'] = base_df_a2['historical_item_price_mean'].fillna(global_mean)
    
    features_a2 = [col for col in base_df_a2.columns if col not in ['price', 'log_price']]
    X_train_a2  = base_df_a2[features_a2]
    
    # Best params dari Optuna (validation notebook)
    model_a2 = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.09,
        num_leaves=50,
        max_depth=11,
        min_child_samples=32,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )
    model_a2.fit(X_train_a2, y_train_log)
    joblib.dump(model_a2, os.path.join("models", "entity_model.pkl"))
    print("[SUCCESS] Approach 2 model and stats saved.")
    print("\n=== ALL MODELS TRAINED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_training()