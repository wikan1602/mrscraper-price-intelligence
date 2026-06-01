import os
import joblib
import pandas as pd
import numpy as np
from src.data_loader import clean_data
from src.features import engineer_features

def run_inference(test_file_path: str, output_destination: str, approach: str):
    print(f"\n=== STARTING INFERENCE VIA APPROACH: {approach.upper()} ===")
    
    # 1. Load Raw Test Data
    raw_test = pd.read_csv(test_file_path)
    raw_test['capturedAt'] = pd.to_datetime(raw_test['capturedAt'])
    processed_test = engineer_features(clean_data(raw_test), is_train=False)
    
    # 2. Setup Approach Configurations
    if approach == "global":
        model_path = os.path.join("models", "global_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError("[ERROR] Global model artifact missing. Run training first.")
        model = joblib.load(model_path)
        features_list = [col for col in processed_test.columns if col not in ['price', 'log_price']]
        
    elif approach == "entity":
        model_path = os.path.join("models", "entity_model.pkl")
        stats_path = os.path.join("models", "entity_stats.pkl")
        if not os.path.exists(model_path) or not os.path.exists(stats_path):
            raise FileNotFoundError("[ERROR] Entity model or statistics artifacts missing. Run training first.")
            
        model = joblib.load(model_path)
        stats = joblib.load(stats_path)
        
        # Map training priors to test data to prevent cold-start leakage
        processed_test = processed_test.merge(stats['shop_stats'], on='shopId', how='left')
        processed_test = processed_test.merge(stats['item_stats'], on='itemId', how='left')
        
        # Handle Cold-Start fallback gracefully using loaded global priors
        processed_test['historical_shop_price_mean'] = processed_test['historical_shop_price_mean'].fillna(stats['global_mean'])
        processed_test['historical_item_price_mean'] = processed_test['historical_item_price_mean'].fillna(stats['global_mean'])
        processed_test['historical_shop_price_std'] = processed_test['historical_shop_price_std'].fillna(0)
        
        features_list = [col for col in processed_test.columns if col not in ['price', 'log_price']]
        
    # 3. Dynamic Anchor-Set Isolation & Calibration Calculation
    anchor_mask = processed_test['price'].notnull() & (processed_test['price'] > 0)
    anchor_set = processed_test[anchor_mask].copy()
    
    # Establish a macro fallback calibration factor
    global_calib_factor = 1.0
    if len(anchor_set) > 0:
        anchor_preds_log = model.predict(anchor_set[features_list])
        anchor_preds_actual = np.expm1(anchor_preds_log)
        global_calib_factor = np.median(anchor_set['price'] / (anchor_preds_actual + 1e-5))
    
    # 4. Apply Custom Calibration Hierarchy Based on Selected Approach
    base_preds_log = model.predict(processed_test[features_list])
    base_preds_actual = np.expm1(base_preds_log)
    
    if approach == "global":
        print(f"[INFO] Applying Macro Platform Calibration Factor: {global_calib_factor:.4f}")
        calibrated_predictions = base_preds_actual * global_calib_factor
        
    elif approach == "entity":
        print("[INFO] Applying Entity-Level Shop Calibration with Fallbacks...")
        anchor_set['pred_base'] = np.expm1(model.predict(anchor_set[features_list]))
        anchor_set['ratio'] = anchor_set['price'] / (anchor_set['pred_base'] + 1e-5)
        
        # Map per-shop multipliers
        shop_calibration_map = anchor_set.groupby('shopId')['ratio'].median().to_dict()
        
        # Fallback hierarchy: Use shop multiplier if anchor exists, else fallback to global multiplier
        processed_test['final_calib_factor'] = processed_test['shopId'].map(shop_calibration_map).fillna(global_calib_factor)
        calibrated_predictions = base_preds_actual * processed_test['final_calib_factor']

    # 5. Build and Save Final Submission Payload
    output_df = raw_test.copy()
    output_df['price'] = np.where(
        output_df['price'].isnull() | (output_df['price'] <= 0),
        np.round(calibrated_predictions),
        output_df['price']
    )
    output_df['price'] = output_df['price'].astype(int)
    
    print(f"[INFO] Saving finalized submission to: {output_destination}")
    output_df.to_csv(output_destination, index=False)
    print("=== INFERENCE COMPLETED SUCCESSFULLY ===")