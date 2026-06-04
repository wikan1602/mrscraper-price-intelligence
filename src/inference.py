import os
import joblib
import pandas as pd
import numpy as np
from src.data_loader import clean_data
from src.features import engineer_features

def run_inference(test_file_path: str, output_destination: str, approach: str):
    print(f"\n=== STARTING INFERENCE VIA APPROACH: {approach.upper()} ===")

    # 1. Load shop_freq_map
    freq_map_path = os.path.join("models", "shop_freq_map.pkl")
    if not os.path.exists(freq_map_path):
        raise FileNotFoundError("[ERROR] shop_freq_map.pkl missing. Run training first.")
    shop_freq_map = joblib.load(freq_map_path)

    # 2. Load & clean raw test data
    raw_test = pd.read_csv(test_file_path)
    raw_test['capturedAt'] = pd.to_datetime(raw_test['capturedAt'])
    cleaned_test = clean_data(raw_test)

    # 3. Split anchor vs to-predict SEBELUM feature engineering
    OUTAGE_COLS = ['capturedAt', 'shopId', 'itemId', 'modelId']
    anchor_raw     = cleaned_test[cleaned_test['price'].notna() & (cleaned_test['price'] > 0)].copy()
    to_predict_raw = cleaned_test[~(cleaned_test['price'].notna() & (cleaned_test['price'] > 0))].copy()

    print(f"  Anchor rows (known price) : {len(anchor_raw)}")
    print(f"  Rows to predict           : {len(to_predict_raw)}")

    # 4. Backfill to-predict dengan LOCF dari anchor
    missing_cols = [col for col in cleaned_test.columns if col not in OUTAGE_COLS + ['price']]
    last_known = (
        anchor_raw.groupby('itemId')[missing_cols]
        .last()
        .reset_index()
        .drop_duplicates(subset='itemId')
    )
    to_predict_backfilled = to_predict_raw[OUTAGE_COLS].merge(last_known, on='itemId', how='left')

    cold_start = 0
    for col in missing_cols:
        n = to_predict_backfilled[col].isna().sum()
        if n > 0:
            cold_start += n
            if pd.api.types.is_numeric_dtype(cleaned_test[col]):
                to_predict_backfilled[col] = to_predict_backfilled[col].fillna(cleaned_test[col].median())
            else:
                to_predict_backfilled[col] = to_predict_backfilled[col].fillna(cleaned_test[col].mode()[0])
    print(f"  Cold-start cells patched  : {cold_start:,}")

    # 5. Feature engineering
    anchor_processed     = engineer_features(anchor_raw,            shop_freq_map, is_train=False)
    to_predict_processed = engineer_features(to_predict_backfilled, shop_freq_map, is_train=False)

    # 6. Load model — dan inject entity stats di LUAR loop kalau approach entity
    if approach == "global":
        model_path = os.path.join("models", "global_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError("[ERROR] global_model.pkl missing. Run training first.")
        model = joblib.load(model_path)

    elif approach == "entity":
        model_path = os.path.join("models", "entity_model.pkl")
        stats_path = os.path.join("models", "entity_stats.pkl")
        if not os.path.exists(model_path) or not os.path.exists(stats_path):
            raise FileNotFoundError("[ERROR] Entity model or stats missing. Run training first.")
        model = joblib.load(model_path)
        stats = joblib.load(stats_path)

        # Inject entity stats sekali di sini — bukan di dalam loop
        def add_entity_stats(df):
            out = df.copy()
            out = out.merge(stats['shop_stats'], on='shopId', how='left')
            out = out.merge(stats['item_stats'], on='itemId', how='left')
            out['historical_shop_price_mean'] = out['historical_shop_price_mean'].fillna(stats['global_mean'])
            out['historical_shop_price_std']  = out['historical_shop_price_std'].fillna(0)
            out['historical_item_price_mean'] = out['historical_item_price_mean'].fillna(stats['global_mean'])
            return out

        anchor_processed     = add_entity_stats(anchor_processed)
        to_predict_processed = add_entity_stats(to_predict_processed)

    else:
        raise ValueError(f"[ERROR] Unknown approach: '{approach}'. Use 'global' or 'entity'.")

    FEATURES = [col for col in to_predict_processed.columns if col not in ['price', 'log_price', 'scrape_date']]

    # 7. Kalibrasi per hari
    anchor_processed['scrape_date']      = anchor_raw['capturedAt'].dt.date.values
    to_predict_processed['scrape_date']  = to_predict_raw['capturedAt'].dt.date.values

    results = []
    for date in sorted(anchor_processed['scrape_date'].unique()):
        anchor_day  = anchor_processed[anchor_processed['scrape_date'] == date]
        predict_day = to_predict_processed[to_predict_processed['scrape_date'] == date]
        anchor_prices = anchor_raw[anchor_raw['capturedAt'].dt.date == date]['price'].values

        anchor_preds = np.expm1(model.predict(anchor_day[FEATURES]))
        base_preds   = np.expm1(model.predict(predict_day[FEATURES]))

        if approach == "global":
            calib_factor = np.median(anchor_prices / (anchor_preds + 1e-5))
            preds = base_preds * calib_factor
            print(f"  {date} — calib factor: {calib_factor:.4f}, rows: {len(predict_day)}")

        elif approach == "entity":
            anchor_eval = pd.DataFrame({
                'shopId': anchor_day['shopId'].astype(str).values,
                'price' : anchor_prices,
                'pred'  : anchor_preds
            })
            anchor_eval['ratio'] = anchor_eval['price'] / (anchor_eval['pred'] + 1e-5)
            shop_calib   = anchor_eval.groupby('shopId')['ratio'].median().to_dict()
            global_calib = float(np.median(anchor_eval['ratio']))
            calib_map    = predict_day['shopId'].astype(str).map(shop_calib).fillna(global_calib)
            preds        = base_preds * calib_map.values
            print(f"  {date} — global calib: {global_calib:.4f}, rows: {len(predict_day)}")

        day_result = to_predict_raw[to_predict_raw['capturedAt'].dt.date == date].copy()
        day_result['price'] = np.round(preds).astype(int)
        results.append(day_result)

    # 8. Gabung anchor + prediksi, simpan
    final_df = pd.concat([anchor_raw] + results, ignore_index=True)
    final_df = final_df.sort_values('capturedAt').reset_index(drop=True)
    final_df['price'] = final_df['price'].astype(int)

    os.makedirs(os.path.dirname(output_destination), exist_ok=True)
    final_df.to_csv(output_destination, index=False)
    print(f"\n[INFO] Saved {len(final_df):,} rows → {output_destination}")
    print("=== INFERENCE COMPLETED SUCCESSFULLY ===")