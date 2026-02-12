import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Add src to path
ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from utils import clinical_preproc as pp
from utils import model_saving as mu

# Config
DATA_PATH = ROOT_PATH / 'data' / 'processed'
OUTPUT_PATH = ROOT_PATH / 'data' / 'output'
MODEL_PATH = ROOT_PATH / 'models'
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

def main():
    print("--- 3. Model Training: LSTM & XGBoost (Physionet) ---")

    # 1. Load Data
    print("Loading Data...")
    ws = pd.read_csv(DATA_PATH / 'ward_stays.csv', low_memory=False)
    vitals = pd.read_csv(DATA_PATH / 'vitals.csv', low_memory=False)
    try:
        labs = pd.read_csv(DATA_PATH / 'labs.csv', low_memory=False)
        demo = pd.read_csv(DATA_PATH / 'demographic.csv', low_memory=False)
    except:
        labs, demo = None, None

    # 2. Preprocessing
    print("Preprocessing...")
    ws_filtered, valid_ids = pp.filter_valid_stays(ws, min_hours=48)
    df_long = pp.build_long_format(ws_filtered, vitals, labs, demo, valid_ids)
    df_clean = pp.clean_vitals_range(df_long)
    
    # 3. Resample
    print("Resampling Time Series (2h)...")
    ts_df = pp.resample_and_pivot(df_clean, freq='2h')
    
    if ts_df.empty: return

    # 4. Get Labels
    print("Aligning Labels...")
    df_labeled = pp.get_labels(ws_filtered)
    y_series = df_labeled.set_index('stay_id')['y']

    # ==========================================
    # MODEL A: XGBoost (Summary Stats)
    # ==========================================
    print("\n--- Training XGBoost ---")
    
    X_xgb = pp.generate_summary_features(ts_df)
    
    # Strict Alignment
    common_ids = X_xgb.index.intersection(y_series.index)
    X_xgb = X_xgb.loc[common_ids].fillna(-999)
    y_xgb = y_series.loc[common_ids]
    
    print(f"XGBoost Data Shape: {X_xgb.shape} (Labels: {len(y_xgb)})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
    )
    
    model_xgb = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        eval_metric='logloss'
    )
    model_xgb.fit(X_train, y_train)
    
    # XGBoost Predictions
    preds_xgb = model_xgb.predict(X_test)
    y_pred_xgb_proba = model_xgb.predict_proba(X_test)[:, 1]
    
    f1_xgb = f1_score(y_test, preds_xgb)
    print(f"XGBoost Test F1: {f1_xgb:.4f}")
    
    mu.export_xgboost_json(model_xgb, X_train, threshold=0.14, output_dir=MODEL_PATH, model_name='xgboost_xb')

    # ==========================================
    # MODEL B: LSTM (Time Series)
    # ==========================================
    print("\n--- Training LSTM ---")
    
    # 1. Prepare Data
    # Fix: Capture all 3 return values
    X_lstm_raw, lstm_ids, max_len = pp.prepare_lstm_data(ts_df, max_seq_len=36)
    y_lstm = y_series.loc[lstm_ids].values

    # 2. Split
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_lstm_raw, y_lstm, test_size=0.2, stratify=y_lstm, random_state=42
    )

    # 3. Scale (Robust Scaling)
    from sklearn.preprocessing import StandardScaler
    ns, ts, nf = X_train_l.shape
    
    # Reshape to 2D for scaling, then back to 3D
    # Using epsilon to avoid division by zero if std is 0
    scaler = StandardScaler()
    X_train_2d = X_train_l.reshape(-1, nf)
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(ns, ts, nf)
    
    X_test_2d = X_test_l.reshape(-1, nf)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test_l.shape[0], ts, nf)
    
    # Fix NaN inputs if scaling failed (e.g. constant columns)
    X_train_scaled = np.nan_to_num(X_train_scaled)
    X_test_scaled = np.nan_to_num(X_test_scaled)

    # 4. Class Balancing (Oversampling)
    X_pos = X_train_scaled[y_train_l == 1]
    y_pos = y_train_l[y_train_l == 1]
    X_neg = X_train_scaled[y_train_l == 0]
    y_neg = y_train_l[y_train_l == 0]

    if len(y_pos) > 0:
        multiplier = int(len(y_neg) / len(y_pos))
        X_pos_oversampled = np.repeat(X_pos, multiplier, axis=0)
        y_pos_oversampled = np.repeat(y_pos, multiplier, axis=0)
        
        X_train_bal = np.concatenate([X_neg, X_pos_oversampled], axis=0)
        y_train_bal = np.concatenate([y_neg, y_pos_oversampled], axis=0)
    else:
        print("Warning: No positive cases in training set!")
        X_train_bal, y_train_bal = X_train_scaled, y_train_l

    # Shuffle
    from sklearn.utils import shuffle
    X_train_bal, y_train_bal = shuffle(X_train_bal, y_train_bal, random_state=42)

    # 5. Model Definition
    # Note: TF 2.16+ prefers Input layer, but Masking layer as first layer is still valid in Sequential
    model_lstm = Sequential([
        Masking(mask_value=0.0, input_shape=(max_len, nf)), 
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    # Using 'accuracy' as a metric so it appears in history keys
    model_lstm.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

    print("Starting LSTM training...")
    history = model_lstm.fit(
        X_train_bal, y_train_bal,
        validation_split=0.1,
        epochs=15,
        batch_size=32,
        verbose=1
    )

    # Predictions
    y_pred_lstm = model_lstm.predict(X_test_scaled).flatten()
    
    # Save Model
    lstm_path = MODEL_PATH / 'lstm_model.h5'
    model_lstm.save(lstm_path) # Legacy .h5 format 
    print(f"LSTM model saved to {lstm_path}")

    # ==========================================
    # PLOTTING
    # ==========================================
    print("Generating Plots...")

    # --- PLOT 1: LSTM Training History ---
    if 'history' in locals():
        plt.figure(figsize=(12, 5))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('LSTM Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy / AUC
        plt.subplot(1, 2, 2)
        # Check what metrics are actually available
        available_metrics = list(history.history.keys())
        print(f"Available metrics: {available_metrics}")
        
        # Prefer AUC, fallback to accuracy
        if 'auc' in available_metrics:
            metric = 'auc'
        elif 'AUC' in available_metrics:
            metric = 'AUC'
        else:
            metric = 'accuracy'
            
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(f'LSTM Model {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / 'lstm_training_history.png')
        plt.close()
        print(" > Saved LSTM history plot.")

    # --- PLOT 2: ROC Curve Comparison ---
    plt.figure(figsize=(8, 6))
    
    # LSTM ROC (Uses y_test_l)
    fpr_lstm, tpr_lstm, _ = roc_curve(y_test_l, y_pred_lstm)
    auc_val_lstm = auc(fpr_lstm, tpr_lstm)
    plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM (AUC = {auc_val_lstm:.3f})', color='blue')
    
    # XGBoost ROC (Uses y_test)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb_proba)
    auc_val_xgb = auc(fpr_xgb, tpr_xgb)
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_val_xgb:.3f})', color='green')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.savefig(OUTPUT_PATH / 'roc_curve_comparison.png')
    plt.close()
    print(" > Saved ROC comparison plot.")

    # --- PLOT 3: Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # LSTM CM
    y_pred_lstm_bin = (y_pred_lstm > 0.5).astype(int)
    cm_lstm = confusion_matrix(y_test_l, y_pred_lstm_bin)
    disp_lstm = ConfusionMatrixDisplay(confusion_matrix=cm_lstm, display_labels=['Stable', 'Deteriorated'])
    disp_lstm.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('LSTM Confusion Matrix')
    
    # XGBoost CM
    y_pred_xgb_bin = (y_pred_xgb_proba > 0.5).astype(int)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb_bin)
    disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['Stable', 'Deteriorated'])
    disp_xgb.plot(ax=axes[1], cmap='Greens', values_format='d')
    axes[1].set_title('XGBoost Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'confusion_matrices.png')
    plt.close()
    print(" > Saved Confusion Matrices.")

    # --- PLOT 4: SHAP ---
    try:
        import shap
        print(" > Generating SHAP plots...")
        explainer = shap.TreeExplainer(model_xgb)
        # Use X_test from XGBoost split
        shap_values = explainer.shap_values(X_test)
        
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / 'shap_summary_xgb.png')
        plt.close()
        print(" > Saved SHAP plot.")
    except Exception as e:
        print(f" ! SHAP plot skipped: {e}")

    print(f"All plots saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()