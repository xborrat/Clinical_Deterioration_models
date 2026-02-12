import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from kmodes.kprototypes import KPrototypes 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
    print("--- 2. K-Prototypes/K-Means (Physionet) ---")

    # 1. Load Data
    print("Loading CSVs...")
    ws = pd.read_csv(DATA_PATH / 'ward_stays.csv', low_memory=False)
    vitals = pd.read_csv(DATA_PATH / 'vitals.csv', low_memory=False)
    try:
        labs = pd.read_csv(DATA_PATH / 'labs.csv', low_memory=False)
        demo = pd.read_csv(DATA_PATH / 'demographic.csv', low_memory=False)
    except:
        labs, demo = None, None

    # 2. Preproc
    print("Preprocessing...")
    ws_filtered, valid_ids = pp.filter_valid_stays(ws, min_hours=48)
    df_long = pp.build_long_format(ws_filtered, vitals, labs, demo, valid_ids)
    df_clean = pp.clean_vitals_range(df_long)

    # 3. Resample
    print("Resampling (2h)...")
    ts_df = pp.resample_and_pivot(df_clean, freq='2h')

    if ts_df.empty: return

    # 4. Features
    print("Generating Features...")
    X = pp.generate_summary_features(ts_df)
    
    # Convert string numbers
    for col in X.columns:
        converted = pd.to_numeric(X[col], errors='coerce')
        if converted.notna().sum() > 0:
            X[col] = converted
    
    print(f"Original shape: {X.shape}")

    # 2. Separate Numeric and Categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols_names = X.select_dtypes(exclude=[np.number]).columns

    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # 4. Impute Categorical with MODE
    if len(cat_cols_names) > 0:
        for c in cat_cols_names:
            if X[c].notna().sum() > 0:
                mode_val = X[c].mode()[0]
                X[c] = X[c].fillna(mode_val)
            else:
                X = X.drop(columns=[c])
                print(f"Dropped empty categorical column: {c}")

    # 5. Drop any remaining columns that are still empty
    X_clean = X.dropna(axis=1)

    print(f"Shape after imputation: {X_clean.shape}")

    if X_clean.empty:
        print("Error: Dataset empty after preprocessing.")
        return

    # 5. Train
    k = 4
    final_cat_cols = X_clean.select_dtypes(exclude=[np.number]).columns
    cat_indices = [X_clean.columns.get_loc(c) for c in final_cat_cols]

    if len(cat_indices) > 0:
        print(f"Categorical features found: {list(final_cat_cols)}")
        print(f"Training K-Prototypes (k={k})...")
        model = KPrototypes(n_clusters=k, init='Cao', verbose=1, random_state=42)
        clusters = model.fit_predict(X_clean.values, categorical=cat_indices)
        model_type = 'kproto'
    else:
        print(f"No categorical features found. Switching to KMeans (k={k})...")
        model = KMeans(n_clusters=k, init='k-means++', verbose=0, random_state=42)
        clusters = model.fit_predict(X_clean.values)
        model_type = 'kmeans'

    # 6. Save Model
    if model_type == 'kproto':
        mu.export_kprototypes_json(model, X_clean, MODEL_PATH, 'kprototypes_physionet')
    else:
        import json
        centroids = model.cluster_centers_.astype(float).tolist()
        model_data = { "n_clusters": int(k), "cluster_centroids": centroids }
        with open(MODEL_PATH / 'kmeans_physionet.json', 'w') as f:
            json.dump(model_data, f, indent=2)
        meta = {"feature_names": X_clean.columns.tolist()}
        with open(MODEL_PATH / 'kmeans_physionet_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print("KMeans model saved to JSON.")

    # 7. Plots
    X_clean['cluster'] = clusters
    X_clean.to_csv(OUTPUT_PATH / 'clusters.csv')

    rename_map = {
        'LAB1300': 'Leucocites', 'LAB1314': 'Hemoglobin', 'LAB2467': 'Creatinine',
        'LAB2508': 'K', 'LAB2575': 'PCR'
    }

    new_columns = []
    for col in X_clean.columns:
        new_col = col
        for code, name in rename_map.items():
            if col.startswith(code):
                new_col = col.replace(code, name)
                break
        new_columns.append(new_col)
    X_clean.columns = new_columns
 
    notebook_features = [
        'Leucocites_last', 'Leucocites_min', 'Leucocites_max', 'Leucocites_mean',
        'Hemoglobin_last', 'Hemoglobin_min', 'Hemoglobin_max', 'Hemoglobin_mean',
        'Creatinine_last', 'Creatinine_min', 'Creatinine_max', 'Creatinine_mean',
        'K_last', 'K_min', 'K_max', 'K_mean',
        'PCR_last', 'PCR_min', 'PCR_max', 'PCR_mean',
        'FC_last', 'FC_min', 'FC_max', 'FC_mean',
        'PULSIOX_last', 'PULSIOX_min', 'PULSIOX_max', 'PULSIOX_mean',
        'FR_last', 'FR_min', 'FR_max', 'FR_mean',
        'TEMP_last', 'TEMP_min', 'TEMP_max', 'TEMP_mean',
        'P_ART_S_last', 'P_ART_S_min', 'P_ART_S_max', 'P_ART_S_mean',
        'age_last', 'age_min', 'age_max', 'age_mean',
        'cluster'
    ]

    available_features = [c for c in notebook_features if c in X_clean.columns]
    X_final_plot = X_clean[available_features]
    
    print(f"Plotting heatmap with {len(available_features)} aligned features...")

    plt.figure(figsize=(14, 10))
    
    # 1. Calculate Cluster Means
    numeric_plot_cols = X_final_plot.select_dtypes(include=[np.number]).columns
    # Dropping 'cluster' from columns to average, but grouping by it
    cluster_means = X_final_plot.groupby('cluster')[numeric_plot_cols].mean().T
    
    # 2. Calculate Z-scores RELATIVE to other clusters
    # (x - row_mean) / row_std
    z_scores = (cluster_means - cluster_means.mean(axis=1).values[:,None]) / cluster_means.std(axis=1).values[:,None]
    
    # 3. Plot
    sns.heatmap(z_scores, cmap='vlag', center=0, 
                cbar_kws={'label': 'Z-score difference from mean'}, 
                linewidths=0.01)
    
    plt.title('Feature Characteristics by Cluster', fontsize=14)
    plt.xticks(fontsize=10)
    plt.ylabel('Features', fontsize=12)
    plt.xlabel('Cluster', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'cluster_heatmap.png')
    plt.close()
    

if __name__ == "__main__":
    main()