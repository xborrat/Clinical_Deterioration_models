import pandas as pd
import numpy as np
from tqdm import tqdm

def filter_valid_stays(ward_stays, min_hours=48):
    """
    Filters for WARD stays longer than min_hours.
    """
    # Ensure dates are datetime
    ward_stays['start_date'] = pd.to_datetime(ward_stays['start_date'])
    ward_stays['end_date'] = pd.to_datetime(ward_stays['end_date'])
    
    # Calculate duration
    ward_stays['duration_h'] = (ward_stays['end_date'] - ward_stays['start_date']).dt.total_seconds() / 3600
    
    # Filter
    mask = (ward_stays['duration_h'] >= min_hours)
    
    # Check if 'care_level_type_ref' exists (Physionet CSVs might skip this if pre-filtered)
    if 'care_level_type_ref' in ward_stays.columns:
        mask = mask & (ward_stays['care_level_type_ref'] == 'WARD')
        
    ward_filter = ward_stays[mask].copy()
    valid_ids = set(ward_filter['stay_id'])
    
    return ward_filter, valid_ids

def build_long_format(ward_stays, vitals, labs, demographics, valid_ids):
    """
    Combines disparate tables into a single Long-Format dataframe.
    Robust to column name variations (Internal vs Physionet).
    """
    # 1. Prepare Static Data (Age, Sex)
    
    # Merge demographics if sex/age missing
    cols_to_check = ['sex', 'age_on_admission', 'age_at_admission', 'age']
    missing_cols = [c for c in cols_to_check if c not in ward_stays.columns]
    
    if missing_cols and demographics is not None:
        # Try to merge on patient_ref
        if 'patient_ref' in ward_stays.columns and 'patient_ref' in demographics.columns:
            # Only bring in columns we don't have
            cols_to_merge = ['patient_ref'] + [c for c in ['sex', 'birth_date'] if c in demographics.columns]
            ws_demo = ward_stays.merge(demographics[cols_to_merge], on='patient_ref', how='left')
        else:
            ws_demo = ward_stays.copy()
    else:
        ws_demo = ward_stays.copy()

    age_col = None
    for col in ['age_on_admission', 'age_at_admission', 'age']:
        if col in ws_demo.columns:
            age_col = col
            break
            
    if age_col:
        df_age = ws_demo[['stay_id', 'start_date', age_col]].rename(
            columns={'start_date': 'time_stamp', age_col: 'value'}
        )
        df_age['concept'] = 'age'
    else:
        # Fallback: Calculate from Birth Date if available
        if 'birth_date' in ws_demo.columns:
            ws_demo['birth_date'] = pd.to_datetime(ws_demo['birth_date'])
            ws_demo['calc_age'] = (ws_demo['start_date'] - ws_demo['birth_date']).dt.days / 365.25
            df_age = ws_demo[['stay_id', 'start_date', 'calc_age']].rename(
                columns={'start_date': 'time_stamp', 'calc_age': 'value'}
            )
            df_age['concept'] = 'age'
        else:
            print("WARNING: Could not find Age column. Skipping Age feature.")
            df_age = pd.DataFrame()

    # handling sex
    if 'sex' in ws_demo.columns:
        df_sex = ws_demo[['stay_id', 'start_date', 'sex']].rename(
            columns={'start_date': 'time_stamp', 'sex': 'value'}
        )
        df_sex['concept'] = 'sex'
    else:
        df_sex = pd.DataFrame()

    # 2. Prepare Vitals
    vitals_filt = vitals[vitals['stay_id'].isin(valid_ids)].copy()
    vitals_filt = vitals_filt.rename(columns={
        'result_date': 'time_stamp', 
        'rc_sap_ref': 'concept', 
        'result_num': 'value'
    })
    # Keep only relevant columns
    vitals_filt = vitals_filt[['stay_id', 'time_stamp', 'concept', 'value']]

    # 3. Prepare Labs
    labs_filt = labs[labs['stay_id'].isin(valid_ids)].copy()
    # Handle column name variations (extract vs extrac)
    date_col = 'extrac_date' if 'extrac_date' in labs.columns else 'extract_date'
    
    labs_filt = labs_filt.rename(columns={
        date_col: 'time_stamp', 
        'lab_sap_ref': 'concept', 
        'result_num': 'value'
    })
    labs_filt = labs_filt[['stay_id', 'time_stamp', 'concept', 'value']]

    # 4. Concatenate
    df_long = pd.concat([df_age, df_sex, vitals_filt, labs_filt], ignore_index=True)
    
    # Ensure timestamp is datetime
    df_long['time_stamp'] = pd.to_datetime(df_long['time_stamp'])
    
    return df_long

def clean_vitals_range(df):
    """
    Applies clinical thresholds to remove artifacts.
    """
    df = df.copy()
    
    # Define clinical limits (Min, Max)
    limits = {
        'FC': (25, 250),
        'P_ART_S': (30, 300),
        'TEMP': (30, 45),
        'PULSIOX': (30, 100),
        'FR': (3, 50),
        'LAB1300': (0, 100000), # Leucocytes
        'LAB1314': (2, 25),     # Hemoglobin
        'age': (18, 110)        # Filter invalid ages
    }
    
    for concept, (min_val, max_val) in limits.items():
        mask = (df['concept'] == concept)
        # Coerce to numeric
        vals = pd.to_numeric(df.loc[mask, 'value'], errors='coerce')
        # Identify valid range
        is_valid = (vals >= min_val) & (vals <= max_val)
        # Set invalid to NaN
        df.loc[mask & ~is_valid, 'value'] = np.nan
        
    return df.dropna(subset=['value'])

def get_labels(ward_stays):
    """Creates binary target variable 'y'."""
    ward_stays = ward_stays.copy()
    # Fill NA for mortality/icu if needed
    ward_stays['to_icu'] = ward_stays['to_icu'].fillna(0)
    ward_stays['hosp_mortality_bin'] = ward_stays['hosp_mortality_bin'].fillna(0)
    
    ward_stays['y'] = ((ward_stays['to_icu'] == 1) | (ward_stays['hosp_mortality_bin'] == 1)).astype(int)
    return ward_stays[['stay_id', 'y']]

def resample_and_pivot(df_long, freq='2h', top_k_features=100, chunk_size=2000):
    """
    Chunked version: Processes patients in batches to prevent MemoryError (OOM).
    """
    print(f"  > Optimizing data structure (Chunked execution)...")

    # 1. GLOBAL FILTER: Keep only top K features to ensure consistency across chunks
    mandatory_cols = ['age', 'sex', 'gender']
    # Count frequency globally first
    concept_counts = df_long['concept'].value_counts()
    top_concepts = concept_counts.head(top_k_features).index.tolist()
    valid_concepts = set(top_concepts) | set(mandatory_cols)
    
    # Filter the main dataframe (removes useless rows)
    df_filtered = df_long[df_long['concept'].isin(valid_concepts)].copy()
    
    # Round time globally
    df_filtered['time_stamp'] = df_filtered['time_stamp'].dt.round(freq)
    
    # Pre-calculate numeric conversion
    df_filtered['value_num'] = pd.to_numeric(df_filtered['value'], errors='coerce')
    
    # 2. CHUNKED PROCESSING
    unique_ids = df_filtered['stay_id'].unique()
    
    if len(unique_ids) == 0: 
        return pd.DataFrame()
    
    # Split patients into chunks (e.g., 2000 patients per chunk)
    num_chunks = max(1, len(unique_ids) // chunk_size)
    id_chunks = np.array_split(unique_ids, num_chunks)
    
    processed_dfs = []
    
    print(f"  > Processing {len(unique_ids)} patients in {len(id_chunks)} chunks...")
    
    # Iterate through chunks instead of the whole dataset
    for ids in tqdm(id_chunks, desc="Pivoting Chunks"):
        # Slice data for this chunk only
        chunk_df = df_filtered[df_filtered['stay_id'].isin(ids)]
        
        # Separate Num/Cat
        mask_num = chunk_df['value_num'].notna()
        
        # Pivot Numeric (Only for this chunk!)
        df_num = chunk_df[mask_num].pivot_table(
            index=['stay_id', 'time_stamp'], 
            columns='concept', 
            values='value_num', 
            aggfunc='last'
        ).astype('float32')
        
        # Pivot Categorical (Only for this chunk!)
        df_cat = chunk_df[~mask_num].pivot_table(
            index=['stay_id', 'time_stamp'], 
            columns='concept', 
            values='value', 
            aggfunc='first'
        )
        
        # Combine
        df_chunk_wide = pd.concat([df_num, df_cat], axis=1)
        
        # Resample (Fill Gaps) for these patients
        chunk_resampled_list = []
        
        # Group by stay_id is now fast because df_chunk_wide is small
        for stay_id, group in df_chunk_wide.groupby('stay_id'):
            # Sort and set index
            g = group.reset_index().set_index('time_stamp').sort_index()
            if 'stay_id' in g.columns: del g['stay_id'] # Cleanup
            
            if len(g) < 2: continue

            # Create time grid
            start = g.index.min()
            end = g.index.max()
            grid = pd.date_range(start=start, end=end, freq=freq)
            
            # Reindex and Fill
            g_res = g.reindex(grid).ffill().bfill()
            g_res['stay_id'] = stay_id
            chunk_resampled_list.append(g_res)
        
        # Add result of this chunk to the main list
        if chunk_resampled_list:
            processed_dfs.append(pd.concat(chunk_resampled_list))
            
    if not processed_dfs:
        return pd.DataFrame()

    print("  > Concatenating chunks...")
    return pd.concat(processed_dfs).rename_axis('time_stamp').reset_index().set_index(['stay_id', 'time_stamp'])

def generate_summary_features(df_ts):
    """
    Flattens time-series into summary statistics for XGBoost.
    """
    grp = df_ts.groupby('stay_id')
    features = pd.DataFrame()
    
    # 1. Last Values
    features = grp.last().add_suffix('_last')
    
    # 2. Aggregations
    numeric_cols = df_ts.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'stay_id']
    
    if numeric_cols:
        agg_funcs = ['mean', 'max', 'min', 'std']
        for func in agg_funcs:
            agg = getattr(grp[numeric_cols], func)().add_suffix(f'_{func}')
            features = pd.merge(features, agg, on='stay_id', how='left')
            
    return features

def get_labels_summary(ward_stays, summary_features):
    """
    Joins the target variable (Y) to the feature set.
    """
    labels = get_labels(ward_stays)
    
    labeled_data = summary_features.merge(
        labels, 
        left_index=True, 
        right_on='stay_id', 
        how='inner'
    ).set_index('stay_id')
    
    return labeled_data

def plot_cluster_heatmap(df_features, cluster_col='cluster', output_path=None, top_n=40):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # 1. Calculate mean values by cluster
    # (Select only numeric columns to avoid errors)
    numeric_df = df_features.select_dtypes(include=[np.number])
    # Ensure cluster_col is preserved if it was numeric, or re-added if it was dropped
    if cluster_col not in numeric_df.columns:
        numeric_df[cluster_col] = df_features[cluster_col]
    cluster_means = numeric_df.groupby(cluster_col).mean().T
    # 2. FILTER: Keep only the most distinct features
    # If we have hundreds of features, the heatmap is unreadable.
    # We calculate variance across clusters to find features that Change the most between groups.
    if len(cluster_means) > top_n:
        variances = cluster_means.var(axis=1)
        top_features = variances.nlargest(top_n).index
        cluster_means = cluster_means.loc[top_features]
        print(f"  > Heatmap reduced to top {top_n} distinctive features (out of {len(variances)})")
    # 3. Calculate Z-scores (Normalize rows)
    z_scores = (cluster_means - cluster_means.mean(axis=1).values[:,None]) / cluster_means.std(axis=1).values[:,None]
    # 4. Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(z_scores, cmap='vlag', center=0, cbar_kws={'label': 'Z-score'}, linewidths=0.5)
    plt.title(f'Top {len(z_scores)} Features Distinguishing Clusters')
    plt.tight_layout()
    if output_path: 
        plt.savefig(output_path)
        print(f"Heatmap saved to {output_path}")
    plt.close()


def prepare_lstm_data(df_ts, max_seq_len=36):
    """
    - Focuses on the first 36 steps (72 hours).
    - Uses 'pre' padding (zeros at the start).
    """
    # Ensure sorted index
    df_ts = df_ts.sort_index()
    
    stay_ids = df_ts.index.get_level_values(0).unique()
    features = df_ts.columns.tolist()
    
    n_samples = len(stay_ids)
    n_features = len(features)
    
    # We follow the notebook's max_len = 36
    X_3d = np.zeros((n_samples, max_seq_len, n_features), dtype=np.float32)
    
    for i, sid in enumerate(tqdm(stay_ids, desc="Building Tensor")):
        patient_data = df_ts.loc[sid].values
        
        # Take up to max_seq_len steps
        seq_len = min(len(patient_data), max_seq_len)
        
        # 'pre' padding (padding=0 by default in np.zeros)
        # We place the data at the END of the fixed-length window
        X_3d[i, -seq_len:, :] = patient_data[:seq_len]
        
    return X_3d, stay_ids.tolist()


def prepare_lstm_data(df_ts, max_seq_len=500):
    """
    Converts Time-Series to 3D Tensor with memory protection.
    Truncates sequences longer than max_seq_len (default 500 steps / ~40 days).
    """
    # Ensure sorted index
    df_ts = df_ts.sort_index()
    
    stay_ids = df_ts.index.get_level_values(0).unique()
    features = df_ts.columns.tolist()
    
    n_samples = len(stay_ids)
    n_features = len(features)
    
    # Check actual max length, but clip it to our safety limit
    actual_max = df_ts.groupby(level=0).size().max()
    final_len = min(actual_max, max_seq_len)
    
    print(f"  > Creating 3D Tensor: {n_samples} patients, Sequence Len: {final_len}, Features: {n_features}")
    
    # Allocate smaller matrix (float32 saves memory)
    X_3d = np.zeros((n_samples, final_len, n_features), dtype=np.float32)
    
    for i, sid in enumerate(tqdm(stay_ids, desc="Building Tensor")):
        patient_data = df_ts.loc[sid].values
        
        # If patient stay is longer than limit, take the LAST 'final_len' steps
        # (Recent data is usually more important for deterioration/mortality)
        if len(patient_data) > final_len:
            patient_data = patient_data[-final_len:]
            
        length = len(patient_data)
        
        # Post-padding: Data at start, zeros at end (Masking layer handles the zeros)
        X_3d[i, :length, :] = patient_data
        
    return X_3d, stay_ids, final_len