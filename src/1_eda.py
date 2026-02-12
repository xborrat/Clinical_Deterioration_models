import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from pathlib import Path
from tqdm import tqdm

# Add src to path
ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from utils import clinical_preproc as pp

# Config
DATA_PATH = ROOT_PATH / 'data' / 'processed'
OUTPUT_PATH = ROOT_PATH / 'data' / 'output'
os.makedirs(OUTPUT_PATH, exist_ok=True)

def main():
    print("--- 1. EDA Analysis (Physionet/Local) ---")

    # 1. Load Data
    print("Loading CSV files...")
    # Using low_memory=False to avoid mixed type warnings on large files
    ward_stays = pd.read_csv(DATA_PATH / 'ward_stays.csv', low_memory=False)
    vitals = pd.read_csv(DATA_PATH / 'vitals.csv', low_memory=False)
    labs = pd.read_csv(DATA_PATH / 'labs.csv', low_memory=False)
    demographics = pd.read_csv(DATA_PATH / 'demographic.csv', low_memory=False) # Note singular/plural check

    # 2. Filter Stays (Shared Logic)
    print("Filtering Ward Stays (>48h)...")
    ws_filtered, valid_ids = pp.filter_valid_stays(ward_stays, min_hours=48)
    print(f"  > Valid Stays: {len(ws_filtered)} (Original: {len(ward_stays)})")

    # 3. Build Long Format (Heavy processing)
    print("Building Long-Format Dataframe...")
    df = pp.build_long_format(ws_filtered, vitals, labs, demographics, valid_ids)
    
    # 4. Cleaning
    print("Cleaning values (Clinical Thresholds)...")
    df_clean = pp.clean_vitals_range(df)
    
    # 5. Merge Labels
    print("Merging outcome labels...")
    labels = pp.get_labels(ws_filtered)
    df_final = df_clean.merge(labels, on='stay_id', how='left')

    # --- PLOTTING ---
    print("\nGenerating Plots...")

    # Plot A: Missing Values Matrix (Sampled for speed)
    print("  > Plotting Missing Values Matrix...")
    # Pivot a sample to visualize missingness structure
    sample_ids = pd.Series(list(valid_ids)).sample(min(100, len(valid_ids)), random_state=42)
    df_sample = df_final[df_final['stay_id'].isin(sample_ids)].pivot_table(
        index=['stay_id', 'time_stamp'], columns='concept', values='value', aggfunc='first'
    )
    plt.figure(figsize=(12, 6))
    msno.matrix(df_sample, sparkline=False)
    plt.title("Missing Values Pattern (Sample)")
    plt.savefig(OUTPUT_PATH / 'missing_matrix.png', bbox_inches='tight')
    plt.close()

    # Plot B: Distributions by Class (Iterating through key concepts)
    key_concepts = ['FC', 'TEMP', 'P_ART_S', 'age']
    print(f"  > Plotting Distributions for: {key_concepts}")
    
    for concept in tqdm(key_concepts, desc="Plots"):
        subset = df_final[df_final['concept'] == concept]
        if subset.empty: continue
        
        plt.figure(figsize=(10, 6))
        # Ensure numeric
        subset['value'] = pd.to_numeric(subset['value'], errors='coerce')
        sns.boxplot(data=subset, x='y', y='value', hue='y', palette="Set2")
        plt.title(f"Distribution of {concept} by Deterioration (0=No, 1=Yes)")
        plt.xlabel("Deterioration")
        plt.savefig(OUTPUT_PATH / f'dist_{concept}.png')
        plt.close()

    print(f"\nEDA Complete. Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()