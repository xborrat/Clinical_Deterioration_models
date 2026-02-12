import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set
import warnings

# --- PATH CONFIGURATION ---
ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))
DATA_PATH = ROOT_PATH / 'data' / 'raw'
PROCESSED_PATH = ROOT_PATH / 'data' / 'processed'
OUTPUT_PATH = ROOT_PATH / 'data' / 'output'

# Ensure output directory exists
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --- LOGGING CONFIGURATION ---
# We set up the logger to write to both console (stdout) and a file.
logger = logging.getLogger("QualityCheck")
logger.setLevel(logging.INFO)

# Clear existing handlers if any (prevents duplicate logs in notebooks/re-runs)
if logger.handlers:
    logger.handlers = []

# 1. Console Handler
c_handler = logging.StreamHandler(sys.stdout)
c_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

# 2. File Handler (Overwrites previous log)
f_handler = logging.FileHandler(OUTPUT_PATH / 'quality_check_report.txt', mode='w')
f_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)

warnings.filterwarnings('ignore')

class PhysioNetValidator:
    
    REQUIRED_FILES = {
        'ward_stays': 'ward_stays.csv',
        'demographics': 'demographic.csv',
        'vitals': 'vitals.csv',
        'labs': 'labs.csv',
        'diagnostics': 'diagnostics.csv', 
        'dictionary': 'laboratory_dic.csv'
    }

    def __init__(self, data_path: Path, output_path: Path):
        self.data_path = data_path
        self.output_path = output_path
        self.data: Dict[str, pd.DataFrame] = {}
        self.orphans_to_remove: Dict[str, tuple] = {}
        self._validate_input_path()

    def _validate_input_path(self):
        if not self.data_path.exists():
            logger.error(f"FATAL: Data directory not found: {self.data_path}")
            sys.exit(1)

    def _print_separator(self, title: str):
        # Using logger.info ensures this gets written to the text file too
        logger.info(f"\n{'='*80}\n{title.upper()}\n{'='*80}")

    def load_data(self) -> None:
        self._print_separator("1. Data Loading & Schema Validation")
        
        for key, filename in self.REQUIRED_FILES.items():
            filepath = self.data_path / filename
            if not filepath.exists():
                if key == 'diagnostics':
                    logger.warning(f"Optional file missing: {filename} (Skipping diagnosis checks)")
                    self.data[key] = pd.DataFrame()
                    continue
                else:
                    logger.error(f"Missing required file: {filename}")
                    sys.exit(1)
            
            # Load Data
            try:
                df = pd.read_csv(filepath, low_memory=False)
                self.data[key] = df
                logger.info(f"Loaded {key:15} | Rows: {len(df):<10,} | Cols: {len(df.columns)}")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                sys.exit(1)

        self._preprocess_types()

    def _preprocess_types(self):
        ws = self.data['ward_stays']
        if 'episode_ref' in ws.columns:
            ws['episode_ref'] = pd.to_numeric(ws['episode_ref'], errors='coerce')
        
        if not self.data['diagnostics'].empty and 'episode_ref' in self.data['diagnostics'].columns:
             self.data['diagnostics']['episode_ref'] = pd.to_numeric(self.data['diagnostics']['episode_ref'], errors='coerce')

        date_cols = ['start_date', 'end_date']
        for c in date_cols: 
            if c in ws.columns: ws[c] = pd.to_datetime(ws[c], errors='coerce')

        ws['age_at_admission'] = pd.to_numeric(ws['age_at_admission'], errors='coerce')
        if 'to_icu' in ws.columns: ws['to_icu'] = pd.to_numeric(ws['to_icu'], errors='coerce')
        if 'hosp_mortality_bin' in ws.columns: ws['hosp_mortality_bin'] = pd.to_numeric(ws['hosp_mortality_bin'], errors='coerce')

    def check_integrity(self):
        self._print_separator("2. Referential Integrity Check (Orphans)")

        ws = self.data['ward_stays']
        demo = self.data['demographics']
        vitals = self.data['vitals']
        labs = self.data['labs']
        diag = self.data['diagnostics']

        self._check_orphan_link(ws, demo, 'patient_ref', 'WardStays', 'Demographics')
        self._check_orphan_link(vitals, ws, 'stay_id', 'Vitals', 'WardStays')
        self._check_orphan_link(labs, ws, 'stay_id', 'Labs', 'WardStays')

        if not diag.empty and 'episode_ref' in ws.columns:
            self._check_orphan_link(diag, ws, 'episode_ref', 'Diagnostics', 'WardStays(Episodes)')
        elif not diag.empty:
            logger.warning("SKIP: 'episode_ref' missing in WardStays. Cannot validate Diagnostics.")

    def _check_orphan_link(self, child_df, parent_df, key, child_name, parent_name):
        if key not in child_df.columns or key not in parent_df.columns:
            logger.warning(f"SKIP: Key '{key}' missing in {child_name} or {parent_name}")
            return

        child_ids = set(child_df[key].dropna())
        parent_ids = set(parent_df[key].dropna())
        
        orphans = child_ids - parent_ids
        n_orphans = len(orphans)
        
        if n_orphans == 0:
            logger.info(f"[PASS] {child_name:15} -> {parent_name:20} : Integrity Intact")
        else:
            n_rows = child_df[child_df[key].isin(orphans)].shape[0]
            logger.error(f"[FAIL] {child_name:15} -> {parent_name:20} : {n_orphans:,} orphaned IDs ({n_rows:,} rows)")
            self.orphans_to_remove[child_name] = (key, orphans)

    def check_coverage(self):
        self._print_separator("3. Data Coverage Analysis (Inverse Integrity)")
        
        ws = self.data['ward_stays']
        demo = self.data['demographics']
        vitals = self.data['vitals']
        labs = self.data['labs']
        diag = self.data['diagnostics']

        self._calc_coverage(demo, ws, 'patient_ref', 'Demographics', 'WardStays')
        self._calc_coverage(ws, vitals, 'stay_id', 'WardStays', 'Vitals')
        self._calc_coverage(ws, labs, 'stay_id', 'WardStays', 'Labs')

        if not diag.empty and 'episode_ref' in ws.columns:
            self._calc_coverage(ws, diag, 'episode_ref', 'WardStays(Epi)', 'Diagnostics')

    def _calc_coverage(self, source_df, target_df, key, source_name, target_name):
        if key not in source_df.columns or key not in target_df.columns:
            return

        total_source = source_df[key].nunique()
        matched_source = source_df[source_df[key].isin(target_df[key])][key].nunique()
        pct = (matched_source / total_source) * 100 if total_source > 0 else 0

        logger.info(f"{source_name:15} w/ {target_name:<15}: {matched_source:>7,} / {total_source:>7,} ({pct:6.2f}%)")

    def clean_orphans(self):
        if not self.orphans_to_remove: 
            return

        self._print_separator("4. Data Cleaning (Orphan Removal)")
        
        for table_name, (key, orphan_ids) in self.orphans_to_remove.items():
            if table_name in self.data:
                df = self.data[table_name]
                initial_len = len(df)
                self.data[table_name] = df[~df[key].isin(orphan_ids)]
                removed = initial_len - len(self.data[table_name])
                logger.info(f"Cleaned {table_name:15}: Removed {removed:>7,} rows (Orphaned {key})")

    def generate_cohort_summary(self):
        self._print_separator("5. Final Cohort Statistics")
        
        ws = self.data['ward_stays']
        total_screened = len(ws)
        
        # Filter Logic
        if 'care_level_type_ref' in ws.columns:
            cohort = ws[ws['care_level_type_ref'].str.upper() == 'WARD']
        else:
            cohort = ws

        # Population
        if 'age_at_admission' in cohort.columns:
            adults = cohort[cohort['age_at_admission'] >= 18]
        else:
            adults = cohort
            
        unique_pts = adults['patient_ref'].nunique()
        
        # Volume
        final_ids = set(adults['stay_id'])
        n_vitals = len(self.data['vitals'][self.data['vitals']['stay_id'].isin(final_ids)])
        n_labs = len(self.data['labs'][self.data['labs']['stay_id'].isin(final_ids)])
        
        # Outcome
        if 'to_icu' in adults.columns and 'hosp_mortality_bin' in adults.columns:
            outcomes = ((adults['to_icu'] == 1) | (adults['hosp_mortality_bin'] == 1)).sum()
            prev = (outcomes / len(adults)) * 100
        else:
            outcomes = 0
            prev = 0

        # Use logger.info to ensure writing to file
        logger.info(f"{'Total Screened Stays':<25}: {total_screened:,}")
        logger.info(f"{'Final Cohort Size':<25}: {len(adults):,} (General Ward / Adults)")
        logger.info(f"{'Patient Population':<25}: {unique_pts:,} unique patients")
        logger.info(f"{'Data Volume':<25}: {n_vitals/1e6:.2f}M vitals | {n_labs/1e6:.2f}M labs")
        
        if not self.data['diagnostics'].empty:
            if 'episode_ref' in adults.columns:
                final_eps = set(adults['episode_ref'])
                n_diag = len(self.data['diagnostics'][self.data['diagnostics']['episode_ref'].isin(final_eps)])
            else:
                n_diag = len(self.data['diagnostics'])
            logger.info(f"{'Diagnostic Coverage':<25}: {n_diag:,} ICD-10 entries")
        
        logger.info(f"{'Primary Outcome':<25}: {outcomes:,} events ({prev:.2f}%) [ICU/Death]")

    def save_data(self):
        self._print_separator("6. Saving Processed Data")
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
            
        for name, df in self.data.items():
            if df.empty: continue
            
            path = self.output_path / self.REQUIRED_FILES[name]
            try:
                df.to_csv(path, index=False)
                logger.info(f"Saved: {path.name:<20} ({len(df):>8,} rows)")
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")

    def run(self):
        self.load_data()
        self.check_integrity()
        self.clean_orphans()
        self.check_coverage()
        self.generate_cohort_summary()
        self.save_data()
        self._print_separator("Pipeline Complete")
        
        print(f"\n[INFO] Report saved to: {OUTPUT_PATH / 'quality_check_report.txt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=DATA_PATH)
    parser.add_argument('--output', type=Path, default=PROCESSED_PATH)
    args = parser.parse_args()

    validator = PhysioNetValidator(args.input, args.output)
    validator.run()