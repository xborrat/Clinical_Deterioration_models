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
# Add a path for processed data
PROCESSED_PATH = ROOT_PATH / 'data' / 'processed'

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class PhysioNetValidator:
    
    REQUIRED_FILES = {
        'ward_stays': 'ward_stays.csv',
        'demographics': 'demographic.csv',
        'vitals': 'vitals.csv',
        'labs': 'labs.csv',
        'dictionary': 'laboratory_dic.csv'
    }

    EXPECTED_COLUMNS = {
        'ward_stays': ['stay_id', 'patient_ref', 'start_date', 'end_date', 'age_at_admission', 
                       'to_icu', 'hosp_mortality_bin', 'ou_med_ref'],
        'demographics': ['patient_ref', 'sex', 'natio_ref'],
        'vitals': ['stay_id', 'rc_sap_ref', 'result_num', 'result_txt', 'result_date'],
        'labs': ['stay_id', 'lab_sap_ref', 'result_num', 'extract_date']
    }

    def __init__(self, data_path: Path, output_path: Path):
        self.data_path = data_path
        self.output_path = output_path
        self.data: Dict[str, pd.DataFrame] = {}
        self._validate_path()

    def _validate_path(self):
        if not self.data_path.exists():
            logger.error(f"❌ Data directory not found: {self.data_path}")
            sys.exit(1)

    def _log_header(self, title: str):
        logger.info("\n" + "=" * 80)
        logger.info(title)
        logger.info("=" * 80 + "\n")

    def _log_section(self, title: str):
        logger.info(f"\n--- {title} ---")

    def load_data(self) -> None:
        self._log_header("1. LOADING PHYSIONET DATASET")
        logger.info(f"Root Path: {ROOT_PATH}")
        logger.info(f"Data Path: {self.data_path}")
        
        # Check file existence
        missing_files = []
        for key, filename in self.REQUIRED_FILES.items():
            filepath = self.data_path / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"✓ {filename:25} ({size_mb:.1f} MB)")
            else:
                logger.error(f"✗ {filename:25} MISSING")
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

        logger.info("\nLoading tables into memory...")

        try:
            self.data['ward_stays'] = pd.read_csv(self.data_path / self.REQUIRED_FILES['ward_stays'], low_memory=False)
            self.data['demographics'] = pd.read_csv(self.data_path / self.REQUIRED_FILES['demographics'])
            self.data['vitals'] = pd.read_csv(self.data_path / self.REQUIRED_FILES['vitals'], low_memory=False)
            self.data['labs'] = pd.read_csv(self.data_path / self.REQUIRED_FILES['labs'], low_memory=False)
            self.data['dictionary'] = pd.read_csv(self.data_path / self.REQUIRED_FILES['dictionary'])
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            sys.exit(1)

        logger.info("Converting data types...")
        self._convert_types()
        
        logger.info("\nRecords loaded:")
        for name, df in self.data.items():
            logger.info(f"  {name:15}: {len(df):,}")

    def _convert_types(self):
        ws = self.data['ward_stays']
        vitals = self.data['vitals']
        labs = self.data['labs']

        ws['start_date'] = pd.to_datetime(ws['start_date'], format='mixed', errors='coerce')
        ws['end_date'] = pd.to_datetime(ws['end_date'], format='mixed', errors='coerce')
        vitals['result_date'] = pd.to_datetime(vitals['result_date'], format='mixed', errors='coerce')
        labs['extract_date'] = pd.to_datetime(labs['extract_date'], format='mixed', errors='coerce')

        ws['age_at_admission'] = pd.to_numeric(ws['age_at_admission'], errors='coerce')
        ws['to_icu'] = pd.to_numeric(ws['to_icu'], errors='coerce')
        ws['hosp_mortality_bin'] = pd.to_numeric(ws['hosp_mortality_bin'], errors='coerce')
        vitals['result_num'] = pd.to_numeric(vitals['result_num'], errors='coerce')
        labs['result_num'] = pd.to_numeric(labs['result_num'], errors='coerce')

    def validate_structure(self):
        self._log_header("2. DATA STRUCTURE VALIDATION")
        all_valid = True
        for name, expected_cols in self.EXPECTED_COLUMNS.items():
            if name not in self.data: continue
            df = self.data[name]
            missing = [c for c in expected_cols if c not in df.columns]
            logger.info(f"{name}:")
            if missing:
                logger.error(f"  ❌ Missing columns: {missing}")
                all_valid = False
            else:
                logger.info(f"  ✓ All expected columns present")
        if all_valid:
            logger.info("\n✓ Structure validation passed.")

    def check_integrity(self):
        self._log_header("3. REFERENTIAL INTEGRITY CHECKS")
        ws = self.data['ward_stays']
        demo = self.data['demographics']
        vitals = self.data['vitals']
        labs = self.data['labs']

        self._log_section("Unique Identifier Counts")
        logger.info(f"Unique stays (ward_stays): {ws['stay_id'].nunique():,}")
        logger.info(f"Unique patients (ward_stays): {ws['patient_ref'].nunique():,}")

        self._log_section("Missing Reference Analysis")
        missing_demo = set(ws['patient_ref']) - set(demo['patient_ref'])
        if missing_demo:
            logger.error(f"❌ {len(missing_demo)} patients missing demographic records")
        else:
            logger.info("✓ All patients have demographic records")

        valid_stays = set(ws['stay_id'])
        orphan_vitals = set(vitals['stay_id']) - valid_stays
        if orphan_vitals:
            logger.error(f"❌ ERROR: Orphaned vital records for {len(orphan_vitals)} stays")
            count = len(vitals[vitals['stay_id'].isin(orphan_vitals)])
            logger.error(f"  -> Total orphaned vital records: {count:,}")
        else:
            logger.info("✓ All vital records link to valid stays")

        orphan_labs = set(labs['stay_id']) - valid_stays
        if orphan_labs:
            logger.error(f"❌ ERROR: Orphaned lab records for {len(orphan_labs)} stays")
            count = len(labs[labs['stay_id'].isin(orphan_labs)])
            logger.error(f"  -> Total orphaned lab records: {count:,}")
        else:
            logger.info("✓ All lab records link to valid stays")

    def clean_orphans(self):
        self._log_header("4. DATA CLEANING (ORPHAN REMOVAL)")
        ws = self.data['ward_stays']
        vitals = self.data['vitals']
        labs = self.data['labs']
        
        valid_stays = set(ws['stay_id'])
        vitals_before = len(vitals)
        labs_before = len(labs)
        
        self.data['vitals'] = vitals[vitals['stay_id'].isin(valid_stays)].reset_index(drop=True)
        self.data['labs'] = labs[labs['stay_id'].isin(valid_stays)].reset_index(drop=True)
        
        vitals_removed = vitals_before - len(self.data['vitals'])
        labs_removed = labs_before - len(self.data['labs'])
        
        if vitals_removed > 0 or labs_removed > 0:
            logger.info("Cleaning Complete:")
            logger.info(f"  - Removed {vitals_removed:,} orphaned vital records")
            logger.info(f"  - Removed {labs_removed:,} orphaned lab records")
        else:
            logger.info("✓ No orphans found. No cleaning needed.")

    def check_temporal_plausibility(self):
        self._log_header("5. TEMPORAL PLAUSIBILITY")
        ws = self.data['ward_stays']
        vitals = self.data['vitals']
        
        ws['duration'] = (ws['end_date'] - ws['start_date']).dt.total_seconds() / (24 * 3600)
        neg_stays = ws[ws['duration'] < 0]
        if not neg_stays.empty:
            logger.error(f"❌ ERROR: {len(neg_stays)} stays with negative duration")
        else:
            logger.info("✓ No negative stay durations")

        all_dates = pd.concat([ws['start_date'], ws['end_date'], vitals['result_date']])
        future_dates = all_dates[all_dates > pd.Timestamp.now()]
        if not future_dates.empty:
            logger.error(f"❌ ERROR: Found {len(future_dates):,} dates in the future. Expected behavior due to anonymization.")
        else:
            logger.info("✓ No future dates found")

    def check_age_verification(self):
        self._log_header("6. AGE VERIFICATION")
        ws = self.data['ward_stays']
        if 'age_at_admission' in ws.columns:
            under_18 = ws[ws['age_at_admission'] < 18]
            if not under_18.empty:
                logger.warning(f"⚠️ WARNING: Found {len(under_18)} records for patients under 18")
            else:
                logger.info("✓ No patients under 18 found")
            logger.info(f"Age Mean: {ws['age_at_admission'].mean():.1f} | Max: {ws['age_at_admission'].max()}")

    def check_statistics(self):
        self._log_header("7. STATISTICAL VALIDATION")
        ws = self.data['ward_stays']
        total = len(ws)
        icu = ws['to_icu'].sum()
        mortality = ws['hosp_mortality_bin'].sum()
        logger.info(f"Total Stays: {total:,}")
        logger.info(f"ICU Transfers: {icu:,} ({icu/total*100:.1f}%)")
        logger.info(f"Mortality: {mortality:,} ({mortality/total*100:.1f}%)")

    # --- NEW METHOD: SAVE DATA ---
    def save_clean_data(self):
        """Saves the cleaned dataframes to the PROCESSED directory."""
        self._log_header("8. SAVING DATA")
        
        # Create output directory if it doesn't exist
        if not self.output_path.exists():
            logger.info(f"Creating directory: {self.output_path}")
            self.output_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Saving cleaned files to: {self.output_path}")
        
        try:
            for name, df in self.data.items():
                filename = self.REQUIRED_FILES[name]
                save_path = self.output_path / filename
                
                logger.info(f"  Saving {filename}...")
                df.to_csv(save_path, index=False)
                
            logger.info("\n✓ All files saved successfully.")
        except Exception as e:
            logger.error(f"❌ Error saving files: {e}")

    def run_full_pipeline(self):
        self.load_data()
        self.validate_structure()
        self.check_integrity()
        self.clean_orphans()
        self.check_temporal_plausibility()
        self.check_age_verification()
        self.check_statistics()
        self.save_clean_data()  # <--- Added saving step
        
        logger.info("\n" + "="*80)
        logger.info("✅ QUALITY CHECK PIPELINE COMPLETE")
        logger.info("="*80)

def main():
    parser = argparse.ArgumentParser(description="PhysioNet Data Quality Checker")
    parser.add_argument('--data_path', type=Path, default=DATA_PATH)
    # Add argument for output path
    parser.add_argument('--output_path', type=Path, default=PROCESSED_PATH)
    
    args = parser.parse_args()
    
    validator = PhysioNetValidator(data_path=args.data_path, output_path=args.output_path)
    validator.run_full_pipeline()

if __name__ == "__main__":
    main()