# Clinical Deterioration Prediction Models

An early warning system using machine learning to predict clinical deterioration in hospitalized patients. This project processes electronic health records (EHR) to identify risk profiles and predict ICU transfer or in-hospital mortality before conventional thresholds are reached.

##  Project Overview

Current early warning systems (like NEWS-2) are often reactive. This project utilizes machine learning to:
1.  **Analyze temporal dynamics** of vital signs and laboratory results.
2.  **Cluster patients** into specific phenotypes using K-Prototypes (or K-Means when no categorical features remain).
3.  **Predict deterioration** (ICU transfer or Mortality) using XGBoost and LSTM models.

##  Repository Structure

The project is structured to run sequentially as a data pipeline:

```text
├── data/
│   ├── raw/                 # Input CSV files (PhysioNet/Local)
│   ├── processed/           # Cleaned data (Output of step 0)
│   └── output/              # Generated plots, visualizations, and clusters
├── models/                  # Saved model files (.json, .h5)
├── src/
│   ├── 0_quality_check.py   # Step 0: Validation & Cleaning
│   ├── 1_eda.py             # Step 1: Exploratory Data Analysis
│   ├── 2_kprototypes.py     # Step 2: Clustering (K-Prototypes/K-Means)
│   ├── 3_model_training.py  # Step 3: XGBoost & LSTM Training
│   └── missingness_report.py  # Optional: missing values per column of the processed data
├── utils/
│   ├── clinical_preproc.py  # Preprocessing logic (Long-format, resampling)
│   └── model_saving.py      # JSON export utilities
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Data Dictionary

Data can be found in Physionet.

The pipeline expects the following CSV files in `data/raw/`:

| File                 | Required |
|----------------------|----------|
| `ward_stays.csv`     | Yes      |
| `demographic.csv`    | Yes      |
| `vitals.csv`         | Yes      |
| `labs.csv`           | Yes      |
| `laboratory_dic.csv` | Yes      |
| `diagnostics.csv`    | Optional (diagnosis checks are skipped if missing) |

The most relevant columns per data file are listed below:
### 1. Ward Stays (`ward_stays.csv`)

| Column               | Description                                           |
|----------------------|-------------------------------------------------------|
| stay_id              | Unique identifier for the ward stay                  |
| patient_ref          | Unique patient identifier                            |
| episode_ref          | Hospital episode identifier                          |
| start_date           | Start of the ward stay                               |
| end_date             | End of the ward stay                                 |
| care_level_type_ref  | Care level (`WARD` or `ICU`); only `WARD` stays are used |
| age_at_admission     | Age at admission (exact age, or a 5-year range such as `40-44` when generalized for de-identification) |
| to_icu               | Target: 1 if transferred to ICU, 0 otherwise         |
| hosp_mortality_bin   | Target: 1 if deceased in hospital, 0 otherwise       |
| ou_med_ref           | Operational Unit / Ward reference                    |

### 2. Demographics (`demographic.csv`)

| Column      | Description                         |
|-------------|-------------------------------------|
| patient_ref | Unique patient identifier           |
| sex         | Sex code (1, 2 or 3)                |
| natio_ref   | Nationality code                   |

### 3. Vitals (`vitals.csv`)

| Column      | Description                         |
|-------------|-------------------------------------|
| stay_id     | Foreign key to Ward Stays          |
| rc_sap_ref  | Vital sign code (Concept)          |
| result_num  | Numeric value                      |
| result_date | Measurement timestamp              |

**Vital Sign Codes**

- `FC` — Heart Rate (bpm)  
- `TEMP` — Temperature (°C)  
- `P_ART_S` — Systolic Blood Pressure (mmHg)  
- `PULSIOX` — Oxygen Saturation (SpO2 %)  
- `FR` — Respiratory Rate (rpm)

### 4. Labs (`labs.csv`)

| Column        | Description                         |
|---------------|-------------------------------------|
| stay_id       | Foreign key to Ward Stays          |
| lab_sap_ref   | Laboratory test code (Concept)     |
| result_num    | Numeric value                      |
| extract_date  | Extraction timestamp               |

**Lab Codes**

- `LAB1300` — Leukocytes  
- `LAB1314` — Hemoglobin  
- `LAB2467` — Creatinine  
- `LAB2508` — Potassium (K)  
- `LAB2575` — PCR (C-Reactive Protein)


---


## Usage Guide

Ensure your downloaded files from Physionet are placed in `data/raw/`.



### 1. Installation
Create a virtual environment and install the dependencies to avoid versioning or dependency conflicts.

```bash
conda create -n clinical_env python=3.9
conda activate clinical_env
pip install -r requirements.txt
```

### 2. Execution Pipeline

#### Step 0 — Quality Check & Cleaning
Validates referential integrity, removes orphaned records, reports data coverage and cohort statistics, and saves cleaned data to `data/processed/`. The report is written to `data/output/quality_check_report.txt`.

```bash
python src/0_quality_check.py
```

#### Step 1 — Exploratory Data Analysis (EDA)
Generates distribution plots and missing value matrices to `data/output/`.

```bash
python src/1_eda.py
```

#### Step 2 — Clustering (Unsupervised)
Performs K-Prototypes clustering (k=4) to identify patient phenotypes, falling back to K-Means when no categorical features remain after preprocessing. Resamples data to 2-hour windows.

```bash
python src/2_kprototypes.py
```

#### Step 3 — Predictive Modeling (Supervised)

Trains and evaluates deterioration prediction models.

1. **XGBoost:** Uses summary statistics (last, mean, std, min, max).
2. **LSTM:** Uses 3D temporal sequences (Time-series).
3. Outputs: ROC curves, Confusion Matrices, and SHAP plots.

```bash
python src/3_model_training.py
```
---

## Methodology

### Preprocessing

- Only general ward stays (`care_level_type_ref = WARD`) are used
- Excluding stays shorter than 48 hours
- Removing physiologically implausible values (set to missing and dropped):

| Variable | Valid range |
|----------|-------------|
| `FC` (heart rate) | 25–250 |
| `P_ART_S` (systolic BP) | 30–300 |
| `TEMP` (temperature) | 30–45 |
| `PULSIOX` (SpO2) | 30–100 |
| `FR` (respiratory rate) | 3–50 |
| `LAB1300` (leukocytes) | 0–100,000 |
| `LAB1314` (hemoglobin) | 2–25 |
| `age` | 18–110 |

- Keeping the 100 most frequent clinical concepts (plus age and sex)
- Resampling to 2-hour intervals using forward-fill and backward-fill

### Models

- **K-Prototypes:** Handles mixed data (numerical vitals + categorical demographics) to group patients into risk clusters. K-Means is used instead when all features are numeric.
- **XGBoost:** A gradient boosting classifier trained on aggregated features for interpretability.
- **LSTM:** A deep learning model that processes the sequential history of vitals and laboratory results to capture trends over time.

Both supervised models use a stratified 80/20 train/test split (`random_state=42`). The LSTM training set is balanced by oversampling the positive class.


### Feature Engineering

The project employs two distinct feature engineering strategies depending on the model architecture:

#### 1. Summary Statistics (XGBoost & K-Prototypes)
Temporal data is aggregated into fixed-size vectors capturing the statistical behavior of physiological variables throughout the stay.

* **Metrics per variable:**

    * `_LAST`: Most recent value observed.
    
    * `_MEAN`: Average value.
    
    * `_STD`: Standard deviation (variability).
    
    * `_MIN`: Minimum value registered.
    
    * `_MAX`: Maximum value registered.
    
* **Input**: 5 aggregated features per clinical variable.

#### 2. Time-Series Sequences (LSTM)
Deep learning models utilize the raw temporal sequences to capture dynamic patterns.

* **Window**: Focuses on the last **72 hours** (36 steps of 2 hours) of hospitalization.

* **Handling**: Longer sequences are truncated to their last 36 steps; shorter ones are zero-padded at the end.

* **Input**: 3D Tensor `(Samples, TimeSteps, Features)`.


---

## Contributing

Contributions are welcome. Please:

1. Fork the project
2. Create a branch for your feature (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

---

## License

MIT License — see `LICENSE` for details.




