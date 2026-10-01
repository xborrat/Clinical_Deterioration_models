import sys
import pandas as pd
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / 'data' / 'processed'
OUTPUT_PATH = ROOT_PATH / 'data' / 'output'

FILES = {
    'ward_stays':   'ward_stays.csv',
    'demographics': 'demographic.csv',
    'vitals':       'vitals.csv',
    'labs':         'labs.csv',
    'diagnostics':  'diagnostics.csv',
    'lab_dict':     'laboratory_dic.csv',
    'vitals_values_dic': 'vitals_values_dic.csv'
}

# Date columns to parse so dtype shows datetime64 instead of object
DATE_COLS = {
    'ward_stays':   ['start_date', 'end_date', 'hosp_adm_date', 'hosp_disch_date', 'hosp_mortality_date'],
    'vitals':       ['result_date'],
    'labs':         ['extract_date'],
}


def build_report() -> pd.DataFrame:
    rows = []
    for table, fname in FILES.items():
        path = PROCESSED_PATH / fname
        if not path.exists():
            print(f"[WARN] File not found, skipping: {path}", file=sys.stderr)
            continue

        parse_dates = DATE_COLS.get(table, [])
        df = pd.read_csv(path, low_memory=False, parse_dates=parse_dates)
        n_rows = len(df)

        for col in df.columns:
            n_miss = int(df[col].isnull().sum())
            pct_miss = round(n_miss / n_rows * 100, 2) if n_rows > 0 else 0.0
            rows.append({
                'table':      table,
                'column':     col,
                'dtype':      str(df[col].dtype),
                'n_rows':     n_rows,
                'missing_n':  n_miss,
                'missing_pct': pct_miss,
            })

    return pd.DataFrame(rows)


def print_table(report: pd.DataFrame) -> None:
    for table, grp in report.groupby('table', sort=False):
        n_rows = grp['n_rows'].iloc[0]
        print(f"\n{'='*78}")
        print(f"  {table.upper()}  ({n_rows:,} rows, {len(grp)} columns)")
        print(f"{'='*78}")
        print(f"{'Column':<40} {'Type':<20} {'Missing N':>10} {'Missing %':>10}")
        print(f"{'-'*40} {'-'*20} {'-'*10} {'-'*10}")
        for _, r in grp.iterrows():
            flag = ' !' if r['missing_pct'] > 20 else ''
            print(
                f"{r['column']:<40} {r['dtype']:<20} "
                f"{r['missing_n']:>10,} {r['missing_pct']:>9.2f}%{flag}"
            )


def main() -> None:
    report = build_report()

    # --- Console output ---
    print_table(report)

    # --- CSV output ---
    csv_path = OUTPUT_PATH / 'missingness_report.csv'
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    report.to_csv(csv_path, index=False)
    print(f"\n[OK] CSV saved to: {csv_path}")

    # --- Summary: columns with >20% missing ---
    high_miss = report[report['missing_pct'] > 20][['table', 'column', 'dtype', 'missing_pct']]
    if not high_miss.empty:
        print(f"\n{'='*78}")
        print("  COLUMNS WITH >20% MISSINGNESS")
        print(f"{'='*78}")
        print(high_miss.to_string(index=False))


if __name__ == '__main__':
    main()
