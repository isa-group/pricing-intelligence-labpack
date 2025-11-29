"""statistical_evaluation.py

Carga el DataFrame generado desde `data/evaluation_report_generated.json`, analiza la
distribución de cada variable numérica y devuelve p-values para pruebas de
normalidad, skewness y kurtosis. También decide si la media es representativa
en base a los tests y criterios simples (normalidad o bajo sesgo y pocos outliers).

Uso:
    python statistical_evaluation.py --input data/evaluation_report_generated.json --out data/statistical_summary.csv

Requisitos:
    pip install pandas scipy numpy
"""
from pathlib import Path
import argparse
import json
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def generate_dataframe_questions_from_file(path: Path) -> pd.DataFrame:
    """Lee `evaluation_report_generated.json` y retorna un DataFrame normalizado.

    Esta función replica la lógica usada en el notebook `visualization.ipynb`.
    """
    with path.open('r', encoding='utf-8') as f:
        j = json.load(f)

    df = pd.json_normalize(j.get('details', []))
    df = df.rename(columns={'question': 'Question'})
    df.insert(0, 'ID', [f'Q{i}' for i in range(1, len(df) + 1)])

    # Renombrado de campos anidados
    rename_map = {}
    for col in df.columns:
        if col.startswith('structure.'):
            rename_map[col] = 'Structure_' + col.split('.', 1)[1]
        if col.startswith('content.'):
            rename_map[col] = 'Content_' + col.split('.', 1)[1]
    if rename_map:
        df = df.rename(columns=rename_map)

    df = df.drop(columns=['index', 'template'], errors='ignore')
    return df


def analyze_column(arr: np.ndarray, alpha: float = 0.05) -> dict:
    """Analiza una columna numérica y devuelve estadísticas y p-values.

    Devuelve un dict con: n, mean, median, std, skew, skew_p, kurtosis, kurt_p,
    normal_test (name), normal_p, outlier_prop, mean_representative, notes.
    """
    res = {
        'n': int(len(arr)),
        'mean': float(np.nan),
        'median': float(np.nan),
        'std': float(np.nan),
        'skew': float(np.nan),
        'skew_p': float('nan'),
        'kurtosis': float(np.nan),
        'kurt_p': float('nan'),
        'normal_test': None,
        'normal_p': float('nan'),
        'outlier_prop': float('nan'),
        'mean_representative': False,
        'notes': '',
    }

    x = np.asarray(arr).astype(float)
    x = x[~np.isnan(x)]
    n = len(x)
    res['n'] = int(n)
    if n == 0:
        res['notes'] = 'empty'
        return res

    res['mean'] = float(np.mean(x))
    res['median'] = float(np.median(x))
    res['std'] = float(np.std(x, ddof=1)) if n > 1 else 0.0

    # Skewness y kurtosis
    try:
        sk = float(stats.skew(x, bias=False))
        res['skew'] = sk
    except Exception as e:
        res['notes'] += f'skew_err:{e};'

    try:
        kt = float(stats.kurtosis(x, fisher=True, bias=False))
        res['kurtosis'] = kt
    except Exception as e:
        res['notes'] += f'kurt_err:{e};'

    # Outliers por regla IQR
    try:
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = np.sum((x < lower) | (x > upper))
        res['outlier_prop'] = float(outliers) / float(n)
    except Exception as e:
        res['notes'] += f'iqr_err:{e};'

    # Tests: normalidad, skewtest, kurtosistest
    # Para normalidad: Shapiro-Wilk si n <= 5000, si no normaltest (D'Agostino)
    try:
        if n >= 3 and n <= 5000:
            st, p = stats.shapiro(x)
            res['normal_test'] = 'shapiro'
            res['normal_p'] = float(p)
        elif n > 5000:
            st, p = stats.normaltest(x)
            res['normal_test'] = 'normaltest'
            res['normal_p'] = float(p)
        else:
            res['normal_test'] = 'n<3'
            res['normal_p'] = float('nan')
    except Exception as e:
        res['notes'] += f'normal_err:{e};'

    try:
        if n >= 8:
            sst = stats.skewtest(x)
            res['skew_p'] = float(sst.pvalue)
        else:
            res['skew_p'] = float('nan')
    except Exception as e:
        res['notes'] += f'skewtest_err:{e};'

    try:
        if n >= 20:
            kst = stats.kurtosistest(x)
            res['kurt_p'] = float(kst.pvalue)
        else:
            res['kurt_p'] = float('nan')
    except Exception as e:
        res['notes'] += f'kurttest_err:{e};'

    # Decisión simple sobre si la media es representativa
    # Criterios (heurísticos):
    # - Si pasa normalidad (p > alpha) => media representativa
    # - Si no normal but |skew| < 0.5 and outliers < 5% and |kurtosis| < 2 => aceptable
    mean_repr = False
    try:
        if not np.isnan(res['normal_p']) and res['normal_p'] > alpha:
            mean_repr = True
        else:
            sk_ok = (not np.isnan(res['skew'])) and (abs(res['skew']) < 0.5)
            kurt_ok = (not np.isnan(res['kurtosis'])) and (abs(res['kurtosis']) < 2)
            out_ok = (not np.isnan(res['outlier_prop'])) and (res['outlier_prop'] < 0.05)
            if sk_ok and kurt_ok and out_ok:
                mean_repr = True
    except Exception as e:
        res['notes'] += f'mean_repr_err:{e};'

    res['mean_representative'] = bool(mean_repr)
    return res


def analyze_dataframe(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Analiza todas las columnas numéricas del DataFrame y retorna un resumen.
    """
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in numeric:
        arr = df[col].to_numpy()
        stats_res = analyze_column(arr, alpha=alpha)
        stats_res['column'] = col
        rows.append(stats_res)

    summary = pd.DataFrame(rows)
    # Orden para legibilidad
    cols_order = ['column', 'n', 'mean', 'median', 'std', 'skew', 'skew_p', 'kurtosis', 'kurt_p', 'normal_test', 'normal_p', 'outlier_prop', 'mean_representative', 'notes']
    summary = summary[cols_order]
    return summary


def main():
    parser = argparse.ArgumentParser(description='Statistical evaluation of DataFrame numeric columns')
    parser.add_argument('--input', '-i', type=str, default='data/evaluation_report_generated.json', help='Input JSON (evaluation_report_generated.json)')
    parser.add_argument('--out', '-o', type=str, default='data/statistical_summary.csv', help='Output CSV path for summary')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--save-json', action='store_true', help='Save summary also as JSON')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    df = generate_dataframe_questions_from_file(input_path)
    summary = analyze_dataframe(df, alpha=args.alpha)

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f'Statistical summary saved to: {out_csv}')

    if args.save_json:
        out_json = out_csv.with_suffix('.json')
        summary.to_json(out_json, orient='records', force_ascii=False, indent=2)
        print(f'Statistical summary (json) saved to: {out_json}')


if __name__ == '__main__':
    main()
