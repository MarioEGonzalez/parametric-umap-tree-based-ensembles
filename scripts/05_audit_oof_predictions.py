# -*- coding: utf-8 -*-
"""Check external out-of-fold predictions."""

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef

ROOT = Path(os.environ.get("WILEY_PROJECT_DIR", Path(__file__).resolve().parents[1]))
SOURCES = {
    '03': ROOT / '03_LGBM_EXTRATREES_BEST_PCA_AND_PUMAP',
    '04': ROOT / '04_REMAINING_CLASSIFIERS_RF_XGB_CATBOOST_BEST_PCA_AND_PUMAP',
    '06': ROOT / '06_CATBOOST_24_CONFIGURACIONES_PCA_PUMAP',
}
OUT = ROOT / '07_AUDITORIA_OOF_FINAL'
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = [0, 1, 2, 3]
NAMES = ['Healthy', '6Nm', '9Nm', 'NoBolt']
EXPECTED_CLASS_SUPPORT = [980, 960, 960, 940]
TARGETS = [
    ('PUMAP_Robust_md005_tw075', 'ExtraTrees', '03'),
    ('PUMAP_Robust_md005_tw075', 'CatBoost', '06'),
    ('PCA_full_nowhiten_32', 'XGBoost', '04'),
]

def find_pair_files(base, representation, classifier, filename):
    # Se buscan archivos por su ubicacion relativa, no se mezclan corridas.
    return [p for p in base.rglob(filename)
            if representation.lower() in str(p.parent).lower()
            and classifier.lower() in str(p.parent).lower()]

def audit_pair(rep, clf, source):
    base = SOURCES[source]
    files = find_pair_files(base, rep, clf, 'external_oof_predictions_all.csv')
    if len(files) != 1:
        print(f'[PENDIENTE] {rep} + {clf}: encontrados {len(files)} CSV de predicciones en {base}')
        return None, None
    path = files[0]
    data = pd.read_csv(path)
    needed = ['sample_index', 'fold', 'y_true', 'y_pred']
    missing = [c for c in needed if c not in data.columns]
    if missing:
        print(f'[ERROR] {path}: faltan columnas {missing}')
        return None, None
    try:
        for c in needed:
            data[c] = pd.to_numeric(data[c], errors='raise').astype(int)
    except (ValueError, TypeError) as exc:
        print(f'[ERROR] {rep} + {clf}: valores no numericos ({exc})')
        return None, None
    cm = confusion_matrix(data.y_true, data.y_pred, labels=CLASSES)
    counts = cm.sum(axis=1).tolist()
    is_valid = (len(data) == 3840 and data.sample_index.nunique() == 3840
                and data.sample_index.min() == 0 and data.sample_index.max() == 3839
                and sorted(data.fold.unique().tolist()) == [1, 2, 3, 4, 5]
                and counts == EXPECTED_CLASS_SUPPORT
                and set(data.y_true.unique()).issubset(CLASSES)
                and set(data.y_pred.unique()).issubset(CLASSES))
    label = f'{rep}__{clf}'
    pd.DataFrame(cm, index=NAMES, columns=NAMES).to_csv(OUT / f'{label}_confusion_counts.csv')
    pct = np.divide(cm * 100, cm.sum(axis=1, keepdims=True),
                    out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True)!=0)
    pd.DataFrame(pct, index=NAMES, columns=NAMES).round(3).to_csv(OUT / f'{label}_confusion_percent_row.csv')
    details = data.copy()
    details['error'] = (details.y_true != details.y_pred).astype(int)
    fold_report = details.groupby('fold').agg(n=('sample_index','size'), errors=('error','sum'))
    fold_report['error_pct'] = 100 * fold_report.errors / fold_report.n
    fold_report.to_csv(OUT / f'{label}_errors_by_fold.csv')
    if 'group' in details and (details['group'] != -1).any():
        details.groupby(['fold', 'group'], as_index=False).agg(
            n=('sample_index', 'size'), errors=('error', 'sum')
        ).to_csv(OUT / f'{label}_errors_by_experiment.csv', index=False)
    metrics = {
        'representation': rep, 'classifier': clf, 'source': source,
        'n_oof': len(data), 'n_unique': int(data.sample_index.nunique()),
        'confusion_sum': int(cm.sum()), 'is_valid': bool(is_valid),
        'accuracy': accuracy_score(data.y_true, data.y_pred),
        'macro_f1': f1_score(data.y_true, data.y_pred, average='macro', labels=CLASSES),
        'mcc': matthews_corrcoef(data.y_true, data.y_pred),
        'errors': int(details.error.sum()), 'file': str(path),
    }
    print(f"[{ 'OK' if is_valid else 'REVISAR' }] {clf} / {rep}: "
          f"N={len(data)}, unicos={data.sample_index.nunique()}, "
          f"errores={metrics['errors']}, Macro-F1={metrics['macro_f1']:.6f}")
    return metrics, details

def check_catboost_grid():
    rep = 'PUMAP_Robust_md005_tw075'
    base = SOURCES['06']
    files = find_pair_files(base, rep, 'CatBoost', 'all_internal_grid_metrics.csv')
    if len(files) != 1:
        print(f'[PENDIENTE] Grid CatBoost PUMAP: {len(files)} archivos encontrados')
        return
    df = pd.read_csv(files[0])
    if not {'fold', 'config_index'}.issubset(df.columns):
        print('[REVISAR] Grid CatBoost: faltan fold/config_index')
        return
    report = df.groupby('fold').agg(rows=('config_index','size'),
                                    unique_configs=('config_index','nunique'))
    print('\nCatBoost - configuraciones por fold:')
    print(report.to_string())
    report.to_csv(OUT / 'catboost_24_grid_check.csv')
    if not (len(report) == 5 and (report.rows == 24).all() and (report.unique_configs == 24).all()):
        print('[REVISAR] No todos los folds tienen exactamente 24 configuraciones')

def main():
    print('AUDITORIA EXTERNA OOF - WILEY\n')
    collected = []
    by_pair = {}
    for rep, clf, source in TARGETS:
        if not SOURCES[source].exists():
            print(f'[PENDIENTE] No existe la carpeta {SOURCES[source]}')
            continue
        m, df = audit_pair(rep, clf, source)
        if m is not None:
            collected.append(m)
            by_pair[(rep, clf)] = df
    if collected:
        summary = pd.DataFrame(collected)
        summary.to_csv(OUT / 'auditoria_tres_pipelines.csv', index=False)
        print('\nResumen:\n', summary[['classifier','n_oof','n_unique','errors','macro_f1','mcc','is_valid']].to_string(index=False))
    a = by_pair.get(('PUMAP_Robust_md005_tw075', 'ExtraTrees'))
    b = by_pair.get(('PUMAP_Robust_md005_tw075', 'CatBoost'))
    if a is not None and b is not None:
        a = a.set_index('sample_index').sort_index()
        b = b.set_index('sample_index').sort_index()
        common = a.index.intersection(b.index)
        same_truth = bool((a.loc[common, 'y_true'] == b.loc[common, 'y_true']).all())
        differences = int((a.loc[common, 'y_pred'] != b.loc[common, 'y_pred']).sum())
        print(f'\nExtraTrees vs CatBoost: indices comunes={len(common)}, etiquetas iguales={same_truth}, predicciones diferentes={differences}')
        if len(common) == 3840 and same_truth:
            compare = pd.DataFrame({'sample_index': common,
                                    'y_true': a.loc[common, 'y_true'].to_numpy(),
                                    'pred_extratrees': a.loc[common, 'y_pred'].to_numpy(),
                                    'pred_catboost': b.loc[common, 'y_pred'].to_numpy()})
            compare[compare.pred_extratrees != compare.pred_catboost].to_csv(
                OUT / 'extratrees_vs_catboost_predicciones_distintas.csv', index=False)
    check_catboost_grid()
    print(f'\nReportes guardados en: {OUT}')

if __name__ == '__main__':
    main()
