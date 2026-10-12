# -*- coding: utf-8 -*-
"""Aggregate results from ten representation-classifier pipelines."""

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef,
                             cohen_kappa_score, confusion_matrix)

ROOT = Path(os.environ.get("WILEY_PROJECT_DIR", Path(__file__).resolve().parents[1]))
SOURCES = {
    '03': ROOT / '03_LGBM_EXTRATREES_BEST_PCA_AND_PUMAP',
    '04': ROOT / '04_REMAINING_CLASSIFIERS_RF_XGB_CATBOOST_BEST_PCA_AND_PUMAP',
    '06': ROOT / '06_CATBOOST_24_CONFIGURACIONES_PCA_PUMAP',
}
OUT = ROOT / '08_CONSOLIDACION_FINAL_WILEY'
CLASSES = [0, 1, 2, 3]
NAMES = ['Healthy', '6Nm', '9Nm', 'NoBolt']
SUPPORT = [980, 960, 960, 940]
REPRESENTATIONS = {'PUMAP': 'PUMAP_Robust_md005_tw075', 'PCA': 'PCA_full_nowhiten_32'}
CLASSIFIERS = {'ExtraTrees': '03', 'LightGBM': '03', 'RandomForest': '04',
               'XGBoost': '04', 'CatBoost': '06'}
EXPECTED_TEST_FOLD_SIZES = [769, 768, 768, 766, 769]

def discover(rep, clf, source):
    base = SOURCES[source]
    if not base.exists():
        raise FileNotFoundError(f'Carpeta no encontrada: {base}')
    # Evita la deteccion ambigua por coincidencias parciales del nombre.
    candidates = [p for p in base.rglob('external_oof_predictions_all.csv')
                  if p.parent.name.lower() == clf.lower()
                  and p.parent.parent.name == rep]
    if len(candidates) != 1:
        raise RuntimeError(f'{rep}/{clf}: se esperaban 1 CSV, encontrados {len(candidates)}: {candidates}')
    return candidates[0]

def read_checked(path, label):
    df = pd.read_csv(path)
    required = ['sample_index', 'fold', 'y_true', 'y_pred']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{label}: faltan columnas {missing}')
    for col in required:
        v = pd.to_numeric(df[col], errors='raise')
        if v.isna().any() or not np.all(v == np.floor(v)):
            raise ValueError(f'{label}: columna {col} no es entera sin nulos')
        df[col] = v.astype(int)
    if len(df) != 3840 or df.sample_index.nunique() != 3840:
        raise ValueError(f'{label}: deben existir 3840 indices unicos')
    if not np.array_equal(np.sort(df.sample_index.to_numpy()), np.arange(3840)):
        raise ValueError(f'{label}: indices distintos de 0..3839')
    if set(df.fold.unique()) != set(range(1, 6)):
        raise ValueError(f'{label}: folds inesperados')
    folds = df.groupby('fold').size().reindex(range(1, 6)).tolist()
    if folds != EXPECTED_TEST_FOLD_SIZES:
        raise ValueError(f'{label}: tamanos de fold {folds} distintos a los esperados')
    if not set(df.y_true.unique()).issubset(CLASSES) or not set(df.y_pred.unique()).issubset(CLASSES):
        raise ValueError(f'{label}: etiquetas fuera de 0,1,2,3')
    cm = confusion_matrix(df.y_true, df.y_pred, labels=CLASSES)
    if cm.sum(axis=1).tolist() != SUPPORT:
        raise ValueError(f'{label}: soportes reales incorrectos {cm.sum(axis=1).tolist()}')
    return df.sort_values('sample_index').reset_index(drop=True), cm

def metrics(y, pred):
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average='macro', labels=CLASSES, zero_division=0),
                weighted_f1=f1_score(y, pred, average='weighted', labels=CLASSES, zero_division=0),
                macro_precision=precision_score(y, pred, average='macro', labels=CLASSES, zero_division=0),
                macro_recall=recall_score(y, pred, average='macro', labels=CLASSES, zero_division=0),
                mcc=matthews_corrcoef(y, pred), kappa=cohen_kappa_score(y, pred))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, fold_records, per_class, loaded, errors = [], [], [], {}, []
    reference = None
    for family, rep in REPRESENTATIONS.items():
        for clf, source in CLASSIFIERS.items():
            label = f'{family}/{clf}'
            try:
                path = discover(rep, clf, source)
                df, cm = read_checked(path, label)
                if reference is None:
                    reference = df[['sample_index', 'fold', 'y_true']].copy()
                else:
                    for col in ['fold', 'y_true']:
                        if not np.array_equal(df[col].to_numpy(), reference[col].to_numpy()):
                            raise ValueError(f'{label}: la columna {col} difiere de otros pipelines')
                mm = metrics(df.y_true, df.y_pred)
                records.append(dict(representation_family=family, representation=rep,
                                    classifier=clf, n_oof=len(df), errors=int((df.y_true!=df.y_pred).sum()),
                                    **mm, source_file=str(path)))
                loaded[(family, clf)] = df
                counts = pd.DataFrame(cm, index=NAMES, columns=NAMES)
                counts.to_csv(OUT / f'CM_{family}_{clf}_counts.csv', encoding='utf-8-sig')
                percentages = 100.0 * cm / cm.sum(axis=1, keepdims=True)
                pd.DataFrame(percentages, index=NAMES, columns=NAMES).to_csv(
                    OUT / f'CM_{family}_{clf}_row_percent.csv', float_format='%.6f', encoding='utf-8-sig')
                for fold, sub in df.groupby('fold', sort=True):
                    fm = metrics(sub.y_true, sub.y_pred)
                    fold_records.append(dict(representation_family=family, classifier=clf, fold=int(fold),
                                             n_test=len(sub), errors=int((sub.y_true!=sub.y_pred).sum()), **fm))
                for cls, name in zip(CLASSES, NAMES):
                    mask = df.y_true == cls
                    sub = df[mask]
                    per_class.append(dict(representation_family=family, classifier=clf, class_name=name,
                                          support=len(sub), errors=int((sub.y_true!=sub.y_pred).sum()),
                                          recall=float(np.mean(sub.y_pred == cls))))
                print(f'[OK] {label:22s} | errores={records[-1]["errors"]:3d} | Macro-F1={mm["macro_f1"]:.6f}')
            except Exception as exc:
                errors.append(dict(pipeline=label, reason=str(exc)))
                print(f'[REVISAR] {label}: {exc}')
    if records:
        summary = pd.DataFrame(records).sort_values(['macro_f1', 'mcc'], ascending=False)
        summary.to_csv(OUT / 'TABLA_MAESTRA_10_PIPELINES_OOF.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(fold_records).to_csv(OUT / 'METRICAS_POR_FOLD.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(per_class).to_csv(OUT / 'METRICAS_POR_CLASE.csv', index=False, encoding='utf-8-sig')
        print('\nTABLA MAESTRA (OOF EXTERNO):')
        print(summary[['representation_family','classifier','accuracy','macro_f1','mcc','errors']].to_string(index=False, float_format=lambda x:f'{x:.6f}'))
    if ('PUMAP','ExtraTrees') in loaded and ('PUMAP','CatBoost') in loaded:
        a, b = loaded[('PUMAP','ExtraTrees')], loaded[('PUMAP','CatBoost')]
        equal = int((a.y_pred == b.y_pred).sum())
        print(f'\nExtraTrees vs CatBoost (PUMAP): {equal}/3840 predicciones identicas')
        pd.DataFrame({'sample_index':a.sample_index, 'y_true':a.y_true,
                      'extratrees':a.y_pred,'catboost':b.y_pred}).to_csv(
                          OUT/'COMPARACION_ET_CATBOOST_PUMAP.csv',index=False,encoding='utf-8-sig')
    for family, rep in REPRESENTATIONS.items():
        clf, base = 'CatBoost', SOURCES['06'] / rep / 'CatBoost'
        candidates = list(base.rglob('all_internal_grid_metrics.csv')) if base.exists() else []
        if len(candidates) != 1:
            errors.append(dict(pipeline=f'{family}/CatBoost grid',reason=f'CSV de grid encontrados: {len(candidates)}'))
            continue
        d = pd.read_csv(candidates[0])
        if not {'fold','config_index'}.issubset(d.columns):
            errors.append(dict(pipeline=f'{family}/CatBoost grid',reason='Faltan fold/config_index'))
            continue
        counts = d.groupby('fold')['config_index'].agg(['size','nunique']).reindex(range(1,6))
        if counts.isna().any().any() or not ((counts['size']==24)&(counts['nunique']==24)).all():
            errors.append(dict(pipeline=f'{family}/CatBoost grid',reason=f'No hay 24 configs unicas por fold: {counts.to_dict()}'))
        print(f'CatBoost {family}: filas por fold {counts["size"].tolist()}, configs unicas {counts["nunique"].tolist()}')
    pd.DataFrame(errors, columns=['pipeline','reason']).to_csv(OUT/'INCIDENCIAS.csv',index=False,encoding='utf-8-sig')
    manifest = {'n_pipelines_ok':len(records),'n_expected':10, 'all_verified':len(records)==10 and not errors,
                'problems':errors, 'out_dir':str(OUT)}
    (OUT/'AUDIT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nAUDITORIA GLOBAL:', 'OK (10/10)' if manifest['all_verified'] else 'PENDIENTE: revisar incidencias')
    print('Salidas:',OUT)

if __name__ == '__main__':
    main()
