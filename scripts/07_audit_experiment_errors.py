# -*- coding: utf-8 -*-
"""Audit classification errors by experimental group."""

import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("WILEY_PROJECT_DIR", Path(__file__).resolve().parents[1]))
PCA_ART = ROOT / 'PCA_32_sensitivity_FROM_VERIFIED_PUMAP_SPLITS' / 'PCA_full_nowhiten_32' / 'artifacts'
OUT = ROOT / '09_AUDITORIA_EXPERIMENTOS_WILEY'
REPS = {'PUMAP':'PUMAP_Robust_md005_tw075','PCA':'PCA_full_nowhiten_32'}
SOURCES = {
    'ExtraTrees':'03_LGBM_EXTRATREES_BEST_PCA_AND_PUMAP',
    'LightGBM':'03_LGBM_EXTRATREES_BEST_PCA_AND_PUMAP',
    'RandomForest':'04_REMAINING_CLASSIFIERS_RF_XGB_CATBOOST_BEST_PCA_AND_PUMAP',
    'XGBoost':'04_REMAINING_CLASSIFIERS_RF_XGB_CATBOOST_BEST_PCA_AND_PUMAP',
    'CatBoost':'06_CATBOOST_24_CONFIGURACIONES_PCA_PUMAP',
}
CLASSES = {0:'Healthy',1:'6Nm',2:'9Nm',3:'NoBolt'}

def reconstruct_metadata():
    records=[]
    for fold in range(1,6):
        path=PCA_ART / f'fold{fold}_PCA_full_nowhiten_32_embeddings.npz'
        if not path.is_file():
            raise FileNotFoundError(f'No se encuentra el embedding PCA del fold {fold}: {path}')
        with np.load(path, allow_pickle=False) as data:
            for name in ['sample_index_test','groups_test','y_test']:
                if name not in data.files:
                    raise ValueError(f'{path.name}: falta {name}; campos: {data.files}')
            idx=np.asarray(data['sample_index_test']).reshape(-1)
            groups=np.asarray(data['groups_test']).reshape(-1)
            labels=np.asarray(data['y_test']).reshape(-1)
        if not (len(idx)==len(groups)==len(labels)):
            raise ValueError(f'Longitudes incompatibles en fold {fold}')
        records.append(pd.DataFrame({'sample_index':idx.astype(int),
                                     'group_original':groups.astype(int),
                                     'y_ref':labels.astype(int),
                                     'fold':fold}))
    ref=pd.concat(records, ignore_index=True)
    if len(ref)!=3840 or not ref.sample_index.is_unique:
        raise ValueError('Los embeddings PCA no cubren exactamente 3840 indices unicos')
    ref=ref.sort_values('sample_index').reset_index(drop=True)
    if not np.array_equal(ref.sample_index.to_numpy(),np.arange(3840)):
        raise ValueError('Faltan indices en los embeddings PCA')
    if ref.group_original.nunique()!=80:
        raise ValueError(f'Se esperaban 80 experimentos, hay {ref.group_original.nunique()}')
    if (ref.groupby('group_original').fold.nunique()>1).any():
        raise ValueError('Un experimento aparece en varios folds externos')
    if (ref.groupby('group_original').y_ref.nunique()>1).any():
        raise ValueError('Un experimento contiene distintas etiquetas')
    ref['class_name']=ref.y_ref.map(CLASSES)
    if ref.class_name.isna().any():
        raise ValueError('Clases no reconocidas en PCA')
    # Grupo numerico original, sin inventar nombres de archivos experimentales.
    return ref

def audit_one(ref, family, clf):
    path=ROOT / SOURCES[clf] / REPS[family] / clf / 'external_oof_predictions_all.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Falta CSV {family}/{clf}: {path}')
    pred=pd.read_csv(path)
    required={'sample_index','fold','y_true','y_pred'}
    if not required.issubset(pred):
        raise ValueError(f'{path}: faltan {required-set(pred.columns)}')
    for col in required:
        vals=pd.to_numeric(pred[col],errors='raise')
        if vals.isna().any() or not np.array_equal(vals.to_numpy(), vals.astype(int).to_numpy()):
            raise ValueError(f'Columna {col} invalida en {family}/{clf}')
        pred[col]=vals.astype(int)
    if len(pred)!=3840 or not pred.sample_index.is_unique:
        raise ValueError(f'{family}/{clf}: faltan o se duplican muestras')
    joined=ref.merge(pred[['sample_index','fold','y_true','y_pred']],on='sample_index',how='left',validate='one_to_one',suffixes=('_ref','_pred'))
    if joined.y_pred.isna().any():
        raise ValueError(f'{family}/{clf}: muestra sin prediccion')
    if not np.array_equal(joined.fold_ref.to_numpy(),joined.fold_pred.to_numpy()):
        raise ValueError(f'{family}/{clf}: folds diferentes respecto de PCA')
    if not np.array_equal(joined.y_ref.to_numpy(),joined.y_true.to_numpy()):
        raise ValueError(f'{family}/{clf}: etiquetas diferentes respecto de PCA')
    joined['error']=joined.y_true.ne(joined.y_pred)
    joined['representation']=family
    joined['classifier']=clf
    joined['predicted_class']=joined.y_pred.map(CLASSES)
    groups=(joined.groupby(['representation','classifier','group_original','fold_ref','class_name'],as_index=False)
            .agg(windows=('error','size'),errors=('error','sum')))
    groups['accuracy_experiment']=1-groups.errors/groups.windows
    errs=joined.loc[joined.error,['representation','classifier','sample_index','group_original','fold_ref',
                                  'class_name','predicted_class','y_true','y_pred']].copy()
    return groups,errs

def main():
    ref=reconstruct_metadata()
    OUT.mkdir(exist_ok=True,parents=True)
    print(f'[OK] Metadatos recuperados de PCA: {len(ref)} ventanas; {ref.group_original.nunique()} experimentos; 5 folds')
    ref.to_csv(OUT/'MAPA_MUESTRA_GRUPO_FOLD.csv',index=False,encoding='utf-8-sig')
    group_rows=[];error_rows=[];summaries=[]
    for family in ['PUMAP','PCA']:
        for clf in SOURCES:
            grouped,err=audit_one(ref,family,clf)
            group_rows.append(grouped);error_rows.append(err)
            summaries.append({'representation':family,'classifier':clf,
                              'errors':len(err),'groups_with_errors':int((grouped.errors>0).sum()),
                              'groups_perfect':int((grouped.errors==0).sum())})
            print(f'[OK] {family:5s}/{clf:12s} : {len(err):3d} errores en {int((grouped.errors>0).sum())} experimento(s)')
    group_df=pd.concat(group_rows,ignore_index=True)
    error_df=pd.concat(error_rows,ignore_index=True)
    summary_df=pd.DataFrame(summaries)
    group_df.to_csv(OUT/'METRICAS_80_EXPERIMENTOS_X_10_PIPELINES.csv',index=False,encoding='utf-8-sig')
    error_df.to_csv(OUT/'ERRORES_OOF_IDENTIFICADOS_POR_EXPERIMENTO.csv',index=False,encoding='utf-8-sig')
    summary_df.to_csv(OUT/'RESUMEN_ERRORES_Y_GRUPOS.csv',index=False,encoding='utf-8-sig')
    print('\n=== EXPERIMENTOS CON ERRORES: PUMAP + CATBOOST ===')
    focus=group_df[(group_df.representation=='PUMAP')&(group_df.classifier=='CatBoost')&(group_df.errors>0)]
    print(focus[['group_original','fold_ref','class_name','windows','errors','accuracy_experiment']].to_string(index=False))
    print('\n=== RESUMEN DE LOS DIEZ PIPELINES ===')
    print(summary_df.to_string(index=False))
    print('\n=== ERRORES POR FOLD ===')
    print(error_df.groupby(['representation','classifier','fold_ref']).size().unstack(fill_value=0).reindex(columns=range(1,6),fill_value=0).to_string())
    print('\nReportes:',OUT)

if __name__=='__main__':
    main()
