# -*- coding: utf-8 -*-
"""Evaluate selected CatBoost configurations."""

import os
from pathlib import Path
from datetime import datetime
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from catboost import CatBoostClassifier

ROOT = Path(os.environ.get("WILEY_PROJECT_DIR", Path(__file__).resolve().parents[1]))
REP = 'PUMAP_Robust_md005_tw075'
ARTIFACTS = ROOT / REP / 'artifacts'
ORIGINAL = ROOT / '06_CATBOOST_24_CONFIGURACIONES_PCA_PUMAP' / REP / 'CatBoost'
ORIGINAL_CSV = ORIGINAL / 'external_oof_predictions_all.csv'
ORIGINAL_GRID = ORIGINAL / 'all_internal_grid_metrics.csv'
OUTPUT = ROOT / '16_CATBOOST_REENTRENAMIENTO_CONFIGS_1_13_24'
N_JOBS = 3
GLOBAL_SEED = 42

CONFIG_INDICES = (1, 13, 24)
EXPECTED = {
  1: {'border_count':128,'depth':4,'iterations':200,'l2_leaf_reg':3.0,'learning_rate':0.03},
  13:{'border_count':128,'depth':6,'iterations':500,'l2_leaf_reg':3.0,'learning_rate':0.03},
  24:{'border_count':128,'depth':8,'iterations':500,'l2_leaf_reg':5.0,'learning_rate':0.08},
}
CLASS_LABELS = [0,1,2,3]
CLASS_NAMES = ['Healthy','6Nm','9Nm','NoBolt']

def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda: f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def get_array(z, options, source):
    for key in options:
        if key in z.files: return np.asarray(z[key])
    raise KeyError(f'{source}: faltan {options}; presentes: {z.files}')

def load_fold(fold):
    file = ARTIFACTS / f'fold{fold}_{REP}_embeddings.npz'
    if not file.is_file(): raise FileNotFoundError(f'No encuentro embedding original: {file}')
    with np.load(file,allow_pickle=False) as d:
        r = {
            'Z_train':get_array(d,['Z_train','z_train'],file).astype(np.float32),
            'Z_val':get_array(d,['Z_val','z_val'],file).astype(np.float32),
            'Z_test':get_array(d,['Z_test','z_test'],file).astype(np.float32),
            'y_train':get_array(d,['y_train','y_tr2'],file).reshape(-1).astype(int),
            'y_val':get_array(d,['y_val','y_va'],file).reshape(-1).astype(int),
            'y_test':get_array(d,['y_test','yte'],file).reshape(-1).astype(int),
            'idx_test':get_array(d,['sample_index_test','idx_test','test_idx'],file).reshape(-1).astype(int),
        }
    for suffix in ['train','val','test']:
        if r['Z_'+suffix].shape != (len(r['y_'+suffix]),32):
            raise ValueError(f'Fold {fold}: Z_{suffix} no coincide con y_{suffix} o dimension 32')
    if len(r['idx_test'])!=len(r['y_test']): raise ValueError('indices/etiquetas incoherentes')
    if len(set(r['idx_test']))!=len(r['idx_test']): raise ValueError('indices duplicados en test')
    return r,file

def main():
    if not ROOT.is_dir(): raise FileNotFoundError(f'ROOT inexistente: {ROOT}')
    for path in [ORIGINAL_CSV,ORIGINAL_GRID]:
        if not path.is_file():raise FileNotFoundError(f'No existe: {path}')
    original=pd.read_csv(ORIGINAL_CSV)
    for c in ('fold','sample_index','y_true','y_pred'):
        if c not in original.columns:raise ValueError(f'CSV sin {c}')
        original[c]=pd.to_numeric(original[c],errors='raise').astype(int)
    if len(original)!=3840 or original.sample_index.nunique()!=3840:
        raise ValueError('CSV original sin los 3840 indices unicos esperados')
    original_idx=original.set_index('sample_index',verify_integrity=True)

    grid=pd.read_csv(ORIGINAL_GRID)
    for c in ('fold','config_index','params_json'):
        if c not in grid.columns:raise ValueError(f'Grid original sin {c}')
    for i in CONFIG_INDICES:
        rows=grid[grid.config_index.eq(i)]
        if len(rows)!=5 or set(rows.fold.astype(int))!=set(range(1,6)):
            raise ValueError(f'Config {i} no fue evaluada exactamente una vez en cada fold')
        for raw in rows.params_json:
            params=json.loads(raw)
            if params!=EXPECTED[i]:
                raise ValueError(f'Config {i}: parametros CSV != parametros esperados: {params}')
    print('Se confirmaron configuraciones 1, 13 y 24 en el CSV original de validacion interna.')

    # Prevalidar todos los folds, clases, indices y scalers antes de comenzar.
    fold_data={}
    for fold in range(1,6):
        d,path=load_fold(fold)
        idx=d['idx_test']
        sub=original.loc[original.fold.eq(fold)].set_index('sample_index')
        if len(sub)!=len(idx) or set(sub.index)!=set(idx):
            raise ValueError(f'Fold {fold}: indices no coinciden con CatBoost original')
        if not np.array_equal(d['y_test'],sub.loc[idx,'y_true'].to_numpy(dtype=int)):
            raise ValueError(f'Fold {fold}: etiquetas externas no coinciden')
        model_path=ORIGINAL/'models'/f'fold{fold}_{REP}_CatBoost_classifier.pkl'
        scaler_path=ORIGINAL/'models'/f'fold{fold}_{REP}_CatBoost_scaler.pkl'
        if not model_path.is_file() or not scaler_path.is_file():
            raise FileNotFoundError(f'Fold {fold}: falta modelo/scaler CatBoost original')
        with open(model_path,'rb') as f: saved_model=pickle.load(f)
        if not isinstance(saved_model,CatBoostClassifier):
            raise TypeError(f'Fold {fold}: el artefacto no contiene CatBoostClassifier')
        for k,v in EXPECTED[1].items():
            if saved_model.get_params().get(k)!=v:
                raise ValueError(f'Fold {fold}: modelo original difiere en {k}')
        with open(scaler_path,'rb') as f:saved_scaler=pickle.load(f)
        dev=np.vstack((d['Z_train'],d['Z_val']))
        s=StandardScaler().fit(dev)
        if not np.allclose(s.mean_,saved_scaler.mean_,rtol=1e-11,atol=1e-11) or not np.allclose(s.scale_,saved_scaler.scale_,rtol=1e-11,atol=1e-11):
            raise ValueError(f'Fold {fold}: scaler recalculado difiere del original')
        fold_data[fold]=(d,s,dev,path)
        print(f'Fold {fold}: indices, etiquetas, modelo original y scaler verificados; test={len(idx)}')

    # Entrenamientos nuevos. No se leen modelos originales para obtener predicciones.
    rows=[]
    for fold,(d,scaler,dev,path) in fold_data.items():
        X_dev=scaler.transform(dev)
        y_dev=np.concatenate((d['y_train'],d['y_val']))
        X_test=scaler.transform(d['Z_test'])
        orig=original_idx.loc[d['idx_test'],'y_pred'].to_numpy(dtype=int)
        seed=GLOBAL_SEED + fold*10000  # semilla final EXACTA del script CatBoost 06
        for i in CONFIG_INDICES:
            model=CatBoostClassifier(**EXPECTED[i],loss_function='MultiClass',random_seed=seed,
                                     thread_count=N_JOBS,verbose=False,allow_writing_files=False)
            model.fit(X_dev,y_dev)
            pred=np.asarray(model.predict(X_test)).reshape(-1).astype(int)
            diff=int((pred!=orig).sum())
            errors=int((pred!=d['y_test']).sum())
            print(f'Fold {fold} | Config {i:02d} | errores={errors:3d} | distintas respecto a CatBoost original={diff:3d}')
            if i==1 and diff:
                raise RuntimeError(f'CONTROL FALLIDO, fold {fold}: la config 1 no reproduce {diff} etiquetas originales. '
                                   'Detenido para no interpretar resultados de otra ejecucion como comparables.')
            rows.append(pd.DataFrame({
                'config_index':i,'fold':fold,'sample_index':d['idx_test'],
                'y_true':d['y_test'],'y_pred':pred,'y_pred_catboost_original':orig,
                'error':pred!=d['y_test'],'different_from_original':pred!=orig,
            }))
    data=pd.concat(rows,ignore_index=True)
    summaries=[]
    per_fold=[]
    cat_ref=original_idx
    for i in CONFIG_INDICES:
        sub=data[data.config_index.eq(i)].sort_values('sample_index')
        if len(sub)!=3840 or sub.sample_index.nunique()!=3840:
            raise ValueError(f'Config {i}: OOF incompleto')
        summaries.append({
            'config_index':i,'accuracy':accuracy_score(sub.y_true,sub.y_pred),
            'macro_f1':f1_score(sub.y_true,sub.y_pred,average='macro',zero_division=0),
            'mcc':matthews_corrcoef(sub.y_true,sub.y_pred),
            'errors':int(sub.error.sum()),
            'different_from_original':int(sub.different_from_original.sum()),
        })
        for fold in range(1,6):
            z=sub[sub.fold.eq(fold)]
            per_fold.append({'config_index':i,'fold':fold,'n':len(z),'errors':int(z.error.sum()),
                             'different_from_original':int(z.different_from_original.sum()),
                             'macro_f1':f1_score(z.y_true,z.y_pred,average='macro',zero_division=0)})
    # Solo se crea carpeta cuando todas las validaciones y los entrenamientos terminaron bien.
    OUTPUT.mkdir(parents=True,exist_ok=True)
    data.to_csv(OUTPUT/'PREDICCIONES_3_CONFIGURACIONES_5_FOLDS.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(per_fold).to_csv(OUTPUT/'RESULTADOS_POR_FOLD.csv',index=False,encoding='utf-8-sig')
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUTPUT/'RESULTADOS_GLOBALES.csv',index=False,encoding='utf-8-sig')
    for i in CONFIG_INDICES:
        subset=data[data.config_index.eq(i)].sort_values('sample_index')
        cm=confusion_matrix(subset.y_true,subset.y_pred,labels=CLASS_LABELS)
        pd.DataFrame(cm,index=['real_'+x for x in CLASS_NAMES],
                     columns=['pred_'+x for x in CLASS_NAMES]).to_csv(OUTPUT/f'MATRIZ_CONFUSION_CONFIG_{i:02d}.csv')
        subset[subset.error|subset.different_from_original].to_csv(
            OUTPUT/f'ERRORES_Y_CAMBIOS_CONFIG_{i:02d}.csv',index=False,encoding='utf-8-sig')
    manifest={
        'source_grid':str(ORIGINAL_GRID),'source_grid_sha256':sha256(ORIGINAL_GRID),
        'original_predictions':str(ORIGINAL_CSV),'original_predictions_sha256':sha256(ORIGINAL_CSV),
        'embedding_folder':str(ARTIFACTS),'configurations':EXPECTED,
        'seed_formula':'42 + fold * 10000','n_jobs':N_JOBS,
        'method':'refit train+val standardized embeddings; external OOF predictions',
        'warning':'Configs 13 and 24 are exploratory external comparisons, not nested model selection.'
    }
    (OUTPUT/'TRAZABILIDAD.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n'+'='*85+'\nRESULTADOS GLOBALES\n'+summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nArchivos guardados en:',OUTPUT)
    print('Config 1 reproducida exactamente. Las alternativas NO modifican el ganador del articulo.')

if __name__=='__main__':main()
