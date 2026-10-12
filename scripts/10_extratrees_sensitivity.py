# -*- coding: utf-8 -*-
"""Evaluate selected ExtraTrees configurations."""

import os
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

ROOT = Path(os.environ.get("WILEY_PROJECT_DIR", Path(__file__).resolve().parents[1]))
REP = 'PUMAP_Robust_md005_tw075'
ART = ROOT / 'PUMAP_Robust_md005_tw075' / 'artifacts'  # se detectara automaticamente si no existe
ORIG_DIR = ROOT / '03_LGBM_EXTRATREES_BEST_PCA_AND_PUMAP' / REP / 'ExtraTrees'
ORIG_CSV = ORIG_DIR / 'external_oof_predictions_all.csv'
OUT = ROOT / '15_REENTRENAMIENTO_EXTRATREES_TRES_CONFIGS'

CONFIGS = {
    'A_GANADORA_50': dict(n_estimators=50, max_depth=None, max_features=0.5,
                          criterion='gini', min_samples_split=2, min_samples_leaf=1,
                          bootstrap=False, class_weight='balanced_subsample'),
    'B_200_ARBOLES': dict(n_estimators=200, max_depth=30, max_features=0.7,
                          criterion='gini', min_samples_split=2, min_samples_leaf=1,
                          bootstrap=False, class_weight='balanced_subsample'),
    'C_650_ARBOLES': dict(n_estimators=650, max_depth=30, max_features=0.7,
                          criterion='gini', min_samples_split=2, min_samples_leaf=1,
                          bootstrap=False, class_weight='balanced_subsample'),
}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def find_embedding(fold):
    filename = f'fold{fold}_{REP}_embeddings.npz'
    matches = list(ROOT.rglob(filename)) + list(ROOT.rglob(f'fold{fold}_{REP}_embeddings(1).npz'))
    # duplicados de copias en varias carpetas -> exigir resolver de manera explicita
    matches = sorted(set(p.resolve() for p in matches))
    if not matches:
        raise FileNotFoundError(f'No se encuentra {filename} en {ROOT}; revisa ROOT')
    if len(matches)>1:
        print(f'\nATENCION: {len(matches)} posibles embeddings fold {fold}:')
        for p in matches: print(' ',p)
        # Elegir solo si tienen hashes identicos. De otro modo detener antes de crear resultados.
        hashes = {sha256(p) for p in matches}
        if len(hashes)>1:
            raise RuntimeError('Hay varias versiones distintas de embeddings. Configura el origen exacto antes de ejecutar.')
    return matches[0]

def original_model(fold):
    p=ORIG_DIR / 'models' / f'fold{fold}_{REP}_ExtraTrees_classifier.pkl'
    if not p.is_file():raise FileNotFoundError(p)
    # Solo archivos propios de confianza: pickle puede ejecutar codigo al cargar.
    with open(p,'rb') as f: m=pickle.load(f)
    if not isinstance(m,ExtraTreesClassifier):
        raise TypeError(f'{p} contiene {type(m)} y no ExtraTreesClassifier')
    return m

def main():
    if not ROOT.is_dir():raise FileNotFoundError(f'Carpeta raiz inexistente: {ROOT}')
    if not ORIG_CSV.is_file():raise FileNotFoundError(f'CSV original inexistente: {ORIG_CSV}')
    old=pd.read_csv(ORIG_CSV)
    must={'sample_index','fold','y_true','y_pred'}
    if not must.issubset(old.columns):raise ValueError(f'CSV original carece de {must-set(old.columns)}')
    old=old[list(must)].copy()
    for c in must:old[c]=pd.to_numeric(old[c]).astype(int)
    if len(old)!=3840 or old.sample_index.nunique()!=3840:raise ValueError('CSV original no tiene 3840 indices unicos')

    # Validar todas las particiones ANTES de entrenar para evitar mezclar distintas versiones.
    folds={}
    for fold in range(1,6):
        p=find_embedding(fold)
        with np.load(p,allow_pickle=True) as data:
            needed=['Z_train','Z_val','Z_test','y_train','y_val','y_test','idx_train','idx_val','idx_test']
            if any(k not in data.files for k in needed):raise ValueError(f'{p}: faltan claves')
            d={k:np.asarray(data[k]).copy() for k in needed}
        for key in ['y_train','y_val','y_test','idx_train','idx_val','idx_test']:
            d[key]=d[key].reshape(-1).astype(int)
        if any(d['Z_'+s].shape!=(len(d['y_'+s]),32) for s in ['train','val','test']):
            raise ValueError(f'Fold {fold}: dimensiones incoherentes')
        sub=old.loc[old.fold.eq(fold)].set_index('sample_index')
        idx=d['idx_test']
        if len(sub)!=len(idx) or set(idx)!=set(sub.index):
            raise ValueError(f'Fold {fold}: indices externos no coinciden con CSV original, detener')
        if not np.array_equal(d['y_test'],sub.loc[idx,'y_true'].to_numpy()):
            raise ValueError(f'Fold {fold}: etiquetas reales no coinciden con CSV original')
        folds[fold]=(d,p,original_model(fold))
        print(f'Fold {fold} verificado: {len(idx)} ventanas, {p}')

    OUT.mkdir(parents=True,exist_ok=True)
    all_predictions=[]
    fold_summary=[]
    for fold,(d,p,m_old) in folds.items():
        Xdev=np.concatenate([d['Z_train'],d['Z_val']],axis=0)
        ydev=np.concatenate([d['y_train'],d['y_val']],axis=0)
        scaler=StandardScaler()
        Xdev_scaled=scaler.fit_transform(Xdev)
        Xtest_scaled=scaler.transform(d['Z_test'])
        old_scaler_path=ORIG_DIR/'models'/f'fold{fold}_{REP}_ExtraTrees_scaler.pkl'
        if old_scaler_path.exists():
            with open(old_scaler_path,'rb') as fh:s_old=pickle.load(fh)
            if not np.allclose(scaler.mean_,s_old.mean_,rtol=1e-11,atol=1e-11) or not np.allclose(scaler.scale_,s_old.scale_,rtol=1e-11,atol=1e-11):
                raise ValueError(f'Fold {fold}: scaler de train+val NO coincide con scaler original. No comparar.')
        params_old=m_old.get_params()
        # Verificar que configuracion ganadora declarada corresponda a modelo realmente guardado
        for k,v in CONFIGS['A_GANADORA_50'].items():
            if params_old.get(k)!=v:
                raise ValueError(f'Fold {fold}: parametro ganador {k}: guardado={params_old.get(k)} esperado={v}')
        seed=params_old.get('random_state')
        print(f'\nFold {fold}: semilla del modelo original = {seed}')
        for name,params in CONFIGS.items():
            clf=ExtraTreesClassifier(**params,random_state=seed,n_jobs=-1)
            clf.fit(Xdev_scaled,ydev)
            pred=clf.predict(Xtest_scaled).astype(int)
            original=old.set_index('sample_index').loc[d['idx_test'],'y_pred'].to_numpy()
            differing=int(np.sum(pred!=original))
            errors=int(np.sum(pred!=d['y_test']))
            print(f'  {name}: errores={errors}, diferencias contra ExtraTrees original={differing}')
            fold_summary.append(dict(config=name,fold=fold,n=len(pred),errors=errors,
                                     different_from_original=differing,accuracy=accuracy_score(d['y_test'],pred),
                                     macro_f1=f1_score(d['y_test'],pred,average='macro',zero_division=0)))
            all_predictions.append(pd.DataFrame(dict(config=name,fold=fold,sample_index=d['idx_test'],
                                                     y_true=d['y_test'],y_pred=pred,
                                                     y_pred_original=original,error=pred!=d['y_test'],
                                                     diferente_original=pred!=original)))
    preds=pd.concat(all_predictions,ignore_index=True)
    preds.to_csv(OUT/'PREDICCIONES_3_CONFIGS_5_FOLDS.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(fold_summary).to_csv(OUT/'RESUMEN_POR_FOLD.csv',index=False,encoding='utf-8-sig')
    results=[]
    for name,g in preds.groupby('config',sort=False):
        g=g.sort_values('sample_index')
        if len(g)!=3840 or g.sample_index.nunique()!=3840:raise ValueError(f'OOF incompleto: {name}')
        results.append(dict(config=name,accuracy=accuracy_score(g.y_true,g.y_pred),
                            macro_f1=f1_score(g.y_true,g.y_pred,average='macro',zero_division=0),
                            MCC=matthews_corrcoef(g.y_true,g.y_pred),errors=int(g.error.sum()),
                            differing_from_original=int(g.diferente_original.sum())))
        pd.DataFrame(confusion_matrix(g.y_true,g.y_pred,labels=[0,1,2,3]),
                     index=['Real_Healthy','Real_6Nm','Real_9Nm','Real_NoBolt'],
                     columns=['Pred_Healthy','Pred_6Nm','Pred_9Nm','Pred_NoBolt']).to_csv(OUT/f'MATRIZ_{name}.csv')
        g.loc[g.error | g.diferente_original].to_csv(OUT/f'ERRORES_Y_CAMBIOS_{name}.csv',index=False,encoding='utf-8-sig')
    summary=pd.DataFrame(results).sort_values('macro_f1',ascending=False)
    summary.to_csv(OUT/'RESUMEN_GLOBAL.csv',index=False,encoding='utf-8-sig')
    (OUT/'CONFIGURACIONES.json').write_text(json.dumps(CONFIGS,indent=2),encoding='utf-8')
    print('\n'+('='*85))
    print('RESULTADOS GLOBALES')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nTodos los archivos originales se conservaron. Salida:',OUT)
    print('Una mejora respecto al OOF original NO reemplaza una seleccion nested independiente.')

if __name__=='__main__':main()
