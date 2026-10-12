# -*- coding: utf-8 -*-
"""Summarize latent-space separability across folds."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ROOT = Path(os.environ.get("WILEY_PROJECT_DIR", Path(__file__).resolve().parents[1]))
OUT = ROOT / '11_DISPERSION_SEPARABILIDAD_WILEY'
PCA_DIR = ROOT / 'PCA_32_sensitivity_FROM_VERIFIED_PUMAP_SPLITS' / 'PCA_full_nowhiten_32' / 'artifacts'
PUMAP_DIR = ROOT / 'PUMAP_Robust_md005_tw075' / 'artifacts'
NAMES = ['Healthy', '6Nm', '9Nm', 'NoBolt']
EPS = 1e-12

def load_fold(rep, fold):
    if rep == 'PCA':
        p = PCA_DIR / f'fold{fold}_PCA_full_nowhiten_32_embeddings.npz'
    else:
        p = PUMAP_DIR / f'fold{fold}_PUMAP_Robust_md005_tw075_embeddings.npz'
    if not p.is_file():
        raise FileNotFoundError(f'No se encuentra: {p}')
    with np.load(p, allow_pickle=False) as f:
        ztr = np.asarray(f['Z_train'], dtype=float)
        zte = np.asarray(f['Z_test'], dtype=float)
        ytr = np.asarray(f['y_train'], dtype=int).ravel()
        yte = np.asarray(f['y_test'], dtype=int).ravel()
        k = 'sample_index_test' if 'sample_index_test' in f.files else 'idx_test'
        idx = np.asarray(f[k], dtype=int).ravel()
        groups = np.asarray(f['groups_test'], dtype=int).ravel() if rep == 'PCA' else None
    if ztr.ndim != 2 or zte.ndim != 2 or ztr.shape[1] != 32 or zte.shape[1] != 32:
        raise ValueError(f'Dimensionalidad incorrecta: {p}')
    if len(ytr)!=len(ztr) or len(yte)!=len(zte) or len(idx)!=len(zte):
        raise ValueError(f'Longitudes incorrectas: {p}')
    if not (np.isfinite(ztr).all() and np.isfinite(zte).all()):
        raise ValueError(f'Hay valores no finitos en: {p}')
    if set(ytr.tolist()) != {0,1,2,3} or set(yte.tolist()) != {0,1,2,3}:
        raise ValueError(f'Clases inesperadas: {p}')
    return dict(ztr=ztr, zte=zte, ytr=ytr, yte=yte, idx=idx, groups=groups)

def build_metadata():
    items=[]
    for fold in range(1,6):
        d=load_fold('PCA',fold)
        if d['groups'] is None: raise ValueError('PCA sin groups_test')
        items.extend(zip(d['idx'], d['groups'], d['yte'], np.full(len(d['idx']),fold)))
    m=pd.DataFrame(items,columns=['sample_index','group_original','y_true','fold'])
    if len(m)!=3840 or not m.sample_index.is_unique or m.group_original.nunique()!=80:
        raise ValueError('Metadatos PCA no satisfacen N=3840, 80 grupos, indices unicos')
    if m.groupby('group_original').y_true.nunique().max()!=1 or m.groupby('group_original').fold.nunique().max()!=1:
        raise ValueError('Incoherencia de etiquetas/grupos/folds')
    return m

def geometry(data, rep, fold, metadata):
    ztr,zte,ytr,yte,idx=(data[k] for k in ('ztr','zte','ytr','yte','idx'))
    aligned = pd.DataFrame({'sample_index':idx,'y':yte}).merge(metadata,on='sample_index',how='left',validate='one_to_one')
    if aligned.group_original.isna().any() or not np.array_equal(aligned.y.to_numpy(),aligned.y_true.to_numpy()) or not (aligned.fold==fold).all():
        raise ValueError(f'No coinciden indices, etiquetas o fold en {rep} {fold}')
    groups=aligned.group_original.astype(int).to_numpy()
    cent=np.stack([ztr[ytr==c].mean(axis=0) for c in range(4)])
    rtrain=[np.linalg.norm(ztr[ytr==c]-cent[c],axis=1) for c in range(4)]
    meanr=np.array([r.mean() for r in rtrain]); medianr=np.array([np.median(r) for r in rtrain])
    dt=np.linalg.norm(zte[:,None,:]-cent[None,:,:],axis=2)
    own=dt[np.arange(len(yte)),yte]
    other=dt.copy(); other[np.arange(len(yte)),yte]=np.inf
    nearest_other=other.min(axis=1)
    other_label=other.argmin(axis=1)
    margin=(nearest_other-own)/(nearest_other+own+EPS) # [-1,+1], escala invariante
    ratio=own/np.maximum(nearest_other,EPS)
    nearest=dt.argmin(axis=1)
    rows=pd.DataFrame({'representation':rep,'fold':fold,'sample_index':idx,'group_original':groups,
        'class_id':yte,'class_name':[NAMES[c] for c in yte],
        'own_centroid_distance':own,'nearest_other_centroid_distance':nearest_other,
        'nearest_other_class':[NAMES[c] for c in other_label],
        'relative_margin':margin,'own_over_other':ratio,
        'own_over_train_class_radius':own/np.maximum(meanr[yte],EPS),
        'nearest_centroid_correct':nearest==yte})
    sep=[]; cls=[]
    for c in range(4):
        zz=zte[yte==c]; dr=own[yte==c]
        rad=rtrain[c]
        cls.append({'representation':rep,'fold':fold,'class_name':NAMES[c],
            'n_train':int((ytr==c).sum()),'n_test':len(zz),
            'train_radius_mean':rad.mean(),'train_radius_std':rad.std(ddof=1),
            'train_radius_p95':np.quantile(rad,.95),'test_distance_mean':dr.mean(),
            'test_distance_std':dr.std(ddof=1),'test_distance_median':np.median(dr),
            'test_distance_p95':np.quantile(dr,.95),
            'test_margin_mean':margin[yte==c].mean(),
            'test_margin_median':np.median(margin[yte==c]),
            'test_margin_negative_pct':100*(margin[yte==c]<0).mean(),
            'nearest_centroid_accuracy_pct':100*(nearest[yte==c]==c).mean()})
        for j in range(c+1,4):
            dd=np.linalg.norm(cent[c]-cent[j])
            sep.append({'representation':rep,'fold':fold,'class_a':NAMES[c],'class_b':NAMES[j],
                'distance_centroids':dd,
                'distance_over_sum_train_mean_radii':dd/max(meanr[c]+meanr[j],EPS),
                'distance_over_mean_train_radii':dd/max((meanr[c]+meanr[j])/2,EPS),
                'train_radius_a':meanr[c],'train_radius_b':meanr[j]})
    # Indices de clustering: caracterizaciones post-hoc sobre test externo,
    # con etiquetas REALES de test solo para calculo descriptivo.
    diag={'representation':rep,'fold':fold,'n_test':len(yte),
          'silhouette_test_32d':silhouette_score(zte,yte),
          'davies_bouldin_test_32d':davies_bouldin_score(zte,yte),
          'calinski_harabasz_test_32d':calinski_harabasz_score(zte,yte),
          'nearest_centroid_accuracy_pct':100*(nearest==yte).mean(),
          'mean_relative_margin':float(margin.mean()),
          'negative_margin_pct':100*float((margin<0).mean())}
    return rows,cls,sep,diag

def save_plots(separations, focus, diagnostics):
    # Graficas de margenes de grupo 20 frente a otros experimentos 9Nm, fold 2.
    for rep in ('PCA','PUMAP'):
        frame=focus[(focus.representation==rep)&(focus.fold==2)]
        g20=frame[(frame.class_name=='9Nm')&(frame.group_original==20)].relative_margin.to_numpy()
        other=frame[(frame.class_name=='9Nm')&(frame.group_original!=20)].relative_margin.to_numpy()
        if len(g20)>0 and len(other)>0:
            fig,ax=plt.subplots(figsize=(7,4.2))
            ax.boxplot([other,g20],tick_labels=['Otros experimentos 9Nm','Experimento 20'],showfliers=True)
            ax.axhline(0,color='black',ls='--',lw=1)
            ax.set(ylabel='Margen relativo en 32D',title=f'{rep} — Fold 2: proximidad a centroides de entrenamiento')
            ax.grid(axis='y',alpha=.2)
            fig.tight_layout()
            fig.savefig(OUT/f'MARGEN_GRUPO20_{rep}_FOLD2.pdf',bbox_inches='tight')
            fig.savefig(OUT/f'MARGEN_GRUPO20_{rep}_FOLD2.png',dpi=240,bbox_inches='tight')
            plt.close(fig)
    for rep in ('PCA','PUMAP'):
        s=separations[(separations.representation==rep)&(separations.fold==2)]
        mat=np.zeros((4,4))
        for _,r in s.iterrows():
            a=NAMES.index(r.class_a); b=NAMES.index(r.class_b)
            mat[a,b]=mat[b,a]=r.distance_over_sum_train_mean_radii
        fig,ax=plt.subplots(figsize=(6.2,5.3))
        im=ax.imshow(mat,cmap='Blues',vmin=0)
        ax.set_xticks(range(4),NAMES);ax.set_yticks(range(4),NAMES)
        for i in range(4):
            for j in range(4):
                ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',weight='bold',
                        color='white' if mat[i,j]>(mat.max()*.6) else 'black',fontsize=11)
        ax.set_title(f'{rep} fold 2 — distancia / suma de radios train')
        fig.colorbar(im,ax=ax,shrink=.77)
        fig.tight_layout()
        fig.savefig(OUT/f'MATRIZ_SEPARACION_RELATIVA_{rep}_FOLD2.pdf',bbox_inches='tight')
        fig.savefig(OUT/f'MATRIZ_SEPARACION_RELATIVA_{rep}_FOLD2.png',dpi=240,bbox_inches='tight')
        plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=build_metadata()
    allrows=[]; allcls=[]; allsep=[]; alld=[]
    for rep in ('PCA','PUMAP'):
        for fold in range(1,6):
            rows,cls,sep,diag=geometry(load_fold(rep,fold),rep,fold,meta)
            allrows.append(rows);allcls.extend(cls);allsep.extend(sep);alld.append(diag)
            print(f'[OK] {rep:5s} fold {fold}: silhouette={diag["silhouette_test_32d"]:.4f}; '
                  f'margen negativo={diag["negative_margin_pct"]:.2f}%; '
                  f'centroide correcto={diag["nearest_centroid_accuracy_pct"]:.2f}%')
    rows=pd.concat(allrows,ignore_index=True)
    cl=pd.DataFrame(allcls);sep=pd.DataFrame(allsep);diag=pd.DataFrame(alld)
    for rep in ('PCA','PUMAP'):
        rr=rows[rows.representation==rep]
        if len(rr)!=3840 or not rr.sample_index.is_unique:
            raise ValueError(f'Validacion OOF incorrecta para {rep}')
    rows.to_csv(OUT/'MARGENES_EXTERNOS_POR_MUESTRA_32D.csv',index=False)
    cl.to_csv(OUT/'DISPERSION_INTRACLASE_POR_FOLD.csv',index=False)
    sep.to_csv(OUT/'DISTANCIAS_ENTRE_CENTROIDES_RELATIVAS_POR_FOLD.csv',index=False)
    diag.to_csv(OUT/'INDICES_SEPARABILIDAD_TEST_POR_FOLD.csv',index=False)
    group=rows.groupby(['representation','fold','group_original','class_name'],as_index=False).agg(
        n=('sample_index','size'),mean_margin=('relative_margin','mean'),
        negative_margin_pct=('relative_margin',lambda x:100*(x<0).mean()),
        nearest_centroid_accuracy_pct=('nearest_centroid_correct',lambda x:100*x.mean()),
        mean_own=('own_centroid_distance','mean'),mean_nearest_other=('nearest_other_centroid_distance','mean'))
    group.to_csv(OUT/'GEOMETRIA_POR_EXPERIMENTO.csv',index=False)
    focus=group[group.group_original.isin([20,60])]
    focus.to_csv(OUT/'DETALLE_GRUPOS_20_60.csv',index=False)
    summary=diag.groupby('representation',as_index=False).agg(
        silhouette_mean=('silhouette_test_32d','mean'),silhouette_std=('silhouette_test_32d','std'),
        db_mean=('davies_bouldin_test_32d','mean'),db_std=('davies_bouldin_test_32d','std'),
        ch_mean=('calinski_harabasz_test_32d','mean'),ch_std=('calinski_harabasz_test_32d','std'),
        margin_mean=('mean_relative_margin','mean'),negative_margin_mean_pct=('negative_margin_pct','mean'))
    summary.to_csv(OUT/'RESUMEN_SEPARABILIDAD_5_FOLDS.csv',index=False)
    save_plots(sep,rows,diag)
    print('\n=== INDICES DE SEPARABILIDAD (por fold, TEST EXTERNO, 32D) ===')
    print(diag[['representation','fold','silhouette_test_32d','davies_bouldin_test_32d','mean_relative_margin','negative_margin_pct']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\n=== GRUPOS 20 y 60 ===')
    print(focus[['representation','fold','group_original','class_name','n','mean_margin','negative_margin_pct','nearest_centroid_accuracy_pct']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nReportes guardados en:',OUT)
    print('NOTA: centroides calculados SOLO con train; test utilizado para diagnosticos post-hoc.')
    print('NOTA: no comparar distancias absolutas de PCA y PUMAP sin normalizacion.')
    print('NOTA: puntuaciones silhouette/DB/CH de test son descriptivas, no sustituyen validacion predictiva.')

if __name__=='__main__':
    main()
