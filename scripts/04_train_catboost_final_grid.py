# -*- coding: utf-8 -*-
"""Evaluate the final CatBoost hyperparameter grid."""

"""
04_RUN_REMAINING_CLASSIFIERS_RF_XGB_CATBOOST_ON_BEST_PCA_AND_PUMAP.py

Corre los tres clasificadores faltantes:

    1. Random Forest
    2. XGBoost
    3. CatBoost

Sobre solo dos representaciones:

    1. Parametric UMAP:
       PUMAP_Robust_md005_tw075

    2. PCA baseline:
       PCA_full_nowhiten_32

Protocolo:
    - Grid search interno usando Z_train -> Z_val.
    - Selección por macro_f1 interno.
    - Refit final con Z_train + Z_val.
    - Evaluación externa solo en Z_test.
    - Consolidación OOF externa con 3840 muestras únicas.
    - No recalcula PCA.
    - No recalcula PUMAP.
    - No crea folds nuevos.

Diseñado para no saturar el PC:
    - Ejecución secuencial.
    - N_JOBS controlado.
    - XGBoost y CatBoost usan grillas limitadas pero representativas.
    - Permite retomar si el PC se apaga.
"""

import os

# ============================================================
# CONTROL DE HILOS
# ============================================================

N_JOBS = 3

os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

from pathlib import Path
import json
import time
import random
import pickle
import itertools
import warnings
import gc

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as exc:
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = repr(exc)

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception as exc:
    CATBOOST_AVAILABLE = False
    CATBOOST_IMPORT_ERROR = repr(exc)

warnings.filterwarnings("ignore")


# ============================================================
# RUTAS
# ============================================================

PROJECT_DIR = Path(os.environ.get("WILEY_PROJECT_DIR", Path(__file__).resolve().parents[1]))

PUMAP_ARTIFACT_DIR = PROJECT_DIR / "PUMAP_Robust_md005_tw075" / "artifacts"

PCA_ARTIFACT_DIR = (
    PROJECT_DIR
    / "PCA_32_sensitivity_FROM_VERIFIED_PUMAP_SPLITS"
    / "PCA_full_nowhiten_32"
    / "artifacts"
)

OUT_DIR = PROJECT_DIR / "06_CATBOOST_24_CONFIGURACIONES_PCA_PUMAP"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TOTAL_TEST = 3840
EXPECTED_CLASSES = [0, 1, 2, 3]

GLOBAL_SEED = 42

SKIP_COMPLETED = True  # Salta solamente resultados existentes de ESTA carpeta nueva

# Random Forest completo no debería ser demasiado pesado.
MAX_CONFIGS_RANDOM_FOREST = None

# XGBoost original completo puede ser muy largo.
# Se toma una muestra representativa y reproducible de la grilla completa.
MAX_CONFIGS_XGBOOST = 160

# CatBoost también se limita para no saturar.
MAX_CONFIGS_CATBOOST = 24

CLASS_NAMES = {
    0: "Healthy",
    1: "6Nm",
    2: "9Nm",
    3: "NoBolt",
}


# ============================================================
# REPRESENTACIONES
# ============================================================

REPRESENTATIONS = [
    {
        "representation_family": "PUMAP",
        "representation": "PUMAP_Robust_md005_tw075",
        "display_name": "Parametric UMAP md005 tw075",
        "artifact_dir": PUMAP_ARTIFACT_DIR,
        "file_template": "fold{fold}_PUMAP_Robust_md005_tw075_embeddings.npz",
    },
    {
        "representation_family": "PCA",
        "representation": "PCA_full_nowhiten_32",
        "display_name": "PCA full no-whiten 32",
        "artifact_dir": PCA_ARTIFACT_DIR,
        "file_template": "fold{fold}_PCA_full_nowhiten_32_embeddings.npz",
    },
]


# ============================================================
# UTILIDADES
# ============================================================

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def product_grid(grid_dict):
    keys = list(grid_dict.keys())
    values = [grid_dict[k] for k in keys]

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def limit_grid(configs, max_configs, seed=GLOBAL_SEED):
    """
    Limita una grilla grande sin tomar simplemente los primeros elementos.
    Selecciona configuraciones repartidas de forma reproducible.
    """
    configs = list(configs)

    if max_configs is None:
        return configs

    if len(configs) <= max_configs:
        return configs

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(configs), size=max_configs, replace=False)
    idx = np.sort(idx)

    return [configs[i] for i in idx]


def pick_key(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None


def get_required_array(npz, candidates, file_path):
    key = pick_key(npz, candidates)

    if key is None:
        raise KeyError(
            f"No encontré ninguna key de {candidates} en:\n{file_path}\n"
            f"Keys disponibles: {npz.files}"
        )

    return np.asarray(npz[key]), key


def ensure_1d_labels(pred, expected_size, context):
    """Normalize classifier label predictions to one dimension.

    Some classifier APIs (notably CatBoost) return shape (n, 1).
    Unexpected multi-column predictions are rejected rather than
    silently interpreting probability matrices as class labels.
    """
    arr = np.asarray(pred)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(
            f"{context}: expected 1D class labels, received shape {arr.shape}."
        )
    if len(arr) != expected_size:
        raise ValueError(
            f"{context}: expected {expected_size} predictions, received {len(arr)}."
        )
    return arr.astype(int)


def metric_dict(y_true, y_pred):
    labels = EXPECTED_CLASSES

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                labels=labels,
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def summarize_mean_std(df, group_cols, metric_cols):
    rows = []

    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: val for col, val in zip(group_cols, keys)}

        for m in metric_cols:
            row[m] = f"{sub[m].mean():.4f} ± {sub[m].std(ddof=1):.4f}"

        rows.append(row)

    return pd.DataFrame(rows)


def load_embedding_fold(rep, fold: int):
    path = Path(rep["artifact_dir"]) / rep["file_template"].format(fold=fold)

    if not path.exists():
        raise FileNotFoundError(f"No existe archivo de embeddings:\n{path}")

    d = np.load(path, allow_pickle=True)

    Z_train, key_Z_train = get_required_array(d, ["Z_train", "z_train"], path)
    Z_val, key_Z_val = get_required_array(d, ["Z_val", "z_val"], path)
    Z_test, key_Z_test = get_required_array(d, ["Z_test", "z_test"], path)

    y_train, key_y_train = get_required_array(d, ["y_train", "y_tr2"], path)
    y_val, key_y_val = get_required_array(d, ["y_val", "y_va"], path)
    y_test, key_y_test = get_required_array(d, ["y_test", "yte"], path)

    idx_test, key_idx_test = get_required_array(
        d,
        ["sample_index_test", "idx_test", "test_idx"],
        path,
    )

    groups_test = None
    groups_test_key = None

    possible_group_key = pick_key(d, ["groups_test", "test_groups", "gte"])
    if possible_group_key is not None:
        groups_test = np.asarray(d[possible_group_key]).reshape(-1).astype(int)
        groups_test_key = possible_group_key

    data = {
        "path": path,
        "Z_train": np.asarray(Z_train, dtype=np.float32),
        "Z_val": np.asarray(Z_val, dtype=np.float32),
        "Z_test": np.asarray(Z_test, dtype=np.float32),
        "y_train": np.asarray(y_train).reshape(-1).astype(int),
        "y_val": np.asarray(y_val).reshape(-1).astype(int),
        "y_test": np.asarray(y_test).reshape(-1).astype(int),
        "idx_test": np.asarray(idx_test).reshape(-1).astype(int),
        "groups_test": groups_test,
        "keys": {
            "Z_train": key_Z_train,
            "Z_val": key_Z_val,
            "Z_test": key_Z_test,
            "y_train": key_y_train,
            "y_val": key_y_val,
            "y_test": key_y_test,
            "idx_test": key_idx_test,
            "groups_test": groups_test_key,
        },
    }

    if data["Z_train"].shape[1] != 32:
        raise RuntimeError(f"{path.name}: Z_train no tiene 32 dimensiones: {data['Z_train'].shape}")

    if data["Z_val"].shape[1] != 32:
        raise RuntimeError(f"{path.name}: Z_val no tiene 32 dimensiones: {data['Z_val'].shape}")

    if data["Z_test"].shape[1] != 32:
        raise RuntimeError(f"{path.name}: Z_test no tiene 32 dimensiones: {data['Z_test'].shape}")

    if len(data["Z_train"]) != len(data["y_train"]):
        raise RuntimeError(f"{path.name}: Z_train/y_train no coinciden.")

    if len(data["Z_val"]) != len(data["y_val"]):
        raise RuntimeError(f"{path.name}: Z_val/y_val no coinciden.")

    if len(data["Z_test"]) != len(data["y_test"]):
        raise RuntimeError(f"{path.name}: Z_test/y_test no coinciden.")

    if len(data["idx_test"]) != len(data["y_test"]):
        raise RuntimeError(f"{path.name}: idx_test/y_test no coinciden.")

    return data


# ============================================================
# GRILLAS
# ============================================================

def get_random_forest_grid():
    grid = {
        "n_estimators": [50, 100, 200, 500, 650, 800],
        "criterion": ["gini"],
        "max_features": [0.5, 0.7, "sqrt"],
        "max_depth": [None, 20, 30, 50],
        "min_samples_split": [2],
        "min_samples_leaf": [1, 2],
        "bootstrap": [True],
        "class_weight": ["balanced_subsample"],
    }

    configs = list(product_grid(grid))
    configs = limit_grid(configs, MAX_CONFIGS_RANDOM_FOREST, seed=GLOBAL_SEED + 10)

    return configs


def get_xgboost_grid():
    grid = {
        "n_estimators": [50, 100, 200, 500, 650],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_depth": [4, 6, 8, 10],
        "min_child_weight": [1, 3],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    }

    configs = list(product_grid(grid))
    configs = limit_grid(configs, MAX_CONFIGS_XGBOOST, seed=GLOBAL_SEED + 20)

    return configs


def get_catboost_grid():
    """24 combinaciones predefinidas, identicas en los cinco folds y ambas representaciones.

    3 profundidades x 2 iteraciones x 2 tasas de aprendizaje x 2 valores de L2 = 24.
    Se fija border_count para limitar el tiempo y mantener una exploracion controlada.
    """
    grid = {
        "depth": [4, 6, 8],
        "iterations": [200, 500],
        "learning_rate": [0.03, 0.08],
        "l2_leaf_reg": [3.0, 5.0],
        "border_count": [128],
    }
    configs = list(product_grid(grid))
    assert len(configs) == 24
    return configs


def build_random_forest(params, seed):
    return RandomForestClassifier(
        **params,
        random_state=seed,
        n_jobs=N_JOBS,
    )


def build_xgboost(params, seed):
    return XGBClassifier(
        **params,
        objective="multi:softprob",
        num_class=4,
        tree_method="hist",
        random_state=seed,
        n_jobs=N_JOBS,
        eval_metric="mlogloss",
        verbosity=0,
    )


def build_catboost(params, seed):
    return CatBoostClassifier(
        **params,
        loss_function="MultiClass",
        random_seed=seed,
        thread_count=N_JOBS,
        verbose=False,
        allow_writing_files=False,
    )


def get_classifier_grid_and_builder(classifier_name):
    if classifier_name == "RandomForest":
        return get_random_forest_grid(), build_random_forest

    if classifier_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ImportError(f"XGBoost no está disponible: {XGBOOST_IMPORT_ERROR}")
        return get_xgboost_grid(), build_xgboost

    if classifier_name == "CatBoost":
        if not CATBOOST_AVAILABLE:
            raise ImportError(f"CatBoost no está disponible: {CATBOOST_IMPORT_ERROR}")
        return get_catboost_grid(), build_catboost

    raise ValueError(f"Clasificador no reconocido: {classifier_name}")


# ============================================================
# ENTRENAMIENTO
# ============================================================

def run_representation_classifier(rep, classifier_name):
    rep_name = rep["representation"]

    out_pair_dir = OUT_DIR / rep_name / classifier_name
    fold_dir = out_pair_dir / "fold_results"
    model_dir = out_pair_dir / "models"

    fold_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_pair_dir / "external_oof_metrics.csv"

    if SKIP_COMPLETED and summary_path.exists():
        print(f"\n[SKIP] Ya existe consolidado: {summary_path}")
        return pd.read_csv(summary_path).iloc[0].to_dict()

    print("\n" + "=" * 100)
    print(f"{classifier_name} sobre {rep_name}")
    print("=" * 100)

    configs, builder = get_classifier_grid_and_builder(classifier_name)

    print("Configuraciones:", len(configs))
    print("Artefactos:", rep["artifact_dir"])

    selected_rows = []
    external_rows = []
    internal_rows_all = []
    pred_rows = []

    t_pair = time.time()

    for fold in range(1, 6):
        print("\n" + "-" * 100)
        print(f"{classifier_name} | {rep_name} | Fold {fold}")
        print("-" * 100)

        selected_path = fold_dir / f"fold{fold}_selected_internal.csv"
        external_path = fold_dir / f"fold{fold}_external_metrics.csv"
        pred_path = fold_dir / f"fold{fold}_external_predictions.csv"
        grid_path = fold_dir / f"fold{fold}_internal_grid.csv"

        if (
            SKIP_COMPLETED
            and selected_path.exists()
            and external_path.exists()
            and pred_path.exists()
            and grid_path.exists()
        ):
            print(f"[SKIP] Fold {fold} ya completo.")

            selected_rows.append(pd.read_csv(selected_path).iloc[0].to_dict())
            external_rows.append(pd.read_csv(external_path).iloc[0].to_dict())
            internal_rows_all.append(pd.read_csv(grid_path))
            pred_rows.append(pd.read_csv(pred_path))
            continue

        t_fold = time.time()

        data = load_embedding_fold(rep, fold)

        Z_train = data["Z_train"]
        y_train = data["y_train"]

        Z_val = data["Z_val"]
        y_val = data["y_val"]

        Z_test = data["Z_test"]
        y_test = data["y_test"]

        idx_test = data["idx_test"]
        groups_test = data["groups_test"]

        print("  Z_train:", Z_train.shape)
        print("  Z_val:  ", Z_val.shape)
        print("  Z_test: ", Z_test.shape)
        print("  y_val clases:", {CLASS_NAMES[c]: int(np.sum(y_val == c)) for c in EXPECTED_CLASSES})

        internal_rows = []

        best_score = -np.inf
        best_config = None
        best_metrics = None
        best_config_index = None

        for config_index, params in enumerate(configs, start=1):
            seed = GLOBAL_SEED + fold * 100 + config_index

            try:
                scaler = StandardScaler()
                Z_train_sc = scaler.fit_transform(Z_train)
                Z_val_sc = scaler.transform(Z_val)

                clf = builder(params, seed)
                clf.fit(Z_train_sc, y_train)

                y_val_pred = ensure_1d_labels(
                    clf.predict(Z_val_sc), len(y_val),
                    f"{classifier_name} / {rep_name} / fold {fold} / validation"
                )
                metrics = metric_dict(y_val, y_val_pred)

                row = {
                    "representation_family": rep["representation_family"],
                    "representation": rep_name,
                    "representation_display": rep["display_name"],
                    "classifier": classifier_name,
                    "fold": fold,
                    "config_index": config_index,
                    "params_json": json.dumps(params, sort_keys=True),
                    **metrics,
                }

                internal_rows.append(row)

                score = metrics["macro_f1"]

                if score > best_score + 1e-12:
                    best_score = score
                    best_config = params
                    best_metrics = metrics
                    best_config_index = config_index

                del clf
                gc.collect()

            except Exception as e:
                row = {
                    "representation_family": rep["representation_family"],
                    "representation": rep_name,
                    "representation_display": rep["display_name"],
                    "classifier": classifier_name,
                    "fold": fold,
                    "config_index": config_index,
                    "params_json": json.dumps(params, sort_keys=True),
                    "error": repr(e),
                }

                internal_rows.append(row)

            if config_index % 20 == 0:
                print(
                    f"  Config {config_index}/{len(configs)} | "
                    f"best macro_f1={best_score:.6f}"
                )

        if best_config is None:
            raise RuntimeError(f"No se pudo entrenar ninguna configuración en fold {fold}")

        internal_df = pd.DataFrame(internal_rows)
        internal_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
        internal_rows_all.append(internal_df)

        # Refit final con train + val.
        Z_dev = np.vstack([Z_train, Z_val])
        y_dev = np.concatenate([y_train, y_val])

        final_scaler = StandardScaler()
        Z_dev_sc = final_scaler.fit_transform(Z_dev)
        Z_test_sc = final_scaler.transform(Z_test)

        final_seed = GLOBAL_SEED + fold * 10000

        final_clf = builder(best_config, final_seed)
        final_clf.fit(Z_dev_sc, y_dev)

        y_test_pred = ensure_1d_labels(
            final_clf.predict(Z_test_sc), len(y_test),
            f"{classifier_name} / {rep_name} / fold {fold} / external test"
        )

        external_metrics = metric_dict(y_test, y_test_pred)

        selected_row = {
            "representation_family": rep["representation_family"],
            "representation": rep_name,
            "representation_display": rep["display_name"],
            "classifier": classifier_name,
            "fold": fold,
            "best_config_index": best_config_index,
            "selected_params_json": json.dumps(best_config, sort_keys=True),
            **best_metrics,
        }

        external_row = {
            "representation_family": rep["representation_family"],
            "representation": rep_name,
            "representation_display": rep["display_name"],
            "classifier": classifier_name,
            "fold": fold,
            "selected_params_json": json.dumps(best_config, sort_keys=True),
            **external_metrics,
        }

        pred_df = pd.DataFrame(
            {
                "representation_family": rep["representation_family"],
                "representation": rep_name,
                "classifier": classifier_name,
                "fold": fold,
                "sample_index": idx_test,
                "group": [
                    int(groups_test[i]) if groups_test is not None and len(groups_test) == len(y_test) else -1
                    for i in range(len(y_test))
                ],
                "y_true": y_test,
                "y_pred": y_test_pred,
            }
        )

        runtime = time.time() - t_fold

        selected_row["runtime_seconds"] = runtime
        external_row["runtime_seconds"] = runtime

        pd.DataFrame([selected_row]).to_csv(selected_path, index=False, encoding="utf-8-sig")
        pd.DataFrame([external_row]).to_csv(external_path, index=False, encoding="utf-8-sig")
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

        safe_name = f"fold{fold}_{rep_name}_{classifier_name}".replace(" ", "_")

        save_pickle(final_scaler, model_dir / f"{safe_name}_scaler.pkl")
        save_pickle(final_clf, model_dir / f"{safe_name}_classifier.pkl")

        selected_rows.append(selected_row)
        external_rows.append(external_row)
        pred_rows.append(pred_df)

        print("\nFold terminado:")
        print("  best_config_index:", best_config_index)
        print("  selected_params:", json.dumps(best_config, sort_keys=True))
        print("  internal macro_f1:", f"{selected_row['macro_f1']:.6f}")
        print("  external macro_f1:", f"{external_row['macro_f1']:.6f}")
        print("  external mcc:", f"{external_row['mcc']:.6f}")
        print("  runtime:", f"{runtime:.2f} s")

        del final_clf
        gc.collect()

    # ========================================================
    # CONSOLIDAR
    # ========================================================

    internal_all_df = pd.concat(internal_rows_all, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    external_df = pd.DataFrame(external_rows)
    pred_df = pd.concat(pred_rows, ignore_index=True)

    internal_all_df.to_csv(
        out_pair_dir / "all_internal_grid_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_df.to_csv(
        out_pair_dir / "selected_internal_metrics_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    external_df.to_csv(
        out_pair_dir / "external_metrics_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_df.to_csv(
        out_pair_dir / "external_oof_predictions_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    internal_summary = summarize_mean_std(
        selected_df,
        group_cols=["representation_family", "representation", "classifier"],
        metric_cols=[
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "mcc",
        ],
    )

    internal_summary.to_csv(
        out_pair_dir / "selected_internal_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    y_true = pred_df["y_true"].to_numpy().astype(int)
    y_pred = pred_df["y_pred"].to_numpy().astype(int)

    oof_metrics = metric_dict(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=EXPECTED_CLASSES)

    cm_df = pd.DataFrame(
        cm,
        index=[CLASS_NAMES[c] for c in EXPECTED_CLASSES],
        columns=[CLASS_NAMES[c] for c in EXPECTED_CLASSES],
    )

    n_rows = len(pred_df)
    n_unique = pred_df["sample_index"].nunique()
    n_duplicates = int(pred_df["sample_index"].duplicated().sum())
    cm_sum = int(cm.sum())

    integrity = {
        "n_rows_oof": int(n_rows),
        "n_unique_sample_index": int(n_unique),
        "n_duplicates": int(n_duplicates),
        "confusion_sum": int(cm_sum),
        "ok_3840_unique": bool(
            n_rows == EXPECTED_TOTAL_TEST
            and n_unique == EXPECTED_TOTAL_TEST
            and n_duplicates == 0
            and cm_sum == EXPECTED_TOTAL_TEST
        ),
    }

    oof_row = {
        "representation_family": rep["representation_family"],
        "representation": rep_name,
        "representation_display": rep["display_name"],
        "classifier": classifier_name,
        **oof_metrics,
        **integrity,
        "runtime_seconds": float(time.time() - t_pair),
    }

    oof_df = pd.DataFrame([oof_row])

    oof_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    save_json(oof_row, out_pair_dir / "external_oof_metrics.json")
    save_json(integrity, out_pair_dir / "integrity_check.json")

    print("\n" + "=" * 100)
    print(f"CONSOLIDADO | {classifier_name} | {rep_name}")
    print("=" * 100)

    print("\nResumen interno seleccionado:")
    print(internal_summary.to_string(index=False))

    print("\nOOF externo:")
    print(oof_df.to_string(index=False))

    print("\nMatriz de confusión:")
    print(cm_df.to_string())

    if not integrity["ok_3840_unique"]:
        raise RuntimeError(
            f"Falla integridad OOF para {rep_name} + {classifier_name}: {integrity}"
        )

    return oof_row


# ============================================================
# MAIN
# ============================================================

def main():
    set_all_seeds(GLOBAL_SEED)

    print("=" * 100)
    print("CATBOOST - 12 CONFIGURACIONES - PCA Y PARAMETRIC UMAP")
    print("=" * 100)

    print("PROJECT_DIR:", PROJECT_DIR)
    print("PUMAP_ARTIFACT_DIR:", PUMAP_ARTIFACT_DIR)
    print("PCA_ARTIFACT_DIR:", PCA_ARTIFACT_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("N_JOBS:", N_JOBS)
    print("MAX_CONFIGS_RANDOM_FOREST:", MAX_CONFIGS_RANDOM_FOREST)
    print("MAX_CONFIGS_XGBOOST:", MAX_CONFIGS_XGBOOST)
    print("MAX_CONFIGS_CATBOOST:", MAX_CONFIGS_CATBOOST)
    print("XGBOOST_AVAILABLE:", XGBOOST_AVAILABLE)
    print("CATBOOST_AVAILABLE:", CATBOOST_AVAILABLE)

    if not PUMAP_ARTIFACT_DIR.exists():
        raise FileNotFoundError(f"No existe PUMAP_ARTIFACT_DIR:\n{PUMAP_ARTIFACT_DIR}")

    if not PCA_ARTIFACT_DIR.exists():
        raise FileNotFoundError(f"No existe PCA_ARTIFACT_DIR:\n{PCA_ARTIFACT_DIR}")

    if not CATBOOST_AVAILABLE:
        raise ImportError(f"CatBoost no está disponible: {CATBOOST_IMPORT_ERROR}")
    classifiers = ["CatBoost"]

    config = {
        "PROJECT_DIR": str(PROJECT_DIR),
        "PUMAP_ARTIFACT_DIR": str(PUMAP_ARTIFACT_DIR),
        "PCA_ARTIFACT_DIR": str(PCA_ARTIFACT_DIR),
        "OUT_DIR": str(OUT_DIR),
        "GLOBAL_SEED": GLOBAL_SEED,
        "N_JOBS": N_JOBS,
        "MAX_CONFIGS_RANDOM_FOREST": MAX_CONFIGS_RANDOM_FOREST,
        "MAX_CONFIGS_XGBOOST": MAX_CONFIGS_XGBOOST,
        "MAX_CONFIGS_CATBOOST": MAX_CONFIGS_CATBOOST,
        "grid_design": "12 explicit configurations: depth 4/6/8 x iterations 200/500 x learning_rate 0.03/0.08; l2_leaf_reg=3 and border_count=128",
        "SKIP_COMPLETED": SKIP_COMPLETED,
        "XGBOOST_AVAILABLE": XGBOOST_AVAILABLE,
        "CATBOOST_AVAILABLE": CATBOOST_AVAILABLE,
        "representations": [
            {
                **r,
                "artifact_dir": str(r["artifact_dir"]),
            }
            for r in REPRESENTATIONS
        ],
        "classifiers": classifiers,
        "protocol": {
            "internal_selection": "Z_train -> Z_val",
            "selection_metric": "macro_f1",
            "final_refit": "Z_train + Z_val",
            "external_evaluation": "Z_test only",
            "no_new_folds": True,
            "no_recompute_embeddings": True,
        },
    }

    save_json(config, OUT_DIR / "config_CATBOOST_12_PCA_PUMAP.json")

    t0 = time.time()

    summaries = []

    for classifier_name in classifiers:
        for rep in REPRESENTATIONS:
            summary = run_representation_classifier(rep, classifier_name)
            summaries.append(summary)

            pd.DataFrame(summaries).to_csv(
                OUT_DIR / "ALL_results_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    total_runtime = time.time() - t0

    summary_df = pd.DataFrame(summaries)

    summary_df = summary_df.sort_values(
        by=["macro_f1", "mcc", "accuracy"],
        ascending=False,
    )

    summary_path = OUT_DIR / "ALL_results_CATBOOST_12_PCA_PUMAP.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    run_summary = {
        "status": "finished",
        "total_runtime_seconds": total_runtime,
        "total_runtime_minutes": total_runtime / 60,
        "out_dir": str(OUT_DIR),
        "summary_path": str(summary_path),
        "classifiers_run": classifiers,
    }

    save_json(run_summary, OUT_DIR / "run_summary_CATBOOST_12.json")

    print("\n" + "=" * 100)
    print("TODO TERMINADO")
    print("=" * 100)

    print("Tiempo total:", f"{total_runtime:.2f} s = {total_runtime / 60:.2f} min")
    print("Resumen final:")
    print(summary_path)

    cols = [
        "representation_family",
        "representation",
        "classifier",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "kappa",
        "confusion_sum",
        "ok_3840_unique",
        "runtime_seconds",
    ]

    print("\nTOP RESULTADOS:")
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
