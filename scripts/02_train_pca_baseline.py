# -*- coding: utf-8 -*-
"""Fit PCA representations using verified group splits."""

import os
from pathlib import Path
import json
import time
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# RUTAS DEFINITIVAS VERIFICADAS
# ============================================================

PROJECT_DIR = Path(os.environ.get("WILEY_PROJECT_DIR", Path(__file__).resolve().parents[1]))

FFT_DATASET_PATH = (PROJECT_DIR / "data" / "A1_windows_fft_W1024_H512_24ch.npz")

PUMAP_ARTIFACT_DIR = PROJECT_DIR / "PUMAP_Robust_md005_tw075" / "artifacts"

OUT_DIR = PROJECT_DIR / "PCA_32_sensitivity_FROM_VERIFIED_PUMAP_SPLITS"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N_FOLDS = 5
EXPECTED_TOTAL_TEST = 3840

GLOBAL_SEED = 42
N_COMPONENTS = 32

CLASS_NAMES = {
    0: "Healthy",
    1: "6Nm",
    2: "9Nm",
    3: "NoBolt",
}

PCA_CONFIGS = [
    {
        "name": "PCA_full_nowhiten_32",
        "n_components": 32,
        "svd_solver": "full",
        "whiten": False,
    },
    {
        "name": "PCA_full_whiten_32",
        "n_components": 32,
        "svd_solver": "full",
        "whiten": True,
    },
    {
        "name": "PCA_randomized_nowhiten_32",
        "n_components": 32,
        "svd_solver": "randomized",
        "whiten": False,
    },
    {
        "name": "PCA_randomized_whiten_32",
        "n_components": 32,
        "svd_solver": "randomized",
        "whiten": True,
    },
]


# ============================================================
# UTILIDADES
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pick_key(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None


def class_counts_dict(y):
    y = np.asarray(y).reshape(-1).astype(int)
    counts = np.bincount(y, minlength=4)
    return {CLASS_NAMES[i]: int(counts[i]) for i in range(4)}


def group_counts_by_class(idx, y_all, groups_all):
    idx = np.asarray(idx).reshape(-1).astype(int)
    y_subset = y_all[idx]
    groups_subset = groups_all[idx]

    row = {
        "n_windows": int(len(idx)),
        "n_groups": int(len(np.unique(groups_subset))),
    }

    for c, name in CLASS_NAMES.items():
        groups_c = []

        for g in np.unique(groups_subset):
            mask_g = groups_subset == g
            y_g = np.unique(y_subset[mask_g])

            if len(y_g) == 1 and int(y_g[0]) == c:
                groups_c.append(g)

        row[f"groups_{name}"] = int(len(groups_c))
        row[f"windows_{name}"] = int(np.sum(y_subset == c))

    return row


def load_fft_dataset():
    if not FFT_DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe FFT_DATASET_PATH:\n{FFT_DATASET_PATH}")

    d = np.load(FFT_DATASET_PATH, allow_pickle=True)

    x_key = pick_key(d, ["X_fft", "X", "features", "X_all"])
    y_key = pick_key(d, ["y", "labels", "y_all"])
    g_key = pick_key(d, ["groups", "group", "g", "groups_all"])

    if x_key is None or y_key is None or g_key is None:
        raise KeyError(
            f"No encontré X/y/groups en el dataset FFT.\n"
            f"Keys disponibles: {d.files}"
        )

    X = np.asarray(d[x_key], dtype=np.float32)
    y = np.asarray(d[y_key]).reshape(-1).astype(int)
    groups = np.asarray(d[g_key]).reshape(-1).astype(int)

    if X.shape[0] != len(y) or X.shape[0] != len(groups):
        raise ValueError(
            f"Dimensiones inconsistentes: X={X.shape}, y={len(y)}, groups={len(groups)}"
        )

    print("\nDataset FFT cargado:")
    print("  archivo:", FFT_DATASET_PATH)
    print("  x_key:", x_key)
    print("  y_key:", y_key)
    print("  g_key:", g_key)
    print("  X:", X.shape)
    print("  y:", y.shape)
    print("  groups:", groups.shape)
    print("  n_groups:", len(np.unique(groups)))
    print("  class_counts:", class_counts_dict(y))

    return X, y, groups, {
        "fft_dataset_path": str(FFT_DATASET_PATH),
        "x_key": x_key,
        "y_key": y_key,
        "groups_key": g_key,
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "groups_shape": list(groups.shape),
        "n_groups": int(len(np.unique(groups))),
        "class_counts": class_counts_dict(y),
    }


def load_verified_pumap_indices(fold):
    path = PUMAP_ARTIFACT_DIR / f"fold{fold}_PUMAP_Robust_md005_tw075_embeddings.npz"

    if not path.exists():
        raise FileNotFoundError(f"No existe artefacto PUMAP fold {fold}:\n{path}")

    d = np.load(path, allow_pickle=True)

    train_key = pick_key(d, ["idx_train", "train_idx", "tr2_abs_idx"])
    val_key = pick_key(d, ["idx_val", "val_idx", "va_abs_idx"])
    test_key = pick_key(d, ["idx_test", "test_idx"])

    if train_key is None or val_key is None or test_key is None:
        raise KeyError(
            f"No encontré índices en {path.name}.\n"
            f"Keys disponibles: {d.files}"
        )

    idx_train = np.asarray(d[train_key]).reshape(-1).astype(int)
    idx_val = np.asarray(d[val_key]).reshape(-1).astype(int)
    idx_test = np.asarray(d[test_key]).reshape(-1).astype(int)

    return idx_train, idx_val, idx_test, {
        "pumap_artifact": str(path),
        "train_key": train_key,
        "val_key": val_key,
        "test_key": test_key,
    }


def validate_split(fold, idx_train, idx_val, idx_test, y_all, groups_all):
    set_train = set(idx_train.tolist())
    set_val = set(idx_val.tolist())
    set_test = set(idx_test.tolist())

    overlap_train_val = len(set_train & set_val)
    overlap_train_test = len(set_train & set_test)
    overlap_val_test = len(set_val & set_test)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError(
            f"Fold {fold}: hay solape entre subconjuntos. "
            f"train-val={overlap_train_val}, train-test={overlap_train_test}, val-test={overlap_val_test}"
        )

    val_comp = group_counts_by_class(idx_val, y_all, groups_all)

    ok_val_12 = val_comp["n_groups"] == 12
    ok_val_3_per_class = (
        val_comp["groups_Healthy"] == 3
        and val_comp["groups_6Nm"] == 3
        and val_comp["groups_9Nm"] == 3
        and val_comp["groups_NoBolt"] == 3
    )

    if not ok_val_12:
        raise RuntimeError(f"Fold {fold}: validación interna no tiene 12 grupos: {val_comp}")

    if not ok_val_3_per_class:
        raise RuntimeError(f"Fold {fold}: validación interna no tiene 3 grupos por clase: {val_comp}")

    return {
        "fold": int(fold),
        "overlap_train_val": int(overlap_train_val),
        "overlap_train_test": int(overlap_train_test),
        "overlap_val_test": int(overlap_val_test),
        "ok_no_overlap": True,
        "validation_composition": val_comp,
    }


def fit_transform_pca_for_config(config, fold, X_all, y_all, groups_all, idx_train, idx_val, idx_test, split_info):
    config_name = config["name"]

    config_dir = OUT_DIR / config_name
    artifacts_dir = config_dir / "artifacts"

    ensure_dir(config_dir)
    ensure_dir(artifacts_dir)

    stem = f"fold{fold}_{config_name}"

    embeddings_path = artifacts_dir / f"{stem}_embeddings.npz"
    pca_path = artifacts_dir / f"{stem}_pca_model.pkl"
    scaler_path = artifacts_dir / f"{stem}_fft_scaler.pkl"
    info_path = artifacts_dir / f"{stem}_info.json"

    if embeddings_path.exists() and pca_path.exists() and scaler_path.exists() and info_path.exists():
        print(f"[SKIP] Ya existe {embeddings_path}")
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)

    t0 = time.time()

    X_train_raw = X_all[idx_train]
    X_val_raw = X_all[idx_val]
    X_test_raw = X_all[idx_test]

    y_train = y_all[idx_train]
    y_val = y_all[idx_val]
    y_test = y_all[idx_test]

    groups_train = groups_all[idx_train]
    groups_val = groups_all[idx_val]
    groups_test = groups_all[idx_test]

    # Escalado FFT ajustado solo con train interno
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val_s = scaler.transform(X_val_raw).astype(np.float32)
    X_test_s = scaler.transform(X_test_raw).astype(np.float32)

    # PCA ajustado solo con train interno
    pca = PCA(
        n_components=config["n_components"],
        svd_solver=config["svd_solver"],
        whiten=config["whiten"],
        random_state=GLOBAL_SEED,
    )

    t_pca = time.time()

    Z_train = pca.fit_transform(X_train_s).astype(np.float32)
    Z_val = pca.transform(X_val_s).astype(np.float32)
    Z_test = pca.transform(X_test_s).astype(np.float32)

    pca_seconds = time.time() - t_pca
    total_seconds = time.time() - t0

    if Z_train.shape[1] != N_COMPONENTS:
        raise RuntimeError(f"{config_name} fold {fold}: Z_train inválido {Z_train.shape}")
    if Z_val.shape[1] != N_COMPONENTS:
        raise RuntimeError(f"{config_name} fold {fold}: Z_val inválido {Z_val.shape}")
    if Z_test.shape[1] != N_COMPONENTS:
        raise RuntimeError(f"{config_name} fold {fold}: Z_test inválido {Z_test.shape}")

    np.savez_compressed(
        embeddings_path,

        Z_train=Z_train,
        Z_val=Z_val,
        Z_test=Z_test,

        y_train=y_train.astype(int),
        y_val=y_val.astype(int),
        y_test=y_test.astype(int),

        groups_train=groups_train.astype(int),
        groups_val=groups_val.astype(int),
        groups_test=groups_test.astype(int),

        sample_index_train=idx_train.astype(int),
        sample_index_val=idx_val.astype(int),
        sample_index_test=idx_test.astype(int),

        idx_train=idx_train.astype(int),
        idx_val=idx_val.astype(int),
        idx_test=idx_test.astype(int),

        train_idx=idx_train.astype(int),
        val_idx=idx_val.astype(int),
        test_idx=idx_test.astype(int),

        fold=np.asarray([fold], dtype=int),
        config_name=np.asarray(config_name, dtype=object),

        n_components=np.asarray([config["n_components"]], dtype=int),
        svd_solver=np.asarray(config["svd_solver"], dtype=object),
        whiten=np.asarray([config["whiten"]], dtype=bool),

        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float64),
        explained_variance_ratio_sum=np.asarray(
            [np.sum(pca.explained_variance_ratio_)],
            dtype=np.float64,
        ),
    )

    save_pickle(pca, pca_path)
    save_pickle(scaler, scaler_path)

    train_comp = group_counts_by_class(idx_train, y_all, groups_all)
    val_comp = group_counts_by_class(idx_val, y_all, groups_all)
    test_comp = group_counts_by_class(idx_test, y_all, groups_all)

    info = {
        "config_name": config_name,
        "fold": int(fold),
        "pca_config": config,
        "source_split": split_info,
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "class_counts": {
            "train": class_counts_dict(y_train),
            "val": class_counts_dict(y_val),
            "test": class_counts_dict(y_test),
        },
        "group_composition": {
            "train": train_comp,
            "val": val_comp,
            "test": test_comp,
        },
        "shapes": {
            "Z_train": list(Z_train.shape),
            "Z_val": list(Z_val.shape),
            "Z_test": list(Z_test.shape),
        },
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "runtime": {
            "pca_seconds": float(pca_seconds),
            "total_seconds": float(total_seconds),
        },
        "files": {
            "embeddings": str(embeddings_path),
            "pca_model": str(pca_path),
            "fft_scaler": str(scaler_path),
            "info": str(info_path),
        },
    }

    save_json(info, info_path)

    print(
        f"  Fold {fold} | {config_name} | "
        f"EVR_sum={info['explained_variance_ratio_sum']:.6f} | "
        f"total={total_seconds:.2f}s"
    )

    return info


def consolidate_config(config_name):
    config_dir = OUT_DIR / config_name
    artifacts_dir = config_dir / "artifacts"

    rows = []
    all_test_indices = []

    for fold in range(1, EXPECTED_N_FOLDS + 1):
        path = artifacts_dir / f"fold{fold}_{config_name}_embeddings.npz"

        if not path.exists():
            rows.append({
                "config_name": config_name,
                "fold": fold,
                "exists": False,
            })
            continue

        d = np.load(path, allow_pickle=True)

        idx_test = d["idx_test"].astype(int)
        all_test_indices.extend(idx_test.tolist())

        rows.append({
            "config_name": config_name,
            "fold": fold,
            "exists": True,
            "Z_train_shape": str(d["Z_train"].shape),
            "Z_val_shape": str(d["Z_val"].shape),
            "Z_test_shape": str(d["Z_test"].shape),
            "n_train": int(len(d["y_train"])),
            "n_val": int(len(d["y_val"])),
            "n_test": int(len(d["y_test"])),
            "explained_variance_ratio_sum": float(d["explained_variance_ratio_sum"][0]),
            "test_unique_indices": int(len(np.unique(idx_test))),
        })

    df = pd.DataFrame(rows)

    total_test = len(all_test_indices)
    unique_test = len(set(all_test_indices))
    duplicated_test = total_test - unique_test

    integrity = {
        "config_name": config_name,
        "n_folds": int(len(df[df["exists"] == True])),
        "total_test": int(total_test),
        "unique_test": int(unique_test),
        "duplicated_test": int(duplicated_test),
        "ok_total_test_3840_unique": bool(
            total_test == EXPECTED_TOTAL_TEST
            and unique_test == EXPECTED_TOTAL_TEST
            and duplicated_test == 0
        ),
    }

    df.to_csv(config_dir / f"{config_name}_artifacts_summary.csv", index=False, encoding="utf-8-sig")
    save_json(integrity, config_dir / f"{config_name}_integrity.json")

    if not integrity["ok_total_test_3840_unique"]:
        raise RuntimeError(f"{config_name}: integridad OOF falló: {integrity}")

    return df, integrity


def main():
    warnings.filterwarnings("default")

    print("=" * 100)
    print("01 - PCA 32 SENSITIVITY FROM VERIFIED PUMAP SPLITS")
    print("=" * 100)
    print("PROJECT_DIR:")
    print(PROJECT_DIR)
    print("FFT_DATASET_PATH:")
    print(FFT_DATASET_PATH)
    print("PUMAP_ARTIFACT_DIR:")
    print(PUMAP_ARTIFACT_DIR)
    print("OUT_DIR:")
    print(OUT_DIR)

    if not PUMAP_ARTIFACT_DIR.exists():
        raise FileNotFoundError(f"No existe PUMAP_ARTIFACT_DIR:\n{PUMAP_ARTIFACT_DIR}")

    X_all, y_all, groups_all, dataset_info = load_fft_dataset()

    save_json(
        {
            "script": "01_generate_pca_32_sensitivity_FROM_VERIFIED_PUMAP_SPLITS.py",
            "dataset_info": dataset_info,
            "pumap_artifact_dir": str(PUMAP_ARTIFACT_DIR),
            "out_dir": str(OUT_DIR),
            "global_seed": int(GLOBAL_SEED),
            "n_components": int(N_COMPONENTS),
            "pca_configs": PCA_CONFIGS,
            "protocol": {
                "splits_source": "verified PUMAP artifacts",
                "fft_scaler_fit": "internal train only, independently per fold",
                "pca_fit": "internal train only",
                "validation": "transform only",
                "external_test": "transform only",
                "no_new_folds_created": True,
                "no_raw_csv_used": True,
            },
        },
        OUT_DIR / "pca_32_sensitivity_config.json",
    )

    split_audit_rows = []
    all_infos = []

    run_start = time.time()

    for fold in range(1, EXPECTED_N_FOLDS + 1):
        idx_train, idx_val, idx_test, split_info = load_verified_pumap_indices(fold)

        split_check = validate_split(
            fold=fold,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            y_all=y_all,
            groups_all=groups_all,
        )

        split_audit_rows.append({
            "fold": fold,
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
            "n_test": int(len(idx_test)),
            "ok_no_overlap": split_check["ok_no_overlap"],
            **{f"val_{k}": v for k, v in split_check["validation_composition"].items()},
        })

        print("\n" + "=" * 100)
        print(f"FOLD {fold} verificado")
        print("=" * 100)
        print("n_train:", len(idx_train), "n_val:", len(idx_val), "n_test:", len(idx_test))
        print("val composition:", split_check["validation_composition"])

        for config in PCA_CONFIGS:
            info = fit_transform_pca_for_config(
                config=config,
                fold=fold,
                X_all=X_all,
                y_all=y_all,
                groups_all=groups_all,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=idx_test,
                split_info=split_info,
            )
            all_infos.append(info)

    split_audit_df = pd.DataFrame(split_audit_rows)
    split_audit_df.to_csv(
        OUT_DIR / "verified_pumap_splits_used_for_pca.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_summary_rows = []
    all_integrities = []

    for config in PCA_CONFIGS:
        config_name = config["name"]
        df_config, integrity = consolidate_config(config_name)

        all_summary_rows.append(df_config)
        all_integrities.append(integrity)

    all_summary_df = pd.concat(all_summary_rows, ignore_index=True)
    all_summary_df.to_csv(
        OUT_DIR / "ALL_PCA_32_artifacts_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        all_integrities,
        OUT_DIR / "ALL_PCA_32_integrity.json",
    )

    total_seconds = time.time() - run_start

    run_summary = {
        "status": "finished",
        "total_seconds": float(total_seconds),
        "total_minutes": float(total_seconds / 60),
        "out_dir": str(OUT_DIR),
        "n_configs": int(len(PCA_CONFIGS)),
        "n_folds": int(EXPECTED_N_FOLDS),
        "all_integrities": all_integrities,
    }

    save_json(run_summary, OUT_DIR / "run_summary_pca_32_sensitivity.json")

    print("\n" + "=" * 100)
    print("PCA 32 SENSITIVITY TERMINADO")
    print("=" * 100)
    print(f"Tiempo total: {total_seconds:.2f} s = {total_seconds / 60:.2f} min")
    print("Salida:")
    print(OUT_DIR)

    print("\nResumen integridad:")
    print(pd.DataFrame(all_integrities).to_string(index=False))


if __name__ == "__main__":
    main()
