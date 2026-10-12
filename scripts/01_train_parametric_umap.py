# -*- coding: utf-8 -*-
"""Train fold-specific Parametric UMAP representations."""

import os
import json
import time
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from umap.parametric_umap import ParametricUMAP
except Exception as exc:
    raise ImportError(
        "No se pudo importar ParametricUMAP. Verifica tu entorno.\n"
        "Prueba:\n"
        "    pip install umap-learn\n\n"
        f"Error original: {exc}"
    )


# ============================================================
# RUTAS
# ============================================================

PROJECT_DIR = Path(os.environ.get("WILEY_PROJECT_DIR", Path(__file__).resolve().parents[1]))

PIPELINE00_DIR = PROJECT_DIR / "00_RESULTS_PIPELINE_01_FFT_FOLDS_REPRODUCIBLE"

CONFIG_NAME = "PUMAP_Robust_md005_tw075"

BASE_DIR = PROJECT_DIR / CONFIG_NAME
ARTIFACTS_DIR = BASE_DIR / "artifacts"

EXPECTED_N_FOLDS = 5
EXPECTED_TOTAL_TEST_SAMPLES = 3840

CLASS_NAMES = ["Healthy", "6Nm", "9Nm", "NoBolt"]

RANDOM_STATE = 42


# ============================================================
# PARAMETRIC UMAP
# ============================================================

PUMAP_CONFIG = {
    "name": CONFIG_NAME,
    "n_neighbors": 30,
    "min_dist": 0.05,
    "target_weight": 0.75,
    "epochs": 75,
    "n_components": 32,
    "metric": "euclidean",
    "batch_size": 256,
    "random_state": RANDOM_STATE,
}

ENCODER_CONFIG = {
    "architecture": [
        {"type": "Dense", "units": 256, "activation": "relu"},
        {"type": "BatchNormalization"},
        {"type": "Dropout", "rate": 0.10},
        {"type": "Dense", "units": 128, "activation": "relu"},
        {"type": "BatchNormalization"},
        {"type": "Dropout", "rate": 0.10},
        {"type": "Dense", "units": 64, "activation": "relu"},
        {"type": "Dense", "units": 32, "activation": "linear"},
    ]
}


# ============================================================
# UTILIDADES
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def configure_tensorflow():
    """
    Evita que TensorFlow tome toda la memoria GPU.
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] GPU detectada(s): {gpus}")
        else:
            print("[INFO] No se detectó GPU. Se usará CPU.")

    except Exception as exc:
        print(f"[WARN] No se pudo configurar memory_growth: {exc}")

    tf.keras.utils.set_random_seed(RANDOM_STATE)


def build_encoder(input_dim: int):
    """
    Encoder final:

        Input D
        Dense 256 ReLU
        BatchNorm
        Dropout 0.10
        Dense 128 ReLU
        BatchNorm
        Dropout 0.10
        Dense 64 ReLU
        Dense 32 Linear
    """
    inputs = keras.Input(shape=(input_dim,), name="fft_features_scaled")

    x = layers.Dense(256, activation="relu", name="dense_256_relu")(inputs)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(0.10, name="dropout_010_1")(x)

    x = layers.Dense(128, activation="relu", name="dense_128_relu")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dropout(0.10, name="dropout_010_2")(x)

    x = layers.Dense(64, activation="relu", name="dense_64_relu")(x)

    outputs = layers.Dense(
        PUMAP_CONFIG["n_components"],
        activation="linear",
        name="latent_32_linear",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{CONFIG_NAME}_encoder",
    )

    return model


def load_preprocessed_fold(fold: int):
    """
    Carga el fold ya preparado por PIPELINE 00.
    """
    fold_dir = PIPELINE00_DIR / f"fold{fold}"
    npz_path = fold_dir / f"fold{fold}_preprocessed_fft.npz"
    scaler_path = fold_dir / f"fold{fold}_fft_scaler.pkl"
    info_path = fold_dir / f"fold{fold}_split_info.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"No existe el fold preprocesado:\n{npz_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"No existe el scaler del fold:\n{scaler_path}")

    if not info_path.exists():
        raise FileNotFoundError(f"No existe el JSON info del fold:\n{info_path}")

    data = np.load(npz_path, allow_pickle=True)

    required_keys = [
        "X_train_scaled",
        "X_val_scaled",
        "X_test_scaled",
        "y_train",
        "y_val",
        "y_test",
        "groups_train",
        "groups_val",
        "groups_test",
        "sample_index_train",
        "sample_index_val",
        "sample_index_test",
        "train_idx",
        "val_idx",
        "test_idx",
    ]

    missing = [k for k in required_keys if k not in data.files]

    if missing:
        raise KeyError(
            f"Faltan keys en {npz_path}:\n{missing}\n"
            f"Keys disponibles:\n{data.files}"
        )

    with open(info_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    fold_data = {
        "fold": fold,
        "npz_path": npz_path,
        "scaler_path": scaler_path,
        "info_path": info_path,
        "split_info": split_info,

        "X_train_scaled": data["X_train_scaled"].astype(np.float32),
        "X_val_scaled": data["X_val_scaled"].astype(np.float32),
        "X_test_scaled": data["X_test_scaled"].astype(np.float32),

        "y_train": data["y_train"].astype(int),
        "y_val": data["y_val"].astype(int),
        "y_test": data["y_test"].astype(int),

        "groups_train": data["groups_train"].astype(int),
        "groups_val": data["groups_val"].astype(int),
        "groups_test": data["groups_test"].astype(int),

        "sample_index_train": data["sample_index_train"].astype(int),
        "sample_index_val": data["sample_index_val"].astype(int),
        "sample_index_test": data["sample_index_test"].astype(int),

        "train_idx": data["train_idx"].astype(int),
        "val_idx": data["val_idx"].astype(int),
        "test_idx": data["test_idx"].astype(int),
    }

    return fold_data


def artifact_fold_done(fold: int):
    """
    Verifica si un fold ya tiene todos los artefactos correctos.
    """
    stem = f"fold{fold}_{CONFIG_NAME}"

    embeddings_path = ARTIFACTS_DIR / f"{stem}_embeddings.npz"
    encoder_path = ARTIFACTS_DIR / f"{stem}_encoder.keras"
    scaler_copy_path = ARTIFACTS_DIR / f"{stem}_fft_scaler.pkl"
    info_path = ARTIFACTS_DIR / f"{stem}_info.json"

    required = [
        embeddings_path,
        encoder_path,
        scaler_copy_path,
        info_path,
    ]

    if not all(p.exists() for p in required):
        return False

    try:
        data = np.load(embeddings_path, allow_pickle=True)

        required_keys = [
            "Z_train",
            "Z_val",
            "Z_test",
            "y_train",
            "y_val",
            "y_test",
            "sample_index_test",
        ]

        if not all(k in data.files for k in required_keys):
            return False

        if data["Z_train"].shape[1] != PUMAP_CONFIG["n_components"]:
            return False

        if data["Z_val"].shape[1] != PUMAP_CONFIG["n_components"]:
            return False

        if data["Z_test"].shape[1] != PUMAP_CONFIG["n_components"]:
            return False

    except Exception:
        return False

    return True


def train_pumap_from_preprocessed_fold(fold_data):
    """
    Entrena Parametric UMAP usando el fold ya preprocesado por PIPELINE 00.
    """
    fold = fold_data["fold"]

    print("\n" + "=" * 90)
    print(f"PIPELINE 01 | Parametric UMAP | Fold {fold}")
    print("=" * 90)

    fold_start = time.time()

    X_train = fold_data["X_train_scaled"]
    X_val = fold_data["X_val_scaled"]
    X_test = fold_data["X_test_scaled"]

    y_train = fold_data["y_train"]
    y_val = fold_data["y_val"]
    y_test = fold_data["y_test"]

    input_dim = X_train.shape[1]

    print(f"X_train_scaled: {X_train.shape}")
    print(f"X_val_scaled  : {X_val.shape}")
    print(f"X_test_scaled : {X_test.shape}")
    print(f"y_train       : {np.bincount(y_train, minlength=4)}")
    print(f"y_val         : {np.bincount(y_val, minlength=4)}")
    print(f"y_test        : {np.bincount(y_test, minlength=4)}")

    encoder = build_encoder(input_dim)

    print("\nEncoder:")
    encoder.summary()

    pumap = ParametricUMAP(
        n_components=PUMAP_CONFIG["n_components"],
        n_neighbors=PUMAP_CONFIG["n_neighbors"],
        min_dist=PUMAP_CONFIG["min_dist"],
        metric=PUMAP_CONFIG["metric"],
        target_metric="categorical",
        target_weight=PUMAP_CONFIG["target_weight"],
        encoder=encoder,
        batch_size=PUMAP_CONFIG["batch_size"],
        n_epochs=PUMAP_CONFIG["epochs"],
        random_state=PUMAP_CONFIG["random_state"],
        verbose=True,
    )

    fit_start = time.time()

    Z_train = pumap.fit_transform(X_train, y=y_train).astype(np.float32)

    fit_seconds = time.time() - fit_start

    Z_val = pumap.transform(X_val).astype(np.float32)
    Z_test = pumap.transform(X_test).astype(np.float32)

    total_seconds = time.time() - fold_start

    if Z_train.shape[1] != PUMAP_CONFIG["n_components"]:
        raise RuntimeError(f"Z_train shape inválida: {Z_train.shape}")

    if Z_val.shape[1] != PUMAP_CONFIG["n_components"]:
        raise RuntimeError(f"Z_val shape inválida: {Z_val.shape}")

    if Z_test.shape[1] != PUMAP_CONFIG["n_components"]:
        raise RuntimeError(f"Z_test shape inválida: {Z_test.shape}")

    stem = f"fold{fold}_{CONFIG_NAME}"

    embeddings_path = ARTIFACTS_DIR / f"{stem}_embeddings.npz"
    encoder_path = ARTIFACTS_DIR / f"{stem}_encoder.keras"
    scaler_copy_path = ARTIFACTS_DIR / f"{stem}_fft_scaler.pkl"
    info_path = ARTIFACTS_DIR / f"{stem}_info.json"

    np.savez_compressed(
        embeddings_path,

        Z_train=Z_train,
        Z_val=Z_val,
        Z_test=Z_test,

        y_train=fold_data["y_train"],
        y_val=fold_data["y_val"],
        y_test=fold_data["y_test"],

        groups_train=fold_data["groups_train"],
        groups_val=fold_data["groups_val"],
        groups_test=fold_data["groups_test"],

        sample_index_train=fold_data["sample_index_train"],
        sample_index_val=fold_data["sample_index_val"],
        sample_index_test=fold_data["sample_index_test"],

        train_idx=fold_data["train_idx"],
        val_idx=fold_data["val_idx"],
        test_idx=fold_data["test_idx"],
    )

    encoder.save(encoder_path)

    shutil.copy2(fold_data["scaler_path"], scaler_copy_path)

    info = {
        "config_name": CONFIG_NAME,
        "fold": int(fold),
        "source": {
            "pipeline00_npz": str(fold_data["npz_path"]),
            "pipeline00_scaler": str(fold_data["scaler_path"]),
            "pipeline00_info": str(fold_data["info_path"]),
        },
        "pumap_config": PUMAP_CONFIG,
        "encoder_config": {
            **ENCODER_CONFIG,
            "input_dim": int(input_dim),
        },
        "n_train": int(len(fold_data["y_train"])),
        "n_val": int(len(fold_data["y_val"])),
        "n_test": int(len(fold_data["y_test"])),
        "class_counts": {
            "train": {
                CLASS_NAMES[i]: int(v)
                for i, v in enumerate(np.bincount(fold_data["y_train"], minlength=4))
            },
            "val": {
                CLASS_NAMES[i]: int(v)
                for i, v in enumerate(np.bincount(fold_data["y_val"], minlength=4))
            },
            "test": {
                CLASS_NAMES[i]: int(v)
                for i, v in enumerate(np.bincount(fold_data["y_test"], minlength=4))
            },
        },
        "groups": {
            "n_train_groups": int(len(np.unique(fold_data["groups_train"]))),
            "n_val_groups": int(len(np.unique(fold_data["groups_val"]))),
            "n_test_groups": int(len(np.unique(fold_data["groups_test"]))),
            "train_groups": sorted([int(v) for v in np.unique(fold_data["groups_train"])]),
            "val_groups": sorted([int(v) for v in np.unique(fold_data["groups_val"])]),
            "test_groups": sorted([int(v) for v in np.unique(fold_data["groups_test"])]),
        },
        "shapes": {
            "Z_train": list(Z_train.shape),
            "Z_val": list(Z_val.shape),
            "Z_test": list(Z_test.shape),
        },
        "timing": {
            "fit_seconds": float(fit_seconds),
            "total_seconds": float(total_seconds),
        },
        "artifacts": {
            "embeddings": str(embeddings_path),
            "encoder": str(encoder_path),
            "fft_scaler_copy": str(scaler_copy_path),
            "info": str(info_path),
        },
    }

    save_json(info, info_path)

    print("\nGuardado:")
    print(f"  {embeddings_path}")
    print(f"  {encoder_path}")
    print(f"  {scaler_copy_path}")
    print(f"  {info_path}")
    print(f"  fit_seconds   = {fit_seconds:.2f}")
    print(f"  total_seconds = {total_seconds:.2f}")

    return info


def consolidate_artifacts():
    """
    Consolida resumen de artefactos.
    """
    rows = []

    for fold in range(1, EXPECTED_N_FOLDS + 1):
        path = ARTIFACTS_DIR / f"fold{fold}_{CONFIG_NAME}_embeddings.npz"

        if not path.exists():
            rows.append({
                "fold": fold,
                "exists": False,
            })
            continue

        data = np.load(path, allow_pickle=True)

        rows.append({
            "fold": fold,
            "exists": True,
            "file": str(path),
            "Z_train_shape": str(data["Z_train"].shape),
            "Z_val_shape": str(data["Z_val"].shape),
            "Z_test_shape": str(data["Z_test"].shape),
            "n_train": int(len(data["y_train"])),
            "n_val": int(len(data["y_val"])),
            "n_test": int(len(data["y_test"])),
            "test_class_counts": str(
                {
                    CLASS_NAMES[i]: int(v)
                    for i, v in enumerate(np.bincount(data["y_test"], minlength=4))
                }
            ),
            "sample_index_test_unique": int(len(np.unique(data["sample_index_test"]))),
        })

    df = pd.DataFrame(rows)

    summary_path = BASE_DIR / f"{CONFIG_NAME}_artifacts_summary_FROM_PIPELINE00.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    total_test = int(df.loc[df["exists"] == True, "n_test"].sum()) if "n_test" in df.columns else 0

    all_test_indices = []

    for fold in range(1, EXPECTED_N_FOLDS + 1):
        path = ARTIFACTS_DIR / f"fold{fold}_{CONFIG_NAME}_embeddings.npz"

        if path.exists():
            data = np.load(path, allow_pickle=True)
            all_test_indices.extend(data["sample_index_test"].astype(int).tolist())

    unique_test = len(set(all_test_indices))
    duplicated_test = len(all_test_indices) - unique_test

    integrity = {
        "config_name": CONFIG_NAME,
        "n_expected_folds": int(EXPECTED_N_FOLDS),
        "n_embedding_files": int(df["exists"].sum()) if "exists" in df.columns else 0,
        "total_test_samples": int(total_test),
        "unique_test_sample_index": int(unique_test),
        "duplicated_test_sample_index": int(duplicated_test),
        "expected_total_test_samples": int(EXPECTED_TOTAL_TEST_SAMPLES),
        "ok_total_test_samples": bool(total_test == EXPECTED_TOTAL_TEST_SAMPLES),
        "ok_unique_test_samples": bool(
            total_test == EXPECTED_TOTAL_TEST_SAMPLES
            and unique_test == EXPECTED_TOTAL_TEST_SAMPLES
            and duplicated_test == 0
        ),
        "artifacts_dir": str(ARTIFACTS_DIR),
        "summary_csv": str(summary_path),
    }

    save_json(integrity, BASE_DIR / f"{CONFIG_NAME}_artifacts_integrity_FROM_PIPELINE00.json")

    print("\nResumen de artefactos:")
    print(df.to_string(index=False))

    print("\nIntegridad:")
    print(json.dumps(integrity, indent=2, ensure_ascii=False))

    if not integrity["ok_unique_test_samples"]:
        raise RuntimeError(
            "La integridad de artefactos falló: los test no suman 3840 únicos."
        )

    return df, integrity


def run():
    print("=" * 90)
    print("01_generate_pumap_artifacts_md005_tw075_FROM_PIPELINE00")
    print("=" * 90)
    print("PROJECT_DIR   :", PROJECT_DIR)
    print("PIPELINE00_DIR:", PIPELINE00_DIR)
    print("BASE_DIR      :", BASE_DIR)
    print("ARTIFACTS_DIR :", ARTIFACTS_DIR)

    if not PIPELINE00_DIR.exists():
        raise FileNotFoundError(
            f"No existe la salida del PIPELINE 00:\n{PIPELINE00_DIR}\n"
            "Primero ejecuta 00_pipeline_01_fft_folds_reproducible.py"
        )

    configure_tensorflow()

    ensure_dir(BASE_DIR)
    ensure_dir(ARTIFACTS_DIR)

    save_json(
        {
            "script": "01_generate_pumap_artifacts_md005_tw075_FROM_PIPELINE00.py",
            "config_name": CONFIG_NAME,
            "pipeline00_dir": str(PIPELINE00_DIR),
            "base_dir": str(BASE_DIR),
            "artifacts_dir": str(ARTIFACTS_DIR),
            "pumap_config": PUMAP_CONFIG,
            "encoder_config": ENCODER_CONFIG,
            "protocol": {
                "folds_source": "PIPELINE 00 preprocessed folds",
                "pumap_fit": "fit only on X_train_scaled of each fold",
                "validation": "transform X_val_scaled only",
                "external_test": "transform X_test_scaled only",
                "no_new_splits_created": True,
                "no_new_fft_scaler_fitted": True,
            },
        },
        BASE_DIR / f"{CONFIG_NAME}_config_FROM_PIPELINE00.json",
    )

    run_start = time.time()

    fold_infos = []

    for fold in range(1, EXPECTED_N_FOLDS + 1):

        if artifact_fold_done(fold):
            print("\n" + "=" * 90)
            print(f"Fold {fold} ya tiene artefactos completos. Se omite.")
            print("=" * 90)

            info_path = ARTIFACTS_DIR / f"fold{fold}_{CONFIG_NAME}_info.json"
            with open(info_path, "r", encoding="utf-8") as f:
                fold_infos.append(json.load(f))

            continue

        fold_data = load_preprocessed_fold(fold)
        info = train_pumap_from_preprocessed_fold(fold_data)
        fold_infos.append(info)

    total_seconds = time.time() - run_start

    summary_df, integrity = consolidate_artifacts()

    run_info = {
        "script": "01_generate_pumap_artifacts_md005_tw075_FROM_PIPELINE00.py",
        "config_name": CONFIG_NAME,
        "pipeline00_dir": str(PIPELINE00_DIR),
        "base_dir": str(BASE_DIR),
        "artifacts_dir": str(ARTIFACTS_DIR),
        "pumap_config": PUMAP_CONFIG,
        "encoder_config": ENCODER_CONFIG,
        "total_seconds": float(total_seconds),
        "total_minutes": float(total_seconds / 60),
        "integrity": integrity,
        "fold_infos": fold_infos,
    }

    save_json(run_info, BASE_DIR / f"{CONFIG_NAME}_run_info_FROM_PIPELINE00.json")

    print("\n" + "=" * 90)
    print("PIPELINE 01 COMPLETADO ✅")
    print("=" * 90)
    print(f"Tiempo total: {total_seconds:.2f} s = {total_seconds/60:.2f} min")
    print(f"Artefactos guardados en:\n{ARTIFACTS_DIR}")
    print(f"Run info:\n{BASE_DIR / f'{CONFIG_NAME}_run_info_FROM_PIPELINE00.json'}")


if __name__ == "__main__":
    warnings.filterwarnings("default")
    run()
