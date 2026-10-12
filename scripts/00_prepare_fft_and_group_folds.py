# -*- coding: utf-8 -*-
"""Prepare FFT features and group-disjoint outer/inner folds."""

import os
import json
import random
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

SEED = 42

PROJECT_DIR = Path(os.environ.get("WILEY_PROJECT_DIR", Path(__file__).resolve().parents[1]))

BASE_DIR = PROJECT_DIR / "A_1"

OUT_DIR = PROJECT_DIR / "00_RESULTS_PIPELINE_01_FFT_FOLDS_REPRODUCIBLE"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUTAS = {
    "6Nm": BASE_DIR / "6Nm" / "level_1",
    "9Nm": BASE_DIR / "9Nm" / "level_1",
    "Healthy": BASE_DIR / "Healthy",
    "NoBolt": BASE_DIR / "NoBolt" / "level_1",
}

LABELS = {
    "Healthy": 0,
    "6Nm": 1,
    "9Nm": 2,
    "NoBolt": 3,
}

CLASS_NAMES = ["Healthy", "6Nm", "9Nm", "NoBolt"]
N_CLASSES = len(CLASS_NAMES)

W = 1024
HOP = 512
EXPECTED_CHANNELS = 24
CSV_SEP = ","

N_SPLITS = 5
N_VAL_GROUPS_PER_CLASS = 3
EXPECTED_VAL_GROUPS_TOTAL = N_VAL_GROUPS_PER_CLASS * N_CLASSES

USE_FLOAT32 = True
USE_FORTRAN_ORDER = False

INNER_VAL_MODE = "balanced_3groups_per_class"

EXPECTED_TOTAL_WINDOWS = 3840


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def get_float_dtype():
    return np.float32 if USE_FLOAT32 else np.float64


FLOAT_DTYPE = get_float_dtype()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def class_counts(y, n_classes=N_CLASSES):
    return np.bincount(np.asarray(y).astype(int), minlength=n_classes)


def class_counts_dict(y):
    bc = class_counts(y, N_CLASSES)
    return {CLASS_NAMES[i]: int(bc[i]) for i in range(N_CLASSES)}


def check_group_overlap(groups_a, groups_b):
    return set(np.unique(groups_a)).intersection(set(np.unique(groups_b)))


def check_no_group_overlap(name_a, groups_a, name_b, groups_b):
    overlap = check_group_overlap(groups_a, groups_b)
    if overlap:
        raise RuntimeError(
            f"FUGA DE GRUPOS entre {name_a} y {name_b}: {sorted(list(overlap))}"
        )


def list_csvs():
    rows = []

    for cls, folder in RUTAS.items():
        folder = Path(folder)

        if not folder.exists():
            raise FileNotFoundError(
                f"No existe la carpeta para clase {cls}:\n{folder}"
            )

        files = sorted(folder.glob("*.csv"))

        for f in files:
            rows.append({
                "path": str(f),
                "class": cls,
                "label": LABELS[cls],
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No encontré ningún CSV. Revisa las rutas.")

    print("\n[INFO] Archivos CSV encontrados:")
    print(df["class"].value_counts())

    df.to_csv(OUT_DIR / "csv_files_index.csv", index=False, encoding="utf-8-sig")

    return df


def load_csv_24ch(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path, sep=CSV_SEP)

    # Caso típico: primera columna de tiempo/índice + 24 sensores
    if df.shape[1] == EXPECTED_CHANNELS + 1:
        df = df.iloc[:, 1:]

    # Si hay más columnas, tomar las últimas 24 como sensores
    if df.shape[1] > EXPECTED_CHANNELS:
        df = df.iloc[:, -EXPECTED_CHANNELS:]

    if df.shape[1] != EXPECTED_CHANNELS:
        raise ValueError(
            f"{csv_path}\n"
            f"Esperaba {EXPECTED_CHANNELS} columnas de sensores, "
            f"pero encontré {df.shape[1]}.\n"
            f"Revisa separador, columnas extra o estructura del CSV."
        )

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill().fillna(0)

    X = df.to_numpy(dtype=FLOAT_DTYPE)

    return X


def make_windows(X: np.ndarray, w: int, hop: int) -> np.ndarray:
    N, C = X.shape

    if N < w:
        return np.empty((0, w, C), dtype=FLOAT_DTYPE)

    starts = np.arange(0, N - w + 1, hop, dtype=int)
    out = np.empty((len(starts), w, C), dtype=FLOAT_DTYPE)

    for i, s in enumerate(starts):
        out[i] = X[s:s + w, :]

    return out


def window_fft_mag(window_1024x24: np.ndarray) -> np.ndarray:
    spec = np.fft.rfft(window_1024x24, axis=0)
    mag = np.abs(spec).astype(FLOAT_DTYPE)

    if USE_FORTRAN_ORDER:
        return mag.reshape(-1, order="F")

    return mag.reshape(-1)


def build_external_splitter():
    return StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )


def validate_group_purity(y, groups):
    """
    Verifica que cada grupo/experimento tenga una sola etiqueta.
    """
    bad = []

    for g in np.unique(groups):
        labels_g = np.unique(y[groups == g])
        if len(labels_g) != 1:
            bad.append((int(g), labels_g.tolist()))

    if bad:
        raise RuntimeError(
            "Hay grupos con más de una clase. Esto rompe StratifiedGroupKFold.\n"
            f"Ejemplos: {bad[:10]}"
        )


# ============================================================
# STEP 1: CREAR DATASET BASE
# ============================================================

def step_1_create_dataset():
    print("\n" + "=" * 80)
    print("STEP 1 - Crear dataset crudo concatenado y FFT por ventanas")
    print("=" * 80)

    df_files = list_csvs()

    (OUT_DIR / "NPM_all_files_PIPELINE_01.txt").write_text(
        "\n".join(df_files["path"].tolist()),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # 1A. Concatenar señales crudas
    # ------------------------------------------------------------

    raw_arrays = []
    raw_file_id = []
    raw_labels_per_row = []

    paths = []
    classes = []
    labels_per_file = []
    starts = []
    ends = []
    lengths = []

    row_cursor = 0

    print("\n[STEP 1A] Uniendo señales crudas de 24 canales...")

    for file_idx, row in df_files.reset_index(drop=True).iterrows():
        path = row["path"]
        lab = int(row["label"])

        X = load_csv_24ch(path)
        n = X.shape[0]

        raw_arrays.append(X)
        raw_file_id.append(np.full((n,), file_idx, dtype=np.int32))
        raw_labels_per_row.append(np.full((n,), lab, dtype=np.int8))

        starts.append(row_cursor)
        ends.append(row_cursor + n)
        lengths.append(n)

        row_cursor += n

        paths.append(path)
        classes.append(row["class"])
        labels_per_file.append(lab)

        if (file_idx + 1) % 10 == 0:
            print(f"  - {file_idx + 1}/{len(df_files)} archivos procesados...")

    X_raw = np.vstack(raw_arrays).astype(FLOAT_DTYPE)
    file_id = np.concatenate(raw_file_id, axis=0)
    y_row = np.concatenate(raw_labels_per_row, axis=0)

    starts = np.array(starts, dtype=np.int64)
    ends = np.array(ends, dtype=np.int64)
    lengths = np.array(lengths, dtype=np.int64)
    labels_per_file = np.array(labels_per_file, dtype=np.int8)

    raw_path = OUT_DIR / "A1_concat_raw_24ch.npz"

    np.savez_compressed(
        raw_path,
        X_raw=X_raw,
        y_row=y_row,
        file_id=file_id,
        starts=starts,
        ends=ends,
        lengths=lengths,
        paths=np.array(paths, dtype=object),
        classes=np.array(classes, dtype=object),
        y_file=labels_per_file,
        expected_channels=np.int32(EXPECTED_CHANNELS),
        seed=np.int32(SEED),
    )

    print(f"[OK] Guardado: {raw_path}")
    print(f"     X_raw shape = {X_raw.shape}")

    # ------------------------------------------------------------
    # 1B. Ventanas + FFT
    # ------------------------------------------------------------

    print("\n[STEP 1B] Ventanas W=1024, HOP=512 + FFT magnitud...")

    fft_features = []
    fft_y = []
    fft_groups = []
    fft_sample_index = []
    fft_window_in_group = []

    total_windows = 0

    per_file_rows = []

    for file_idx, row in df_files.reset_index(drop=True).iterrows():
        path = row["path"]
        lab = int(row["label"])
        cls = row["class"]

        X = load_csv_24ch(path)
        Wn = make_windows(X, W, HOP)
        nW = Wn.shape[0]

        if nW == 0:
            print(f"[WARN] {path} no genera ventanas. N={X.shape[0]}")
            continue

        feats_this_file = []

        for k in range(nW):
            feat = window_fft_mag(Wn[k])
            feats_this_file.append(feat)

        feats_this_file = np.vstack(feats_this_file).astype(FLOAT_DTYPE)

        fft_features.append(feats_this_file)
        fft_y.append(np.full((nW,), lab, dtype=np.int8))
        fft_groups.append(np.full((nW,), file_idx, dtype=np.int32))
        fft_sample_index.append(np.arange(total_windows, total_windows + nW, dtype=np.int64))
        fft_window_in_group.append(np.arange(nW, dtype=np.int32))

        per_file_rows.append({
            "file_idx": int(file_idx),
            "class": cls,
            "label": lab,
            "path": path,
            "n_raw_samples": int(X.shape[0]),
            "n_windows": int(nW),
        })

        total_windows += nW

        if (file_idx + 1) % 10 == 0:
            print(
                f"  - {file_idx + 1}/{len(df_files)} archivos -> "
                f"ventanas acumuladas: {total_windows}"
            )

    X_fft = np.vstack(fft_features).astype(FLOAT_DTYPE)
    y_fft = np.concatenate(fft_y, axis=0).astype(np.int64)
    groups = np.concatenate(fft_groups, axis=0).astype(np.int64)
    sample_index = np.concatenate(fft_sample_index, axis=0).astype(np.int64)
    window_in_group = np.concatenate(fft_window_in_group, axis=0).astype(np.int64)

    validate_group_purity(y_fft, groups)

    fft_path = OUT_DIR / "A1_windows_fft_W1024_H512_24ch.npz"

    np.savez_compressed(
        fft_path,
        X_fft=X_fft,
        X=X_fft,  # alias útil para scripts posteriores
        y=y_fft,
        labels=y_fft,  # alias
        groups=groups,
        group=groups,  # alias
        sample_index=sample_index,
        window_in_group=window_in_group,
        paths=np.array(paths, dtype=object),
        classes=np.array(classes, dtype=object),
        y_file=labels_per_file,
        W=np.int32(W),
        HOP=np.int32(HOP),
        n_channels=np.int32(EXPECTED_CHANNELS),
        seed=np.int32(SEED),
        class_names=np.array(CLASS_NAMES, dtype=object),
    )

    per_file_df = pd.DataFrame(per_file_rows)
    per_file_df.to_csv(
        OUT_DIR / "A1_windows_per_file_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dataset_info = {
        "raw_path": str(raw_path),
        "fft_path": str(fft_path),
        "X_raw_shape": list(X_raw.shape),
        "X_fft_shape": list(X_fft.shape),
        "n_windows": int(X_fft.shape[0]),
        "n_features": int(X_fft.shape[1]),
        "n_groups": int(len(np.unique(groups))),
        "class_counts": class_counts_dict(y_fft),
        "W": int(W),
        "HOP": int(HOP),
        "n_channels": int(EXPECTED_CHANNELS),
        "use_float32": bool(USE_FLOAT32),
        "use_fortran_order": bool(USE_FORTRAN_ORDER),
        "seed": int(SEED),
    }

    save_json(dataset_info, OUT_DIR / "A1_dataset_info.json")

    print(f"[OK] Guardado: {fft_path}")
    print(f"     X_fft shape = {X_fft.shape}")
    print(f"     grupos únicos = {len(np.unique(groups))}")
    print(f"     clases = {class_counts_dict(y_fft)}")

    if X_fft.shape[0] != EXPECTED_TOTAL_WINDOWS:
        warnings.warn(
            f"El total de ventanas es {X_fft.shape[0]}, "
            f"pero se esperaban {EXPECTED_TOTAL_WINDOWS}."
        )

    return fft_path


# ============================================================
# STEP 2: SPLITS EXTERNOS + SPLIT INTERNO BALANCEADO
# ============================================================

def select_balanced_internal_validation_groups(dev_idx, y, groups, fold):
    """
    Selecciona exactamente 3 grupos por clase dentro del conjunto externo de desarrollo.

    Esto reemplaza el criterio viejo de:
        "primer split con 12 grupos"

    Ahora la validación interna queda balanceada:
        3 grupos Healthy
        3 grupos 6Nm
        3 grupos 9Nm
        3 grupos NoBolt
    """
    rng = np.random.default_rng(SEED + 1000 + fold)

    dev_groups = np.unique(groups[dev_idx])
    val_groups = []

    for c in range(N_CLASSES):
        class_groups = []

        for g in dev_groups:
            labels_g = np.unique(y[groups == g])
            if len(labels_g) != 1:
                raise RuntimeError(f"Grupo {g} con múltiples etiquetas: {labels_g}")

            if int(labels_g[0]) == int(c):
                class_groups.append(int(g))

        class_groups = np.array(sorted(class_groups), dtype=np.int64)

        if len(class_groups) < N_VAL_GROUPS_PER_CLASS:
            raise RuntimeError(
                f"Fold {fold}: clase {CLASS_NAMES[c]} tiene solo "
                f"{len(class_groups)} grupos disponibles para validación interna. "
                f"Se requieren {N_VAL_GROUPS_PER_CLASS}."
            )

        selected = rng.choice(
            class_groups,
            size=N_VAL_GROUPS_PER_CLASS,
            replace=False,
        )

        val_groups.extend([int(v) for v in selected])

    val_groups = np.array(sorted(val_groups), dtype=np.int64)

    if len(val_groups) != EXPECTED_VAL_GROUPS_TOTAL:
        raise RuntimeError(
            f"Fold {fold}: validación interna debería tener "
            f"{EXPECTED_VAL_GROUPS_TOTAL} grupos, pero tiene {len(val_groups)}."
        )

    val_mask = np.isin(groups, val_groups)
    dev_mask = np.zeros_like(y, dtype=bool)
    dev_mask[dev_idx] = True

    val_idx = np.where(val_mask & dev_mask)[0].astype(np.int64)

    train_idx = np.array(
        sorted(list(set(dev_idx.tolist()) - set(val_idx.tolist()))),
        dtype=np.int64,
    )

    return train_idx, val_idx, val_groups


def validate_split_integrity(fold, train_idx, val_idx, test_idx, y, groups):
    """
    Verificaciones duras contra fuga.
    """
    set_train = set(train_idx.tolist())
    set_val = set(val_idx.tolist())
    set_test = set(test_idx.tolist())

    if set_train & set_val:
        raise RuntimeError(f"Fold {fold}: solape de muestras train-val.")
    if set_train & set_test:
        raise RuntimeError(f"Fold {fold}: solape de muestras train-test.")
    if set_val & set_test:
        raise RuntimeError(f"Fold {fold}: solape de muestras val-test.")

    g_train = groups[train_idx]
    g_val = groups[val_idx]
    g_test = groups[test_idx]

    check_no_group_overlap("train_internal", g_train, "val_internal", g_val)
    check_no_group_overlap("train_internal", g_train, "test_external", g_test)
    check_no_group_overlap("val_internal", g_val, "test_external", g_test)

    if len(np.unique(g_val)) != EXPECTED_VAL_GROUPS_TOTAL:
        raise RuntimeError(
            f"Fold {fold}: validación interna tiene {len(np.unique(g_val))} grupos; "
            f"se esperaban {EXPECTED_VAL_GROUPS_TOTAL}."
        )

    val_group_class_counts = {}

    for c in range(N_CLASSES):
        groups_c = []
        for g in np.unique(g_val):
            labels_g = np.unique(y[groups == g])
            if int(labels_g[0]) == c:
                groups_c.append(int(g))

        val_group_class_counts[CLASS_NAMES[c]] = len(groups_c)

        if len(groups_c) != N_VAL_GROUPS_PER_CLASS:
            raise RuntimeError(
                f"Fold {fold}: validación interna clase {CLASS_NAMES[c]} tiene "
                f"{len(groups_c)} grupos; se esperaban {N_VAL_GROUPS_PER_CLASS}."
            )

    return val_group_class_counts


def step_2_create_folds():
    print("\n" + "=" * 80)
    print("STEP 2 - Crear folds externos, validación interna balanceada y escalado")
    print("=" * 80)

    npz_path = OUT_DIR / "A1_windows_fft_W1024_H512_24ch.npz"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"No encuentro el archivo FFT:\n{npz_path}\n"
            "Primero ejecuta step_1_create_dataset()."
        )

    data = np.load(npz_path, allow_pickle=True)

    X_fft = data["X_fft"].astype(np.float32 if USE_FLOAT32 else np.float64)
    y = data["y"].astype(np.int64)
    groups = data["groups"].astype(np.int64)

    if "sample_index" in data.files:
        sample_index = data["sample_index"].astype(np.int64)
    else:
        sample_index = np.arange(len(y), dtype=np.int64)

    W_local = int(data["W"]) if "W" in data.files else W
    HOP_local = int(data["HOP"]) if "HOP" in data.files else HOP
    n_channels = int(data["n_channels"]) if "n_channels" in data.files else EXPECTED_CHANNELS

    validate_group_purity(y, groups)

    print("\nDATASET LOADED")
    print("X_fft:", X_fft.shape)
    print("y:", y.shape)
    print("groups:", groups.shape)
    print("Unique groups:", len(np.unique(groups)))
    print("Class counts:", class_counts_dict(y))
    print("W:", W_local)
    print("HOP:", HOP_local)
    print("n_channels:", n_channels)

    splitter = build_external_splitter()

    summary_rows = []
    test_indices_all = []

    fold = 0

    for dev_idx, test_idx in splitter.split(X_fft, y, groups):
        fold += 1

        print("\n" + "=" * 80)
        print(f"Fold {fold}/{N_SPLITS}")
        print("=" * 80)

        dev_idx = dev_idx.astype(np.int64)
        test_idx = test_idx.astype(np.int64)

        train_idx, val_idx, val_groups = select_balanced_internal_validation_groups(
            dev_idx=dev_idx,
            y=y,
            groups=groups,
            fold=fold,
        )

        val_group_class_counts = validate_split_integrity(
            fold=fold,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            y=y,
            groups=groups,
        )

        X_train_fft = X_fft[train_idx]
        X_val_fft = X_fft[val_idx]
        X_test_fft = X_fft[test_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

        groups_train = groups[train_idx]
        groups_val = groups[val_idx]
        groups_test = groups[test_idx]

        sample_index_train = sample_index[train_idx]
        sample_index_val = sample_index[val_idx]
        sample_index_test = sample_index[test_idx]

        print("\n[EXTERNAL + INTERNAL SPLIT]")
        print("Train internal samples:", len(train_idx))
        print("Val internal samples  :", len(val_idx))
        print("Test external samples :", len(test_idx))

        print("Train internal groups:", len(np.unique(groups_train)))
        print("Val internal groups  :", len(np.unique(groups_val)))
        print("Test external groups :", len(np.unique(groups_test)))

        print("Train internal class counts:", class_counts_dict(y_train))
        print("Val internal class counts  :", class_counts_dict(y_val))
        print("Test external class counts :", class_counts_dict(y_test))
        print("Val groups per class       :", val_group_class_counts)

        # ------------------------------------------------------------
        # StandardScaler train-only
        # ------------------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train_fft).astype(np.float32)
        X_val_scaled = scaler.transform(X_val_fft).astype(np.float32)
        X_test_scaled = scaler.transform(X_test_fft).astype(np.float32)

        # También guardamos external train completo escalado con scaler de train interno.
        # Esto sirve para auditoría, pero NO para ajustar UMAP si se quiere usar solo train interno.
        X_dev_scaled = scaler.transform(X_fft[dev_idx]).astype(np.float32)

        # ------------------------------------------------------------
        # Guardar archivos por fold
        # ------------------------------------------------------------

        fold_dir = OUT_DIR / f"fold{fold}"
        ensure_dir(fold_dir)

        indices_path = fold_dir / f"fold{fold}_indices.npz"
        data_path = fold_dir / f"fold{fold}_preprocessed_fft.npz"
        scaler_path = fold_dir / f"fold{fold}_fft_scaler.pkl"
        info_path = fold_dir / f"fold{fold}_split_info.json"

        np.savez_compressed(
            indices_path,
            dev_idx=dev_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            val_groups=val_groups,
            groups_dev=groups[dev_idx],
            groups_train=groups_train,
            groups_val=groups_val,
            groups_test=groups_test,
            unique_groups_dev=np.unique(groups[dev_idx]).astype(np.int64),
            unique_groups_train=np.unique(groups_train).astype(np.int64),
            unique_groups_val=np.unique(groups_val).astype(np.int64),
            unique_groups_test=np.unique(groups_test).astype(np.int64),
        )

        np.savez_compressed(
            data_path,

            # Nombres nuevos claros
            X_train_fft=X_train_fft,
            X_val_fft=X_val_fft,
            X_test_fft=X_test_fft,
            X_train_scaled=X_train_scaled,
            X_val_scaled=X_val_scaled,
            X_test_scaled=X_test_scaled,

            y_train=y_train,
            y_val=y_val,
            y_test=y_test,

            groups_train=groups_train,
            groups_val=groups_val,
            groups_test=groups_test,

            sample_index_train=sample_index_train,
            sample_index_val=sample_index_val,
            sample_index_test=sample_index_test,

            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            dev_idx=dev_idx,

            # Alias compatibles con scripts anteriores
            X_tr2_fft=X_train_fft,
            X_va_fft=X_val_fft,
            Xte_fft=X_test_fft,
            X_tr2_s=X_train_scaled,
            X_va_s=X_val_scaled,
            Xte_s=X_test_scaled,
            y_tr2=y_train,
            y_va=y_val,
            yte=y_test,
            g_tr2=groups_train,
            g_va=groups_val,
            gte=groups_test,
            tr2_abs_idx=train_idx,
            va_abs_idx=val_idx,

            # External train completo solo para auditoría
            X_dev_scaled=X_dev_scaled,
            y_dev=y[dev_idx],
            groups_dev=groups[dev_idx],
            sample_index_dev=sample_index[dev_idx],

            # Scaler params
            scaler_mean=scaler.mean_.astype(np.float64),
            scaler_scale=scaler.scale_.astype(np.float64),
            scaler_var=scaler.var_.astype(np.float64),

            W=np.int32(W_local),
            HOP=np.int32(HOP_local),
            n_channels=np.int32(n_channels),
            fold=np.int32(fold),
            seed=np.int32(SEED),
            inner_val_mode=np.array(INNER_VAL_MODE, dtype=object),
            scaler_fit_on=np.array("X_train_fft_internal_only", dtype=object),
            class_names=np.array(CLASS_NAMES, dtype=object),
        )

        save_pickle(scaler, scaler_path)

        info = {
            "fold": int(fold),
            "seed": int(SEED),
            "protocol": {
                "external_split": "StratifiedGroupKFold",
                "n_external_splits": int(N_SPLITS),
                "internal_validation": INNER_VAL_MODE,
                "n_val_groups_per_class": int(N_VAL_GROUPS_PER_CLASS),
                "total_val_groups": int(EXPECTED_VAL_GROUPS_TOTAL),
                "scaler_fit_on": "internal train only",
            },
            "sizes": {
                "n_dev": int(len(dev_idx)),
                "n_train_internal": int(len(train_idx)),
                "n_val_internal": int(len(val_idx)),
                "n_test_external": int(len(test_idx)),
            },
            "groups": {
                "n_dev_groups": int(len(np.unique(groups[dev_idx]))),
                "n_train_groups": int(len(np.unique(groups_train))),
                "n_val_groups": int(len(np.unique(groups_val))),
                "n_test_groups": int(len(np.unique(groups_test))),
                "val_groups": sorted([int(v) for v in np.unique(groups_val)]),
                "test_groups": sorted([int(v) for v in np.unique(groups_test)]),
                "val_group_class_counts": val_group_class_counts,
            },
            "class_counts": {
                "train_internal": class_counts_dict(y_train),
                "val_internal": class_counts_dict(y_val),
                "test_external": class_counts_dict(y_test),
            },
            "shapes": {
                "X_train_fft": list(X_train_fft.shape),
                "X_val_fft": list(X_val_fft.shape),
                "X_test_fft": list(X_test_fft.shape),
                "X_train_scaled": list(X_train_scaled.shape),
                "X_val_scaled": list(X_val_scaled.shape),
                "X_test_scaled": list(X_test_scaled.shape),
            },
            "files": {
                "indices": str(indices_path),
                "data": str(data_path),
                "scaler": str(scaler_path),
                "info": str(info_path),
            },
        }

        save_json(info, info_path)

        print("\nSaved:")
        print(" ", indices_path)
        print(" ", data_path)
        print(" ", scaler_path)
        print(" ", info_path)

        summary_rows.append({
            "fold": fold,
            "n_dev": len(dev_idx),
            "n_train_internal": len(train_idx),
            "n_val_internal": len(val_idx),
            "n_test_external": len(test_idx),
            "n_dev_groups": len(np.unique(groups[dev_idx])),
            "n_train_groups": len(np.unique(groups_train)),
            "n_val_groups": len(np.unique(groups_val)),
            "n_test_groups": len(np.unique(groups_test)),
            "train_class_counts": str(class_counts_dict(y_train)),
            "val_class_counts": str(class_counts_dict(y_val)),
            "test_class_counts": str(class_counts_dict(y_test)),
            "val_group_class_counts": str(val_group_class_counts),
            "indices_file": str(indices_path),
            "data_file": str(data_path),
            "scaler_file": str(scaler_path),
        })

        test_indices_all.extend(test_idx.tolist())

    # ------------------------------------------------------------
    # Consolidar resumen
    # ------------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "cv_scaled_fft_summary_reproducible.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    total_test = len(test_indices_all)
    unique_test = len(set(test_indices_all))
    duplicated_test = total_test - unique_test

    integrity = {
        "n_folds": int(N_SPLITS),
        "total_test_indices": int(total_test),
        "unique_test_indices": int(unique_test),
        "duplicated_test_indices": int(duplicated_test),
        "expected_total_windows": int(len(y)),
        "ok_each_window_once_as_external_test": bool(
            total_test == len(y)
            and unique_test == len(y)
            and duplicated_test == 0
        ),
        "summary_csv": str(summary_path),
        "out_dir": str(OUT_DIR),
    }

    save_json(integrity, OUT_DIR / "cv_integrity_reproducible.json")

    print("\n" + "=" * 80)
    print("FOLDS FINALIZADOS")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\nIntegridad OOF externa:")
    print(json.dumps(integrity, indent=2, ensure_ascii=False))

    if not integrity["ok_each_window_once_as_external_test"]:
        raise RuntimeError(
            "La integridad OOF externa falló: alguna ventana no aparece exactamente una vez en test."
        )

    return summary_df, integrity


# ============================================================
# RUN PRINCIPAL
# ============================================================

def run():
    print("=" * 80)
    print("00_pipeline_01_fft_folds_reproducible")
    print("=" * 80)
    print("PROJECT_DIR:", PROJECT_DIR)
    print("BASE_DIR   :", BASE_DIR)
    print("OUT_DIR    :", OUT_DIR)

    config = {
        "script": "00_pipeline_01_fft_folds_reproducible.py",
        "project_dir": str(PROJECT_DIR),
        "base_dir": str(BASE_DIR),
        "out_dir": str(OUT_DIR),
        "seed": int(SEED),
        "window": int(W),
        "hop": int(HOP),
        "expected_channels": int(EXPECTED_CHANNELS),
        "csv_sep": CSV_SEP,
        "external_cv": "StratifiedGroupKFold",
        "n_splits": int(N_SPLITS),
        "internal_validation": INNER_VAL_MODE,
        "n_val_groups_per_class": int(N_VAL_GROUPS_PER_CLASS),
        "use_float32": bool(USE_FLOAT32),
        "use_fortran_order": bool(USE_FORTRAN_ORDER),
        "class_names": CLASS_NAMES,
        "labels": LABELS,
        "rutas": {k: str(v) for k, v in RUTAS.items()},
    }

    save_json(config, OUT_DIR / "pipeline_01_config_reproducible.json")

    fft_path = OUT_DIR / "A1_windows_fft_W1024_H512_24ch.npz"

    if not fft_path.exists():
        step_1_create_dataset()
    else:
        print("\n[INFO] Ya existe el dataset FFT. Se omite STEP 1:")
        print(fft_path)

    step_2_create_folds()

    print("\n" + "=" * 80)
    print("PIPELINE 01 COMPLETADO ✅")
    print("=" * 80)
    print("Archivos guardados en:")
    print(OUT_DIR)


if __name__ == "__main__":
    warnings.filterwarnings("default")
    run()
