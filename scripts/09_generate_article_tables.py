# -*- coding: utf-8 -*-
"""Generate final manuscript result tables."""

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("WILEY_PROJECT_DIR", Path(__file__).resolve().parents[1]))
OUT = ROOT / "13_TABLAS_FINALES_WILEY"
S08 = ROOT / "08_CONSOLIDACION_FINAL_WILEY"
S09 = ROOT / "09_AUDITORIA_EXPERIMENTOS_WILEY"
S11 = ROOT / "11_DISPERSION_SEPARABILIDAD_WILEY"
S12D = ROOT / "12_FIGURAS_FINALES_WILEY" / "06_FIGURAS_RECUPERADAS"

EXPECTED_ERRORS = {
    ("PUMAP", "ExtraTrees"): 50, ("PUMAP", "CatBoost"): 50,
    ("PUMAP", "RandomForest"): 51, ("PUMAP", "LightGBM"): 52,
    ("PUMAP", "XGBoost"): 55, ("PCA", "XGBoost"): 60,
    ("PCA", "RandomForest"): 68, ("PCA", "CatBoost"): 72,
    ("PCA", "LightGBM"): 78, ("PCA", "ExtraTrees"): 81,
}
FOLD_SIZES = [769, 768, 768, 766, 769]
FAMILIES = {"PCA": "PCA", "PUMAP": "Parametric UMAP"}


def read_checked(path, required, title):
    if not path.is_file():
        raise FileNotFoundError(f"[{title}] No existe: {path}. Ejecuta primero el script correspondiente.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"[{title}] Faltan columnas {missing} en {path.name}. Disponibles: {df.columns.tolist()}")
    print(f"[OK] {title}: {len(df)} filas ({path.name})")
    return df


def check_keys(df, columns, title):
    if df.duplicated(columns).any():
        raise ValueError(f"[{title}] Filas duplicadas para {columns}.")


def tex_escape(value):
    text = str(value)
    table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
             "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
             "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(table.get(c, c) for c in text)


def latex_table(df, filename, caption, label, digits=4, column_labels=None):
    columns = list(df.columns)
    if column_labels is None:
        column_labels = columns
    if len(column_labels) != len(columns):
        raise ValueError("Numero de etiquetas LaTeX diferente al de columnas")
    n = len(columns)
    align = "l" + "r" * (n - 1)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{" + tex_escape(caption) + "}", r"\label{" + label + "}",
             r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(column_labels) + " " + "\\\\", r"\midrule"]
    for row in df.itertuples(index=False, name=None):
        pieces = []
        for val in row:
            if pd.isna(val):
                pieces.append("--")
            elif isinstance(val, (float, np.floating)):
                pieces.append(f"{float(val):.{digits}f}")
            elif isinstance(val, (int, np.integer)):
                pieces.append(str(int(val)))
            else:
                pieces.append(tex_escape(val))
        lines.append(" & ".join(pieces) + " " + "\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_csv(df, filename):
    df.to_csv(OUT / filename, index=False, encoding="utf-8-sig")


def family_name(df):
    return df["representation_family"].replace(FAMILIES)


def table_global():
    df = read_checked(S08 / "TABLA_MAESTRA_10_PIPELINES_OOF.csv",
                      ["representation_family", "classifier", "n_oof", "errors", "accuracy", "balanced_accuracy", "macro_f1", "mcc", "kappa"],
                      "Rendimiento OOF")
    check_keys(df, ["representation_family", "classifier"], "Rendimiento OOF")
    if len(df) != 10:
        raise ValueError("La tabla OOF debe contener exactamente 10 pipelines")
    keys = set(zip(df.representation_family, df.classifier))
    if keys != set(EXPECTED_ERRORS):
        raise ValueError(f"Pipelines incorrectos: faltantes={set(EXPECTED_ERRORS)-keys}; extras={keys-set(EXPECTED_ERRORS)}")
    for row in df.itertuples():
        pair = (row.representation_family, row.classifier)
        if int(row.n_oof) != 3840 or int(row.errors) != EXPECTED_ERRORS[pair]:
            raise ValueError(f"Inconsistencia en OOF {pair}: N={row.n_oof}, errores={row.errors}")
    df = df.sort_values(["macro_f1", "mcc"], ascending=False).copy()
    display = pd.DataFrame({"Representation": family_name(df), "Classifier": df.classifier,
        "Accuracy": df.accuracy, "Balanced accuracy": df.balanced_accuracy,
        "Macro-F1": df.macro_f1, "MCC": df.mcc, "Kappa": df.kappa, "Errors": df.errors.astype(int)})
    save_csv(display, "TABLE_01_GLOBAL_OOF.csv")
    latex_table(display, "TABLE_01_GLOBAL_OOF.tex",
       "External out-of-fold performance of ten representation-classifier pipelines (N = 3,840 windows).",
       "tab:global_oof", column_labels=["Representation", "Classifier", "Accuracy", "Bal. acc.", "Macro-F1", "MCC", "Kappa", "Errors"])
    return df


def table_internal():
    path = S12D / "fig08_selected_internal_macrof1_data.csv"
    df = read_checked(path, ["representation", "classifier", "fold", "selected_internal_macro_f1"], "Validacion interna")
    df["representation"] = df["representation"].replace({"Parametric UMAP": "PUMAP"})
    check_keys(df, ["representation", "classifier", "fold"], "Validacion interna")
    if len(df) != 20 or set(df.fold.astype(int)) != set(range(1, 6)):
        raise ValueError("Se esperan 4 pipelines por 5 folds internos")
    wide = df.pivot(index=["representation", "classifier"], columns="fold", values="selected_internal_macro_f1")
    if wide.isna().any().any():
        raise ValueError("Faltan valores en algun fold de validacion interna")
    wide = wide.reset_index()
    wide.columns = ["Representation", "Classifier", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
    wide["Representation"] = wide["Representation"].replace(FAMILIES)
    save_csv(wide, "TABLE_02_SELECTED_INTERNAL_MACROF1.csv")
    latex_table(wide, "TABLE_02_SELECTED_INTERNAL_MACROF1.tex",
        "Selected internal-validation Macro-F1 for four representative pipelines in each external fold. Test folds were not used for hyperparameter selection.",
        "tab:internal_macro_f1", column_labels=["Representation", "Classifier", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"])


def table_external_folds():
    df = read_checked(S08 / "METRICAS_POR_FOLD.csv",
       ["representation_family", "classifier", "fold", "n_test", "errors", "accuracy", "macro_f1", "mcc"],
       "Metricas externas por fold")
    check_keys(df, ["representation_family", "classifier", "fold"], "Metricas externas por fold")
    if len(df) != 50:
        raise ValueError("Se esperan 50 filas: diez pipelines por cinco folds")
    for row in df.itertuples():
        if int(row.fold) not in range(1, 6) or int(row.n_test) != FOLD_SIZES[int(row.fold) - 1]:
            raise ValueError("Tamano de fold incorrecto en tabla externa")
    display = pd.DataFrame({"Representation": family_name(df), "Classifier": df.classifier,
       "Fold": df.fold.astype(int), "Test windows": df.n_test.astype(int), "Errors": df.errors.astype(int),
       "Accuracy": df.accuracy, "Macro-F1": df.macro_f1, "MCC": df.mcc})
    display = display.sort_values(["Representation", "Classifier", "Fold"])
    save_csv(display, "TABLE_03_EXTERNAL_FOLD_DETAILS.csv")
    latex_table(display, "TABLE_03_EXTERNAL_FOLD_DETAILS.tex",
       "External-test metrics for all ten pipelines across the five group-disjoint outer folds.",
       "tab:external_fold_results", column_labels=["Representation", "Classifier", "Fold", "N", "Errors", "Accuracy", "Macro-F1", "MCC"])
    # Tabla compacta: numero de errores por fold; recomendada para manuscrito principal
    errors = df.pivot(index=["representation_family", "classifier"], columns="fold", values="errors")
    errors = errors.reset_index()
    errors.columns = ["Representation", "Classifier", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
    errors["Representation"] = errors["Representation"].replace(FAMILIES)
    errors["Total errors"] = errors[[f"Fold {i}" for i in range(1, 6)]].sum(axis=1)
    save_csv(errors, "TABLE_03B_ERRORS_BY_FOLD_COMPACT.csv")
    latex_table(errors, "TABLE_03B_ERRORS_BY_FOLD_COMPACT.tex",
       "Misclassified external-test windows by fold for the ten pipelines.",
       "tab:errors_by_fold", column_labels=["Representation", "Classifier", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Total"])
    return df


def table_separability():
    df = read_checked(S11 / "RESUMEN_SEPARABILIDAD_5_FOLDS.csv",
       ["representation", "silhouette_mean", "silhouette_std", "db_mean", "db_std", "margin_mean", "negative_margin_mean_pct"],
       "Separabilidad geometrica")
    if set(df.representation) != {"PCA", "PUMAP"} or len(df) != 2:
        raise ValueError("Se esperan resultados geometricos de PCA y PUMAP")
    display = pd.DataFrame({"Representation": df.representation.replace(FAMILIES),
       "Silhouette mean": df.silhouette_mean, "Silhouette SD": df.silhouette_std,
       "DB mean": df.db_mean, "DB SD": df.db_std,
       "Relative margin": df.margin_mean, "Negative margins (%)": df.negative_margin_mean_pct})
    save_csv(display, "TABLE_04_GEOMETRIC_SEPARABILITY.csv")
    latex_table(display, "TABLE_04_GEOMETRIC_SEPARABILITY.tex",
       "Descriptive 32-dimensional latent-space geometry computed on external-test embeddings, aggregated over five independently fitted folds.",
       "tab:geometric_separability", column_labels=["Representation", "Silhouette", "SD", "DB", "SD", "Rel. margin", "Neg. (\\%)"])
    fold = read_checked(S11 / "INDICES_SEPARABILIDAD_TEST_POR_FOLD.csv",
       ["representation", "fold", "silhouette_test_32d", "davies_bouldin_test_32d", "mean_relative_margin", "negative_margin_pct"],
       "Geometria por fold")
    check_keys(fold, ["representation", "fold"], "Geometria por fold")
    displayfold = fold[["representation", "fold", "silhouette_test_32d", "davies_bouldin_test_32d", "mean_relative_margin", "negative_margin_pct"]].copy()
    displayfold.columns = ["Representation", "Fold", "Silhouette", "Davies-Bouldin", "Relative margin", "Negative margins (%)"]
    displayfold["Representation"] = displayfold["Representation"].replace(FAMILIES)
    save_csv(displayfold, "TABLE_04B_GEOMETRY_BY_FOLD.csv")
    latex_table(displayfold, "TABLE_04B_GEOMETRY_BY_FOLD.tex",
       "External-test latent-space geometry for each independently trained representation and fold (32 dimensions).",
       "tab:geometry_by_fold", column_labels=["Representation", "Fold", "Silhouette", "DB", "Rel. margin", "Neg. (\\%)"])


def table_problematic_experiments():
    df = read_checked(S11 / "DETALLE_GRUPOS_20_60.csv",
       ["representation", "fold", "group_original", "class_name", "n", "mean_margin", "negative_margin_pct", "nearest_centroid_accuracy_pct"],
       "Grupos 20 y 60")
    check_keys(df, ["representation", "group_original"], "Grupos 20 y 60")
    if len(df) != 4 or set(df.group_original) != {20, 60}:
        raise ValueError("Se esperan 4 filas: PCA/PUMAP por cada grupo 20/60")
    display = df[["representation", "group_original", "class_name", "fold", "n", "mean_margin", "negative_margin_pct", "nearest_centroid_accuracy_pct"]].copy()
    display.columns = ["Representation", "Experiment", "True class", "Fold", "N", "Mean relative margin", "Negative margins (%)", "Nearest-centroid accuracy (%)"]
    display["Representation"] = display["Representation"].replace(FAMILIES)
    display = display.sort_values(["Experiment", "Representation"])
    save_csv(display, "TABLE_05_EXPERIMENTS_20_60.csv")
    latex_table(display, "TABLE_05_EXPERIMENTS_20_60.tex",
       "Post-hoc nearest-training-centroid analysis of experiments 20 and 60 in external fold 2 (32 dimensions).",
       "tab:groups_20_60", column_labels=["Representation", "Group", "Class", "Fold", "N", "Mean margin", "Neg. (\\%)", "NC acc. (\\%)"])

    # errores de clasificador para los dos grupos (diferentes de errores de centroides)
    audit = read_checked(S09 / "METRICAS_80_EXPERIMENTOS_X_10_PIPELINES.csv",
       ["representation", "classifier", "group_original", "errors"], "Errores OOF experimentales")
    details = audit[audit["group_original"].isin([20, 60])].copy()
    if len(details) != 20:
        raise ValueError(f"Se esperaban 20 combinaciones (10 pipelines x 2 grupos); encontradas: {len(details)}")
    fields = ["representation", "classifier", "group_original", "errors"]
    extra = [x for x in ["windows", "accuracy_experiment"] if x in details.columns]
    details = details[fields + extra].sort_values(["group_original", "representation", "classifier"])
    save_csv(details, "TABLE_05B_CLASSIFIER_ERRORS_EXPERIMENTS_20_60.csv")
    latex_table(details, "TABLE_05B_CLASSIFIER_ERRORS_EXPERIMENTS_20_60.tex",
       "External OOF classification errors on experimental groups 20 and 60, distinguished from nearest-centroid errors.",
       "tab:group_specific_classifier_errors")


def table_feature_importances():
    path = S12D / "fig15_latent_importance_mean_std.csv"
    df = read_checked(path, [], "Importancia media de dimensiones latentes")
    save_csv(df, "TABLE_06_LATENT_IMPORTANCES_REFERENCE.csv")
    # Las importancias por dimension son material auxiliar, la figura 15 las presenta.
    return len(df)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    global_df = table_global()
    table_internal()
    fold_df = table_external_folds()
    table_separability()
    table_problematic_experiments()
    n_importance = table_feature_importances()
    summary = {"n_pipelines": 10, "n_oof_per_pipeline": 3840,
               "n_outer_folds": 5, "latent_dimensions": 32,
               "feature_importance_rows": n_importance,
               "pipeline_errors": {f"{row.representation_family}/{row.classifier}": int(row.errors)
                                   for row in global_df.itertuples()},
               "notes": ["No models retrained", "External OOF scores are distinct from selected internal validation metrics",
                         "Latent geometry computed post hoc on test representations", "All 32D representations were independently trained per fold"]}
    (OUT / "AUDIT_MANIFEST.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 80)
    print("TABLAS WILEY COMPLETADAS")
    print("=" * 80)
    print("Pipelines auditados: 10/10; predicciones externas: 3840 por pipeline")
    print(f"Archivos .tex: {len(list(OUT.glob('*.tex')))}")
    print(f"Archivos .csv: {len(list(OUT.glob('*.csv')))}")
    print("Carpeta:", OUT)
    print("IMPORTANTE: cada .tex usa \\toprule, \\midrule y \\bottomrule (paquete booktabs).")
    print("Revisar primero TABLE_01_GLOBAL_OOF.tex y TABLE_03B_ERRORS_BY_FOLD_COMPACT.tex")


if __name__ == "__main__":
    main()
