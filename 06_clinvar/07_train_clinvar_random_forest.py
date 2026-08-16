"""Train the ClinVar multi-feature random forest classifiers.

The input contains intronic variants with deletion, shuffle, KangarooGene-EN,
Pangolin and SpliceAI scores. Variants are labelled pathogenic (1) or benign
(0), nonsense variants are removed, and the five-fold split follows the
orthogroup partition. The minority (pathogenic) class is upsampled with
replacement inside each training fold only, so the held-out fold never
contains resampled rows and the evaluation is unbiased.

Seven model types are trained with the configured hyper-parameters.
Predictions are the positive-class probabilities from
predict_proba, and binary accuracy uses the 0.5 threshold.

Usage:
    python 07_train_clinvar_random_forest.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
)

CLINVAR_DIR = "/data1/lty/clinvar2024/new_mut/new_allscan"
SCORE_CSV = os.path.join(
    CLINVAR_DIR,
    "clinvar2024_deleteEN+delete+shuffle+pangolin+spliceai_shuffle_score_filterall.csv",
)
OUT_CSV = os.path.join(CLINVAR_DIR, "clinvar2024_random_forest_pred_out.csv")
MODEL_DIR = os.path.join(CLINVAR_DIR, "final_modelnew")
ACC_FILE = os.path.join(MODEL_DIR, "random_forest_modelacc_multi_5foldup")
PRECISION_FILE = os.path.join(
    MODEL_DIR, "random_forest_modelprec_multi_5foldup"
)
RECALL_FILE = os.path.join(MODEL_DIR, "random_forest_modelrecall_multi_5foldup")
AUPRC_FILE = os.path.join(MODEL_DIR, "random_forest_modelauprc_multi_5foldup")

FEATURE_SETS = [
    ["pangolin_score"],
    ["spliceai_score"],
    ["pangolin_score", "spliceai_score"],
    [f"shuffle_change{i}" for i in range(1, 6)],
    [f"dele_en_change{i}" for i in range(1, 6)],
    [f"dele_en_change{i}" for i in range(1, 6)] +
    [f"shuffle_change{i}" for i in range(1, 6)],
    [f"dele_en_change{i}" for i in range(1, 6)] +
    [f"shuffle_change{i}" for i in range(1, 6)] +
    ["pangolin_score", "spliceai_score"],
]

MODEL_NAMES = [
    "pangolin",
    "spliceai",
    "pangolin+spliceai",
    "shuffle5",
    "dele_en5",
    "dele_en5+shuffle5",
    "dele_en5+shuffle5+pangolin+spliceai",
]


def make_rf():
    """Return the random forest configuration."""
    return RandomForestClassifier(
        n_estimators=624,
        max_depth=16,
        max_features=0.2,
        max_samples=0.4779,
        min_samples_leaf=4,
        n_jobs=20,
        bootstrap=True,
    )


def write_metric(path, rows):
    """Write the two-column model_fold/value format."""
    with open(path, "w", encoding="utf-8") as handle:
        for model_name, fold, value in rows:
            handle.write(f"{model_name}_{fold}\t{value}\n")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    score_df = pd.read_csv(SCORE_CSV)
    score_df["label"] = 1
    score_df.loc[score_df["pheno"] == "benign", "label"] = 0
    score_df = score_df[score_df["mut_type"] == "intron_variant"]
    score_df = score_df[score_df["gene_name"] == score_df["mut_gene_name"]]
    score_df = score_df[~score_df["INFO"].astype(str).str.contains(
        "nonsense", na=False
    )]
    score_df = score_df.dropna(subset=["label"]).reset_index(drop=True)

    for i in range(1, 6):
        score_df[f"change{i}"] = score_df[f"change{i}"].abs()
        score_df[f"shuffle_change{i}"] = score_df[f"shuffle_change{i}"].abs()

    all_pred = []
    metric_rows = {"acc": [], "precision": [], "recall": [], "auprc": []}
    for fold in range(1, 6):
        train_full = score_df[score_df["group_type"] != fold].sample(
            frac=1
        ).reset_index(drop=True)
        test_data = score_df[score_df["group_type"] == fold].sample(
            frac=1
        ).reset_index(drop=True)
        benign = train_full[train_full["label"] == 0]
        pathogenic = train_full[train_full["label"] == 1]
        if len(pathogenic) == 0 or len(benign) == 0:
            print(f"fold {fold}: only one class, skipping")
            continue
        # Upsample only inside the training fold to avoid data leakage.
        pathogenic_up = pathogenic.sample(n=len(benign), replace=True)
        train_data = pd.concat(
            [benign, pathogenic_up], ignore_index=True
        ).sample(frac=1).reset_index(drop=True)

        for features, model_name in zip(FEATURE_SETS, MODEL_NAMES):
            model = make_rf()
            model.fit(train_data[features], train_data["label"])
            model_path = os.path.join(
                MODEL_DIR, f"random_forest_{model_name}_{fold}_newdata.pkl"
            )
            with open(model_path, "wb") as handle:
                pickle.dump(model, handle)

            proba = model.predict_proba(test_data[features])[:, 1]
            test_data["pred_out"] = proba
            test_data["model_type"] = model_name
            test_data["label_keep"] = test_data["label"]
            all_pred.append(test_data.copy())

            pred_label = (proba >= 0.5).astype(int)
            true_label = test_data["label"].to_numpy()
            acc = float(np.mean(pred_label == true_label))
            precision = float(
                np.sum((pred_label == 1) & (true_label == 1)) /
                max(np.sum(pred_label == 1), 1)
            )
            recall = float(
                np.sum((pred_label == 1) & (true_label == 1)) /
                max(np.sum(true_label == 1), 1)
            )
            auprc = float(average_precision_score(true_label, proba))
            metric_rows["acc"].append((model_name, fold, acc))
            metric_rows["precision"].append((model_name, fold, precision))
            metric_rows["recall"].append((model_name, fold, recall))
            metric_rows["auprc"].append((model_name, fold, auprc))
            print(
                f"fold {fold} {model_name}: acc {acc:.4f}, "
                f"precision {precision:.4f}, recall {recall:.4f}, "
                f"auprc {auprc:.4f}"
            )

    pred_out = pd.concat(all_pred, ignore_index=True)
    pred_out["label"] = pred_out["label_keep"]
    pred_out = pred_out.drop(columns=["label_keep"])
    pred_out.to_csv(OUT_CSV, index=False)

    write_metric(ACC_FILE, metric_rows["acc"])
    write_metric(PRECISION_FILE, metric_rows["precision"])
    write_metric(RECALL_FILE, metric_rows["recall"])
    write_metric(AUPRC_FILE, metric_rows["auprc"])
    print(f"saved: {OUT_CSV}")
    print(f"saved models: {MODEL_DIR}")
    print(f"saved metrics: {ACC_FILE}, {PRECISION_FILE}, {RECALL_FILE}, {AUPRC_FILE}")


if __name__ == "__main__":
    main()
