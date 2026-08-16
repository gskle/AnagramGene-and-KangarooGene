"""Score ClinVar intronic variants with the five dual-input PI models.

The paired model receives the wild-type transcript on the left input and the
allele-substituted transcript on the right input. All five orthogroup folds
are scored, and the SI table (homo_clinvar_all_snp_step2.csv) supplies the
variant rows and the wild-type scores. The PI mutant scores overwrite the
score_mut columns of that table, which is then written directly as
homo_clinvar_dualinput_all_snp_step2.csv. A second table adds the
snp/mut_type/pheno/group_type columns needed by the random-forest script.

Usage:
    python 05_score_pi.py
"""

import os

import numpy as np
import pandas as pd
import keras.backend as K
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEQ_MAX = 300000
EACH_BATCH = 80
MODEL_BATCH = 32

INTROS_HUFFLE_CSV = "/data1/lty/intron_pred/data/animal14_intronshuffle.csv"
TX_INFO_CSV = "/data1/lty/clinvar2024/homo_tx_info.csv"
TX_SEQS_CSV = "/data1/lty/clinvar2024/human_tx_seqs.csv"
SI_SCORE_CSV = (
    "/data1/lty/clinvar2024/new_mut/new_allscan/homo_clinvar_all_snp_step2.csv"
)
VCF_INFO_CSV = (
    "/data1/lty/clinvar2024/new_mut/new_allscan/"
    "clinvar2024_delete+shuffle+pangolin+spliceai_shuffle_score_filterall.csv"
)
OUT_CSV = (
    "/data1/lty/clinvar2024/new_mut/new_allscan/"
    "homo_clinvar_dualinput_all_snp_step2.csv"
)
OUT_RF_CSV = (
    "/data1/lty/clinvar2024/new_mut/new_allscan/"
    "homo_clinvar_dualinput_all_snp_for_rf.csv"
)

MODEL_PREFIX = (
    "/data1/lty/intron_pred/new_model/final_model/animal14spe_30wbp_group"
)

NTS = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}
POS_NEG_DICT = {"A": "T", "G": "C", "C": "G", "T": "A"}


def one_hot_encoding(seq, seq_max):
    """Encode one DNA string as a (seq_max, 4) matrix."""
    one_hot_encoded = np.zeros(shape=(seq_max, 4))
    for i, nt in enumerate(seq):
        one_hot_encoded[i, NTS[nt]] = 1
    return one_hot_encoded


def mutate_intronseqs(seqs, wild_base, mut_base, mut_loc, strand):
    """Build mutant sequences, skipping rows whose reference base mismatches."""
    out_seqs = []
    for i in range(len(mut_base)):
        if strand == "-":
            use_wildbase = POS_NEG_DICT[wild_base[i]]
            use_mutbase = POS_NEG_DICT[mut_base[i]]
        else:
            use_wildbase = wild_base[i]
            use_mutbase = mut_base[i]
        mut_loc_one = mut_loc[i] - 1
        if seqs[mut_loc_one] != use_wildbase:
            print("base error")
        else:
            one_seqs = seqs[:mut_loc_one] + use_mutbase + seqs[mut_loc[i]:]
            out_seqs.append(one_seqs)
    return out_seqs


def load_div_data():
    """Build the tx_name -> group_type mapping used by the PI models."""
    div_dataall = pd.read_csv(INTROS_HUFFLE_CSV)
    div_dataall = div_dataall[div_dataall["spe"] == "Homo"]
    div_data = div_dataall[["gene", "group_type"]]

    tx_gene_data = pd.read_csv(TX_INFO_CSV)
    tx_gene_data = tx_gene_data[["tx_name", "gene_id"]]
    tx_gene_data.columns = ["tx_name", "gene"]
    div_data = pd.merge(div_data, tx_gene_data, on="gene")
    return div_data


def predict_mutant_scores(model, model_path, results_intronout, seqs_data, col_id):
    """Score paired (wild, mutant) sequences transcript by transcript."""
    tx_all = results_intronout["tx_id"].unique().tolist()
    count = 0
    for tx in tx_all:
        count += 1
        subresults = results_intronout[results_intronout["tx_id"] == tx]
        seqs_one = seqs_data[seqs_data["tx_name"] == tx]
        gene_seq = seqs_one["seqs"].iloc[0]
        strand = seqs_one["strand"].iloc[0]

        mut_seqs = mutate_intronseqs(
            gene_seq,
            subresults["REF"].tolist(),
            subresults["ALT"].tolist(),
            subresults["mut_loc"].tolist(),
            strand,
        )
        pred_score = []
        if len(mut_seqs) % EACH_BATCH != 0:
            use_epoch = (len(mut_seqs) // EACH_BATCH) + 1
        else:
            use_epoch = len(mut_seqs) // EACH_BATCH
        for j in range(use_epoch):
            use_matrix = np.array([
                one_hot_encoding(i, SEQ_MAX)
                for i in mut_seqs[j * EACH_BATCH:(j + 1) * EACH_BATCH]
            ])
            natural_mt = np.array([one_hot_encoding(gene_seq, SEQ_MAX)])
            natural_mt = np.repeat(natural_mt, len(use_matrix), axis=0)
            pred_out = model.predict(
                {"seqs_left": natural_mt, "seqs_right": use_matrix},
                batch_size=MODEL_BATCH,
            )
            pred_score.extend(pd.DataFrame(pred_out)[1].tolist())

        # Assign back to the whole transcript block by position.
        results_intronout.loc[results_intronout["tx_id"] == tx, col_id] = pred_score
        if count % 20 == 0:
            del model
            K.clear_session()
            model = load_model(model_path)
        print(count)
    return model


def main():
    div_data = load_div_data()

    seqs_data = pd.read_csv(TX_SEQS_CSV)
    seqs_data = seqs_data[~seqs_data["seqs"].isna()]
    seqs_data = seqs_data[seqs_data["width"] <= SEQ_MAX]
    seqs_data = seqs_data[~seqs_data["seqs"].str.contains("N")]
    seqs_data = pd.merge(seqs_data, div_data, on="tx_name")
    seqs_data = seqs_data.reset_index(drop=True)

    results_intronout = pd.read_csv(SI_SCORE_CSV)
    results_intronout["mut_loc"] = results_intronout["mut_loc"].astype(int)

    # Keep only the intronic variants that survive the four-model merge,
    # using the snp key (#CHROM_POS_REF_ALT).
    vcf_info = pd.read_csv(VCF_INFO_CSV)
    vcf_info = vcf_info[vcf_info["mut_type"] == "intron_variant"]
    results_intronout["snp"] = (
        results_intronout["#CHROM"].astype(str) + "_" +
        results_intronout["POS"].astype(str) + "_" +
        results_intronout["REF"] + "_" +
        results_intronout["ALT"]
    )
    results_intronout = results_intronout[
        results_intronout["snp"].isin(vcf_info["snp"])
    ].reset_index(drop=True)

    for fold in range(1, 6):
        model_path = MODEL_PREFIX + str(fold)
        model = load_model(model_path)

        col_id = "score_mut" + str(fold)
        model = predict_mutant_scores(
            model, model_path, results_intronout, seqs_data, col_id
        )
        del model
        K.clear_session()
        print("fold", fold, "done")

    # The SI table already carries the tx-level wild_score1-5 columns, so no
    # re-merge is needed. If they are missing, fill them from a tx-level copy
    # of the SI table using the tx_id -> tx_name key.
    wild_cols = ["wild_score" + str(i) for i in range(1, 6)]
    missing_wild = [c for c in wild_cols if c not in results_intronout.columns]
    if missing_wild:
        si_wild = pd.read_csv(SI_SCORE_CSV)[["tx_id"] + missing_wild]
        si_wild = si_wild.drop_duplicates("tx_id")
        results_intronout = pd.merge(
            results_intronout,
            si_wild,
            on="tx_id",
            how="left",
            suffixes=("", "_fallback"),
        )
        for c in missing_wild:
            results_intronout[c] = results_intronout[c].fillna(
                results_intronout[c + "_fallback"]
            )
        results_intronout = results_intronout.drop(
            columns=[c + "_fallback" for c in missing_wild]
        )
    results_intronout.to_csv(OUT_CSV, index=False)
    print("saved:", OUT_CSV)

    # Add the variant-level columns consumed by the random-forest script.
    results_intronout["snp"] = (
        results_intronout["#CHROM"].astype(str) + "_" +
        results_intronout["POS"].astype(str) + "_" +
        results_intronout["REF"] + "_" +
        results_intronout["ALT"]
    )
    vcf_info = vcf_info[["snp", "mut_type", "pheno", "group_type"]]
    for_rf = pd.merge(vcf_info, results_intronout, on="snp")
    for_rf.to_csv(OUT_RF_CSV, index=False)
    print("saved:", OUT_RF_CSV)


if __name__ == "__main__":
    main()
