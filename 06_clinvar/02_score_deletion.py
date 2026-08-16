"""Score ClinVar intronic variants with the five KangarooGene deletion models.

The wild-type transcript is scored with all five orthogroup folds, and every
intronic variant is scored after substituting the ClinVar allele at mut_loc.
 The per-transcript prediction block is assigned back to the corresponding 
 result rows. The merged output is consumed by annotate_deletion.py.

Usage:
    python 02_score_deletion.py
"""

import os

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

SEQ_MAX = 300000
EACH_BATCH = 80
MODEL_BATCH = 32

ANIMAL_DELETED_CSV = "/data1/lty/intron_pred/data/animal_deleted_all.csv"
ORTHO_CSV = "/data1/lty/intron_pred/data/orth40spe_out.csv"
TX_INFO_CSV = "/data1/lty/clinvar2024/homo_tx_info.csv"
TX_SEQS_CSV = "/data1/lty/clinvar2024/human_tx_seqs.csv"
VARIANT_CSV = "/data1/lty/clinvar2024/new_mut/new_allscan/clinvar2024_intron_mut_allchr.csv"
OUT_CSV = "/data1/lty/clinvar2024/new_mut/new_allscan/homo_clinvar_all_snp_delete_animalmodel_finalall.csv"

MODEL_PREFIX = "/data1/lty/intron_pred/new_model/final_model/animal_random_delete1_final_group"

# Species whose orthogroup gene ids carry a transcript-style ".x" suffix.
CHANGE_SPE = [
    "Bta", "Chi", "Cin", "Dre", "Fca", "Gga", "Homo",
    "Mmu", "Nna", "Ptro", "Ssa", "Xtr",
]

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


def predict_wild_scores(model, seqs_data, seq_max, each_batch, model_batch):
    """Score every wild-type transcript with one fold model."""
    score_wildall = []
    for j in range((len(seqs_data) // each_batch) + 1):
        use_matrix = np.array([
            one_hot_encoding(i, seq_max)
            for i in seqs_data["seqs"].iloc[j * each_batch:(j + 1) * each_batch]
        ])
        pred_out = model.predict(use_matrix, batch_size=model_batch)
        score_wildall.extend(pd.DataFrame(pred_out)[1].tolist())
    return score_wildall


def load_div_data():
    """Build the gene -> group_type mapping used by the deletion models."""
    div_dataall = pd.read_csv(ANIMAL_DELETED_CSV)
    div_dataall = div_dataall[div_dataall["spe"] == "homo"]

    group_type = pd.read_csv(ORTHO_CSV)
    mask = group_type["spe"].isin(CHANGE_SPE)
    group_type.loc[mask, "gene"] = group_type.loc[mask, "gene"].apply(
        lambda x: x.split(".")[0] if "." in x else x
    )
    group_type = group_type[["gene", "group_type"]]

    div_data = group_type[group_type["gene"].isin(div_dataall["gene_id"])]
    tx_gene_data = pd.read_csv(TX_INFO_CSV)
    tx_gene_data = tx_gene_data[["tx_name", "gene_id"]]
    tx_gene_data.columns = ["tx_name", "gene"]
    div_data = pd.merge(div_data, tx_gene_data, on="gene")
    return div_data


def main():
    div_data = load_div_data()

    seqs_data = pd.read_csv(TX_SEQS_CSV)
    seqs_data = seqs_data[~seqs_data["seqs"].isna()]
    seqs_data = seqs_data[seqs_data["width"] <= SEQ_MAX]
    seqs_data = seqs_data[~seqs_data["seqs"].str.contains("N")]
    seqs_data = pd.merge(seqs_data, div_data, on="tx_name")
    seqs_data = seqs_data.reset_index(drop=True)

    results_intronall = pd.read_csv(VARIANT_CSV)
    results_intronall = results_intronall[results_intronall["POS"] != "POS"]
    results_intronall = results_intronall[results_intronall["REF"].str.len() == 1]
    results_intronall = results_intronall[results_intronall["ALT"].str.len() == 1]
    results_intronall = results_intronall[results_intronall["REF"].isin(["A", "G", "C", "T"])]
    results_intronall = results_intronall[results_intronall["ALT"].isin(["A", "G", "C", "T"])]

    results_intronout = results_intronall[
        results_intronall["tx_id"].isin(seqs_data["tx_name"])
    ]
    results_intronout["mut_loc"] = results_intronout["mut_loc"].astype(int)

    for fold in range(1, 6):
        model_path = MODEL_PREFIX + str(fold)
        model = load_model(model_path)

        # Wild-type scores for this fold.
        seqs_data["wild_score" + str(fold)] = predict_wild_scores(
            model, seqs_data, SEQ_MAX, EACH_BATCH, MODEL_BATCH
        )

        # Mutant scores, transcript by transcript.
        col_id = "score_mut" + str(fold)
        results_intronout[col_id] = "NA"
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
                pred_score.extend(
                    pd.DataFrame(
                        model.predict(use_matrix, batch_size=MODEL_BATCH)
                    )[1].tolist()
                )

            # Assign successfully predicted scores back to the whole
            # transcript block by position.
            results_intronout.loc[results_intronout["tx_id"] == tx, col_id] = pred_score
            if count % 20 == 0:
                del model
                model = load_model(model_path)
            print(count)

    # Wild scores from all five folds. The confirmed target keeps
    # wild_score1-5 so the downstream change1..5 columns are all defined.
    seqs_data_use = seqs_data[
        ["tx_name"] + ["wild_score" + str(i) for i in range(1, 6)]
    ]
    results_intronouta = pd.merge(
        results_intronout,
        seqs_data_use,
        left_on="tx_id",
        right_on="tx_name",
        how="left",
    )
    results_intronouta.to_csv(OUT_CSV, index=False)
    print("saved:", OUT_CSV)


if __name__ == "__main__":
    main()
