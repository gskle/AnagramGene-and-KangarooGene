"""Annotate the KangarooGene deletion ClinVar scores with ClinVar labels.

CLNSIG flags are added from the ClinVar VCF,
the deletion score table is merged with those flags, the ANN= fields are
parsed for the effect type, and only benign/pathogenic intronic variants
are kept. The per-fold wild - mutant changes and their mean are computed.

Usage:
    python 03_annotate_deletion.py
"""

import pandas as pd

CLINVAR_VCF = "/data1/lty/clinvar2024/clinvar_20240624.vcf"
CLINVAR_ANNOT_VCF = "/data1/lty/clinvar2024/clinvar_20240624.annot.vcf"
CLNSIG_CSV = "/data1/lty/clinvar2024/homo_clinvar_CLNSIG_annot.csv"
DELETION_SCORE_CSV = (
    "/data1/lty/clinvar2024/new_mut/new_allscan/"
    "homo_clinvar_all_snp_delete_animalmodel_finalall.csv"
)
DELETION_ADD_TYPE_CSV = (
    "/data1/lty/clinvar2024/new_mut/new_allscan/"
    "homo_clinvar_all_snp_delete_animalmodel_finalall_addtype.csv"
)
OUT_CSV = (
    "/data1/lty/clinvar2024/new_mut/new_allscan/"
    "clinvar2024_delete_newanimal_div1_step2_annot_photo.csv"
)

# Column name -> INFO flag text.
CLNSIG_FLAGS = [
    ("benign", "Benign"),
    ("Likely_benign", "Likely_benign"),
    ("Pathogenic", "Pathogenic"),
    ("Likely_pathogenic", "Likely_pathogenic"),
    ("Conflicting", "Conflicting"),
    ("Uncertain", "Uncertain"),
]


def snp_from_vcf(df):
    """Build the snp id used throughout the ClinVar pipeline."""
    return (
        df["#CHROM"].astype(str) + "_" +
        df["POS"].astype(str) + "_" +
        df["ID"].astype(str) + "_" +
        df["REF"] + "_" +
        df["ALT"]
    )


def read_annot_vcf(path):
    """Read a VCF, skipping ## comment lines."""
    with open(path) as fh:
        lines = [
            line.strip()
            for line in fh
            if line.strip() and not line.startswith("##")
        ]
    header = lines[0].lstrip("#").split("\t")
    return pd.DataFrame([line.split("\t") for line in lines[1:]], columns=header)


def add_clnsig_flags(df):
    """Add one-hot CLNSIG columns from the INFO field."""
    for col, flag in CLNSIG_FLAGS:
        df[col] = [1 if "CLNSIG=" + flag in i else 0 for i in df["INFO"]]
    return df


def parse_effect_info(eff_info, tx_id):
    """Return the ANN= effect type matching the transcript id."""
    if not isinstance(eff_info, str) or not eff_info:
        return ""
    ann_fields = eff_info.split(",")
    match = [f for f in ann_fields if tx_id in f]
    if not match:
        return ""
    parts = match[0].split("|")
    return parts[1] if len(parts) > 1 else ""


def main():
    # One-hot ClinVar significance flags.
    out_df = pd.read_csv(CLINVAR_VCF, skiprows=41, sep="\t")
    out_df["snp"] = snp_from_vcf(out_df)
    out_df = add_clnsig_flags(out_df)
    out_df.to_csv(CLNSIG_CSV, index=False)

    # Merge the flags into the deletion score table.
    out_df = pd.read_csv(CLNSIG_CSV)
    mut_all = pd.read_csv(DELETION_SCORE_CSV)
    mut_all["snp"] = snp_from_vcf(mut_all)
    flag_cols = ["snp"] + [col for col, _ in CLNSIG_FLAGS]
    mut_all = pd.merge(mut_all, out_df[flag_cols], on="snp", how="left")
    mut_all.to_csv(DELETION_ADD_TYPE_CSV, index=False)

    # INFO comes from the annotated VCF. Rename the score-table INFO column
    # before the merge so that the annotated INFO keeps the name INFO.
    mut_all = mut_all.rename(columns={"INFO": "clinvar_info"})
    clinvar_info = read_annot_vcf(CLINVAR_ANNOT_VCF)
    clinvar_info["snp"] = (
        clinvar_info["#CHROM"].astype(str) + "_" +
        clinvar_info["POS"].astype(str) + "_" +
        clinvar_info["ID"].astype(str) + "_" +
        clinvar_info["REF"] + "_" +
        clinvar_info["ALT"]
    )
    clinvar_out = pd.merge(
        mut_all,
        clinvar_info[["snp", "INFO"]],
        on="snp",
        how="left",
    )

    # Keep only benign/pathogenic variants with a usable ID.
    clinvar_out = clinvar_out[
        clinvar_out["benign"].notna() & clinvar_out["ID"].notna()
    ]
    clinvar_out = clinvar_out[
        (clinvar_out["benign"] == 1) | (clinvar_out["Pathogenic"] == 1)
    ].copy()

    # Effect type from ANN=, with stop_gained mapped to nonsense.
    full_info = clinvar_out["INFO"].fillna("")
    clinvar_out["INFO"] = full_info.str.split(";ANN=").str[0]
    clinvar_out["eff_INFO"] = [
        x.split(";ANN=", 1)[1] if ";ANN=" in x else ""
        for x in full_info
    ]
    clinvar_out["eff_type"] = [
        parse_effect_info(eff, tx)
        for eff, tx in zip(clinvar_out["eff_INFO"], clinvar_out["tx_id"])
    ]
    clinvar_out["eff_typeb"] = (
        clinvar_out["eff_type"].str.split("&").str[0]
    )
    clinvar_out.loc[clinvar_out["eff_typeb"] == "stop_gained", "eff_typeb"] = "nonsense"

    # Per-fold change scores and their mean.
    for i in range(1, 6):
        clinvar_out["change" + str(i)] = (
            clinvar_out["wild_score" + str(i)] -
            clinvar_out["score_mut" + str(i)]
        )
    clinvar_out["score_change"] = clinvar_out[
        ["change" + str(i) for i in range(1, 6)]
    ].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Mutation type from the last '|' field of INFO, then the first ';' part.
    out_info = clinvar_out["INFO"].str.split("|").str[-1]
    clinvar_out["mut_type"] = out_info.str.split(";").str[0]

    # Restore the two-class pheno labels.
    benign = clinvar_out[clinvar_out["benign"] == 1].copy()
    pathogenic = clinvar_out[clinvar_out["Pathogenic"] == 1].copy()
    benign["pheno"] = "benign"
    pathogenic["pheno"] = "pathogenic"
    clinvar_out_use = pd.concat([benign, pathogenic], ignore_index=True)

    clinvar_out_use.to_csv(OUT_CSV, index=False)
    print("saved:", OUT_CSV)


if __name__ == "__main__":
    main()
