"""Scan the ClinVar VCF for intronic variants per chromosome.

Variants between the start and end of every representative human transcript
are kept, and the variant position is converted to transcript coordinates.
The output is one chromosome file used by all ClinVar scoring scripts.

Usage:
    python 01_find_intron_variants.py <chromosome>
"""

import pandas as pd
from sys import argv

use_chr = argv[1]

tx_info = pd.read_csv("/data1/lty/clinvar2024/homo_mutant/homo_tx_seqs.csv")
tx_set = set(tx_info["tx_name"])


def is_representative(txid):
    """Return True when the transcript is in the representative set."""
    return txid in tx_set


vcf = pd.read_csv(
    "/data1/lty/clinvar2024/clinvar_20240624.vcf",
    skiprows=41,
    sep="\t",
)
vcf["#CHROM"] = vcf["#CHROM"].astype("str")
tx_info["seqnames"] = tx_info["seqnames"].astype("str")

vcf_use = vcf[vcf["#CHROM"] == use_chr]
tx_info_use = tx_info[tx_info["seqnames"] == str(use_chr)]

use_index = []
mut_loc = []
tx_id = []
for i in range(len(tx_info_use)):
    intron_mut = vcf_use[
        (tx_info_use["start"].iloc[i] < vcf_use["POS"])
        & (vcf_use["POS"] < tx_info_use["end"].iloc[i])
    ]
    if len(intron_mut) > 0:
        tx_gene = tx_info_use["tx_name"].iloc[i]
        strand = tx_info_use["strand"].iloc[i]
        if strand == "+":
            start = tx_info_use["start"].iloc[i]
            mut_loc.extend(intron_mut["POS"] - start + 1)
        else:
            end = tx_info_use["end"].iloc[i]
            mut_loc.extend(end - intron_mut["POS"] + 1)
        tx_id.extend([tx_gene] * len(intron_mut))
        use_index.extend(intron_mut.index)

vcf_out = vcf_use.loc[use_index]
vcf_out["tx_id"] = tx_id
vcf_out["mut_loc"] = mut_loc

vcf_out.to_csv(
    "/data1/lty/clinvar2024/new_mut/new_allscan/clinvar2024_intron_mut_chr"
    + str(use_chr)
    + ".csv"
)
