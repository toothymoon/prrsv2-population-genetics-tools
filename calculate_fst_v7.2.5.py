"""
calculate_fst_v7.2.5.py

Calculates pairwise Hudson's F_ST between user-defined sequence groups
extracted from a single aligned FASTA file.

For each pair of groups the script computes:
  - Hudson's F_ST (point estimate)
  - Permutation-based empirical p-value for F_ST
  - Welch's t-statistic and p-value comparing per-SNP nucleotide
    diversity (Patterson's pi) between the two groups

Input:
  A pre-aligned FASTA file whose sequence headers follow the pipe-
  delimited format:
      ID | Isolate | Geo_loc | Year | Lineage

Output:
  A CSV file containing the full pairwise result table.

Usage:
  1. Set `fasta_file` and `output_fst_matrix` at the bottom of the script.
  2. Edit `group_conditions` to define your groups (Geo_loc, year_range,
     and/or lineage filters).  Add or remove entries as needed.
  3. Run: python calculate_fst_v7.2.5.py
"""

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from scipy.stats import ttest_ind

# Fix random seed for reproducibility of the permutation test
np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. Parse FASTA and build a metadata DataFrame
# ---------------------------------------------------------------------------

def parse_fasta(fasta_file):
    """
    Read an aligned FASTA file and return a DataFrame with columns:
    ID, Isolate, Geo_loc, Year, Lineage, Sequence.

    Expected header format (pipe-separated, 5 fields):
        >ID | Isolate | Geo_loc | Year | Lineage
    Records with malformed headers are skipped with a warning.
    """
    records = []
    for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc="Parsing FASTA"):
        fields = record.description.split(" | ")
        if len(fields) != 5:
            print(f"  [SKIP] Malformed header: {record.description}")
            continue
        try:
            records.append({
                "ID":       fields[0],
                "Isolate":  fields[1],
                "Geo_loc":  fields[2],
                "Year":     int(fields[3]),
                "Lineage":  fields[4],
                "Sequence": str(record.seq).upper()
            })
        except ValueError as exc:
            print(f"  [SKIP] Could not parse record '{record.description}': {exc}")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Filter sequences by Geo_loc, year range, and/or lineage
# ---------------------------------------------------------------------------

def filter_sequences(df, Geo_loc=None, year_range=None, lineage=None):
    """
    Return a subset of `df` matching all supplied criteria.

    Parameters
    ----------
    df         : pd.DataFrame  Full metadata DataFrame from parse_fasta().
    Geo_loc    : str or None   Exact match on the 'Geo_loc' column.
    year_range : tuple or None (min_year, max_year) inclusive.
    lineage    : str or None   Exact match on the 'Lineage' column.
    """
    out = df.copy()
    if Geo_loc:
        out = out[out["Geo_loc"] == Geo_loc]
    if year_range:
        out = out[(out["Year"] >= year_range[0]) & (out["Year"] <= year_range[1])]
    if lineage:
        out = out[out["Lineage"] == lineage]
    return out


# ---------------------------------------------------------------------------
# 3. Extract SNPs and encode sequences as integer genotype matrices
# ---------------------------------------------------------------------------

def sequences_to_genotypes(seq_df1, seq_df2):
    """
    Identify variable sites (SNPs) shared between two groups and encode
    them as integer matrices (A=0, C=1, G=2, T=3).

    Sites containing gaps ('-') or ambiguous bases are excluded.

    Returns
    -------
    geno1, geno2 : np.ndarray  shape (n_samples, n_snps), dtype int8
    snp_positions : list of int  0-based column indices of retained SNPs
    """
    combined_seqs = pd.concat([seq_df1, seq_df2])["Sequence"].values
    seq_matrix    = np.array([list(s) for s in combined_seqs])

    lengths = {len(s) for s in combined_seqs}
    if len(lengths) > 1:
        raise ValueError(f"Sequences have inconsistent lengths: {lengths}")

    # Identify polymorphic columns with no gaps
    snp_positions = [
        i for i in range(seq_matrix.shape[1])
        if len({a for a in seq_matrix[:, i] if a in "ACGT"}) > 1
        and "-" not in seq_matrix[:, i]
    ]

    if not snp_positions:
        raise ValueError("No valid SNPs found between the two groups.")

    print(f"  Found {len(snp_positions)} SNP positions.")

    nuc_map     = {"A": 0, "C": 1, "G": 2, "T": 3}
    geno_matrix = np.zeros((len(combined_seqs), len(snp_positions)), dtype=np.int8)
    for col_idx, pos in enumerate(snp_positions):
        for row_idx, nuc in enumerate(seq_matrix[:, pos]):
            geno_matrix[row_idx, col_idx] = nuc_map[nuc]

    n1    = len(seq_df1)
    geno1 = geno_matrix[:n1, :]
    geno2 = geno_matrix[n1:, :]

    if geno1.shape[1] != geno2.shape[1]:
        raise ValueError("SNP count mismatch between the two groups after encoding.")

    return geno1, geno2, snp_positions


# ---------------------------------------------------------------------------
# 4. Allele counting
# ---------------------------------------------------------------------------

def count_alleles(geno, n_alleles=4):
    """
    Count allele occurrences at each SNP site.

    Returns
    -------
    counts : np.ndarray  shape (n_snps, n_alleles), dtype int32
    """
    n_snps   = geno.shape[1]
    counts   = np.zeros((n_snps, n_alleles), dtype=np.int32)
    for snp in range(n_snps):
        for allele in range(n_alleles):
            counts[snp, allele] = np.sum(geno[:, snp] == allele)
    return counts


# ---------------------------------------------------------------------------
# 5. Per-SNP nucleotide diversity (Patterson's pi)
# ---------------------------------------------------------------------------

def patterson_pi(ac):
    """
    Compute per-SNP Patterson's pi (unbiased nucleotide diversity).

    Parameters
    ----------
    ac : np.ndarray  shape (n_snps, n_alleles)  allele count matrix

    Returns
    -------
    pi : np.ndarray  shape (n_snps,)  NaN where n <= 1
    """
    pi = np.zeros(ac.shape[0])
    for i in range(ac.shape[0]):
        n = np.sum(ac[i])
        if n <= 1:
            pi[i] = np.nan
        else:
            p      = ac[i] / n
            pi[i]  = (n / (n - 1)) * (1 - np.sum(p ** 2))
    return pi


# ---------------------------------------------------------------------------
# 6. Hudson's F_ST
# ---------------------------------------------------------------------------

def hudson_fst(ac1, ac2):
    """
    Compute Hudson's F_ST (ratio-of-averages estimator).

    Negative values are floored to 0.0 to correct for downward bias at
    low-diversity loci.

    Parameters
    ----------
    ac1, ac2 : np.ndarray  shape (n_snps, n_alleles)  allele count matrices

    Returns
    -------
    fst : float
    """
    numerator   = np.zeros(ac1.shape[0])
    denominator = np.zeros(ac1.shape[0])

    for i in range(ac1.shape[0]):
        n1 = np.sum(ac1[i])
        n2 = np.sum(ac2[i])
        if n1 <= 1 or n2 <= 1:
            continue
        p1 = ac1[i] / n1
        p2 = ac2[i] / n2
        numerator[i]   = np.sum((p1 - p2) ** 2
                                - p1 * (1 - p1) / (n1 - 1)
                                - p2 * (1 - p2) / (n2 - 1))
        denominator[i] = np.sum(p1 * (1 - p2) + p2 * (1 - p1))

    total_den = np.sum(denominator)
    if total_den == 0:
        return 0.0
    fst = np.sum(numerator) / total_den
    return max(fst, 0.0)


# ---------------------------------------------------------------------------
# 7. F_ST with permutation test and Welch's t-test on per-SNP pi
# ---------------------------------------------------------------------------

def calculate_fst_with_tests(geno1, geno2, n_permutations=1000):
    """
    Compute F_ST, its permutation p-value, and a Welch's t-test comparing
    per-SNP nucleotide diversity between the two groups.

    Parameters
    ----------
    geno1, geno2   : np.ndarray  Encoded genotype matrices.
    n_permutations : int         Number of label-permutation replicates.

    Returns
    -------
    fst_obs   : float  Observed Hudson's F_ST.
    fst_pval  : float  Empirical (permutation) p-value for F_ST.
    t_stat    : float  Welch's t-statistic for per-SNP pi comparison.
    pi_pval   : float  Two-tailed p-value for the t-test.
    """
    if geno1.shape[1] != geno2.shape[1]:
        raise ValueError("Genotype matrices have different SNP counts.")

    # Observed F_ST
    ac1     = count_alleles(geno1)
    ac2     = count_alleles(geno2)
    fst_obs = hudson_fst(ac1, ac2)

    # Welch's t-test on per-SNP pi
    pi1       = patterson_pi(ac1)
    pi2       = patterson_pi(ac2)
    valid_pi1 = pi1[~np.isnan(pi1)]
    valid_pi2 = pi2[~np.isnan(pi2)]

    if len(valid_pi1) > 1 and len(valid_pi2) > 1:
        t_stat, pi_pval = ttest_ind(valid_pi1, valid_pi2, equal_var=False)
    else:
        t_stat, pi_pval = np.nan, np.nan

    # Permutation test for F_ST significance
    print(f"  Running {n_permutations} permutations for F_ST significance...")
    geno_all = np.vstack((geno1, geno2))
    n1       = geno1.shape[0]
    n_total  = geno_all.shape[0]
    count_ge = 0

    for _ in tqdm(range(n_permutations), desc="  Permutations", leave=False):
        idx        = np.random.permutation(n_total)
        ac1_perm   = count_alleles(geno_all[idx[:n1], :])
        ac2_perm   = count_alleles(geno_all[idx[n1:], :])
        fst_perm   = hudson_fst(ac1_perm, ac2_perm)
        if fst_perm >= fst_obs:
            count_ge += 1

    # Bias-corrected empirical p-value
    fst_pval = (count_ge + 1) / (n_permutations + 1)

    return fst_obs, fst_pval, t_stat, pi_pval


# ---------------------------------------------------------------------------
# 8. Main pipeline
# ---------------------------------------------------------------------------

def main(fasta_file, group_conditions, output_csv=None, n_permutations=1000):
    """
    Run the full pairwise F_ST analysis for all group pairs defined in
    `group_conditions`.

    Parameters
    ----------
    fasta_file       : str   Path to the aligned FASTA file.
    group_conditions : dict  Mapping of group labels to filter criteria.
                             Each value is a dict with optional keys:
                             'Geo_loc', 'year_range', 'lineage'.
    output_csv       : str or None  Path for the CSV output file.
    n_permutations   : int   Number of permutation replicates (default 1000).
    """
    print("Parsing FASTA file...")
    df = parse_fasta(fasta_file)

    # Build filtered group DataFrames
    groups = {}
    for name, cond in group_conditions.items():
        groups[name] = filter_sequences(
            df,
            Geo_loc    = cond.get("Geo_loc"),
            year_range = cond.get("year_range"),
            lineage    = cond.get("lineage")
        )
        n = len(groups[name])
        if n == 0:
            raise ValueError(f"Group '{name}' is empty after filtering. "
                             "Check your group_conditions.")
        print(f"  Group '{name}': {n} sequences")

    group_names = list(groups.keys())
    n_groups    = len(group_names)

    # Initialise result matrices
    fst_mat    = np.zeros((n_groups, n_groups))
    fst_p_mat  = np.zeros((n_groups, n_groups))
    t_mat      = np.zeros((n_groups, n_groups))
    pi_p_mat   = np.zeros((n_groups, n_groups))

    # Pairwise comparisons (upper triangle only; mirror after)
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            name_i = group_names[i]
            name_j = group_names[j]
            print(f"\n[{name_i}  vs  {name_j}]")

            geno1, geno2, _ = sequences_to_genotypes(
                groups[name_i], groups[name_j]
            )
            fst_obs, fst_pval, t_stat, pi_pval = calculate_fst_with_tests(
                geno1, geno2, n_permutations=n_permutations
            )

            # Fill both halves of the symmetric matrices
            fst_mat[i, j]  = fst_mat[j, i]  = fst_obs
            fst_p_mat[i, j]= fst_p_mat[j, i]= fst_pval
            t_mat[i, j]    =  t_stat
            t_mat[j, i]    = -t_stat          # sign reflects which group is 'first'
            pi_p_mat[i, j] = pi_p_mat[j, i]  = pi_pval

            print(f"  -> F_ST: {fst_obs:.4f}  (permutation p = {fst_pval:.4f})")
            print(f"  -> Welch's t: {t_stat:.4f}  (pi p-value = {pi_pval:.4f})")

    # Save CSV output
    if output_csv:
        rows = []
        for i in range(n_groups):
            for j in range(n_groups):
                if i != j:
                    rows.append({
                        "Group1":       group_names[i],
                        "Group2":       group_names[j],
                        "Fst":          fst_mat[i, j],
                        "Fst_p_value":  fst_p_mat[i, j],
                        "pi_t_stat":    t_mat[i, j],
                        "pi_p_value":   pi_p_mat[i, j]
                    })
        pd.DataFrame(rows).to_csv(output_csv, index=False, float_format="%.4f")
        print(f"\nResults saved to: {output_csv}")

    return fst_mat, fst_p_mat, groups


# ---------------------------------------------------------------------------
# Entry point — edit the variables below to configure your analysis
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # USER CONFIGURATION
    # ------------------------------------------------------------------

    # Path to the pre-aligned FASTA file
    fasta_file = "alignment.fasta"

    # Output CSV filename
    output_csv = "fst_results.csv"

    # Number of permutation replicates for F_ST significance testing
    n_permutations = 1000

    # Define groups to compare.
    # Each entry maps a group label to a dict of filter criteria.
    # Available keys (all optional; omit to skip that filter):
    #   "Geo_loc"    : str   — exact match on the Geo_loc field
    #   "year_range" : tuple — (min_year, max_year) inclusive
    #   "lineage"    : str   — exact match on the Lineage field
    #
    # Add or remove entries as needed.
    group_conditions = {
        "Group1": {
            "Geo_loc":    "Geo_loc_A",
            "year_range": (2015, 2024),
            "lineage":    "LineageX"
        },
        "Group2": {
            "Geo_loc":    "Geo_loc_B",
            "year_range": (2015, 2024),
            "lineage":    "LineageX"
        },
        "Group3": {
            "Geo_loc":    "Geo_loc_C",
            "year_range": (2015, 2024),
            "lineage":    "LineageX"
        },
        "Group4": {
            "Geo_loc":    "Geo_loc_D",
            "year_range": (2015, 2024),
            "lineage":    "LineageX"
        },
    }
    # ------------------------------------------------------------------

    fst_matrix, fst_p_matrix, groups = main(
        fasta_file, group_conditions, output_csv, n_permutations
    )

    # Print final summary table
    print("\n" + "=" * 56)
    print("FINAL SUMMARY  —  F_ST (permutation p-value)")
    print("=" * 56)
    names = list(group_conditions.keys())
    for i, g1 in enumerate(names):
        for j, g2 in enumerate(names):
            if j > i:
                print(f"  {g1} vs {g2}: "
                      f"{fst_matrix[i, j]:.4f} "
                      f"(p = {fst_p_matrix[i, j]:.4f})")
