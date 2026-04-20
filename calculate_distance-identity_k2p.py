"""
calculate_distance-identity_k2p.py

Calculates pairwise genetic distances and nucleotide identities between
user-defined sequence groups using the Kimura two-parameter (K2P) model.

Input:  Pre-aligned FASTA files (one per group), placed in the same
        directory as this script.
Output: Distance matrix CSV, identity matrix CSV, and a Markdown summary.

Usage:
    1. Edit the `groups` dictionary in main() to match your group names
       and corresponding FASTA filenames.
    2. Run: python calculate_distance-identity_k2p.py
"""

import os
import math
from itertools import combinations

import numpy as np
import pandas as pd
from Bio import AlignIO

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core distance functions
# ---------------------------------------------------------------------------

def calculate_pairwise_distance(seq1, seq2, model="k2p"):
    """
    Calculate K2P genetic distance and nucleotide identity between two
    aligned sequences of equal length.

    Parameters
    ----------
    seq1, seq2 : Bio.Seq or str
        Aligned nucleotide sequences (same length).
    model : str
        Substitution model. Only 'k2p' is currently supported.

    Returns
    -------
    distance : float or None
        K2P distance. Returns None if computation is undefined.
    identity : float or None
        Raw nucleotide identity (1 - p-distance). Returns None on failure.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be the same length.")

    transitions  = {('A','G'),('G','A'),('C','T'),('T','C')}
    transversions = {('A','C'),('C','A'),('A','T'),('T','A'),
                     ('G','C'),('C','G'),('G','T'),('T','G')}

    valid_sites = transitions_n = transversions_n = differences = 0

    for a, b in zip(seq1, seq2):
        if a in 'ACGT' and b in 'ACGT':   # skip gaps and ambiguous bases
            valid_sites += 1
            if a != b:
                differences += 1
                if (a, b) in transitions:
                    transitions_n += 1
                elif (a, b) in transversions:
                    transversions_n += 1

    if valid_sites == 0:
        return None, None

    if model == "k2p":
        P = transitions_n  / valid_sites
        Q = transversions_n / valid_sites
        try:
            t1 = 1 - 2*P - Q
            t2 = 1 - 2*Q
            if t1 <= 0 or t2 <= 0:
                return None, None
            distance = -0.5 * math.log(t1) - 0.25 * math.log(t2)
            identity  = 1 - (differences / valid_sites)
            return distance, identity
        except (ValueError, ZeroDivisionError):
            return None, None
    else:
        raise ValueError(f"Unsupported model '{model}'. Only 'k2p' is supported.")


def within_group_distance(fasta_path):
    """
    Compute the mean within-group K2P distance and nucleotide identity
    for all pairwise combinations in an aligned FASTA file.

    Returns
    -------
    mean_distance : float
    mean_identity : float
    seq_count     : int
    """
    try:
        alignment = AlignIO.read(fasta_path, "fasta")
    except Exception as exc:
        print(f"  [ERROR] Could not read '{fasta_path}': {exc}")
        return np.nan, np.nan, 0

    seq_count = len(alignment)
    if seq_count < 2:
        print(f"  [WARNING] '{fasta_path}' has fewer than 2 sequences "
              f"({seq_count}); within-group distance cannot be computed.")
        return np.nan, np.nan, seq_count

    distances, identities = [], []
    for s1, s2 in combinations(alignment, 2):
        d, i = calculate_pairwise_distance(s1.seq, s2.seq)
        if d is not None:
            distances.append(d)
            identities.append(i)

    return (np.mean(distances)  if distances  else np.nan,
            np.mean(identities) if identities else np.nan,
            seq_count)


def between_group_distance(fasta_path1, fasta_path2):
    """
    Compute the mean between-group K2P distance and nucleotide identity
    for all cross-group sequence pairs.

    Returns
    -------
    mean_distance : float
    mean_identity : float
    """
    try:
        aln1 = AlignIO.read(fasta_path1, "fasta")
        aln2 = AlignIO.read(fasta_path2, "fasta")
    except Exception as exc:
        print(f"  [ERROR] Could not read alignment files: {exc}")
        return np.nan, np.nan

    if len(aln1[0]) != len(aln2[0]):
        print(f"  [ERROR] Sequence lengths differ between "
              f"'{fasta_path1}' and '{fasta_path2}'.")
        return np.nan, np.nan

    ids1 = {r.id for r in aln1}
    combined = list(aln1) + list(aln2)

    distances, identities = [], []
    for s1, s2 in combinations(combined, 2):
        cross = (s1.id in ids1) != (s2.id in ids1)   # XOR: one from each group
        if cross:
            d, i = calculate_pairwise_distance(s1.seq, s2.seq)
            if d is not None:
                distances.append(d)
                identities.append(i)

    return (np.mean(distances)  if distances  else np.nan,
            np.mean(identities) if identities else np.nan)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # USER CONFIGURATION
    # Edit the dictionary below to define your groups.
    # Keys   = group labels used in output tables.
    # Values = FASTA filenames (must be pre-aligned; placed in the same
    #          directory as this script, or provide absolute paths).
    # You may add or remove entries as needed.
    # ------------------------------------------------------------------
    groups = {
        "Group1": "Group1.fasta",
        "Group2": "Group2.fasta",
        "Group3": "Group3.fasta",
        "Group4": "Group4.fasta",
    }
    # Output prefix for CSV and Markdown files
    output_prefix = "distance_identity"
    # ------------------------------------------------------------------

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    os.makedirs(output_dir, exist_ok=True)

    group_names = list(groups.keys())
    n = len(group_names)
    dist_mat = np.full((n, n), np.nan)
    iden_mat = np.full((n, n), np.nan)
    seq_counts = {}
    report = [
        "# Genetic Distance and Nucleotide Identity Analysis (K2P model)\n"
    ]

    # ---- Within-group distances ----------------------------------------
    report.append("## Within-group mean distance and identity\n")
    print("=" * 60)
    print("Within-group statistics")
    print("=" * 60)

    for i, name in enumerate(group_names):
        fpath = os.path.join(script_dir, groups[name])
        if not os.path.exists(fpath):
            print(f"  [SKIP] '{fpath}' not found.")
            report.append(f"- {name}: file not found\n")
            continue

        d, ident, cnt = within_group_distance(fpath)
        seq_counts[name] = cnt
        dist_mat[i, i] = d
        iden_mat[i, i] = ident

        print(f"  {name}  (n = {cnt})")
        print(f"    Mean distance : {d:.4f}")
        print(f"    Mean identity : {ident:.4f}")
        report.append(f"- **{name}** (n = {cnt})")
        report.append(f"  - Mean distance : {d:.4f}")
        report.append(f"  - Mean identity : {ident:.4f}\n")

    # ---- Between-group distances ----------------------------------------
    report.append("## Between-group mean distance and identity\n")
    print("\n" + "=" * 60)
    print("Between-group statistics")
    print("=" * 60)

    for i, name_i in enumerate(group_names):
        for j, name_j in enumerate(group_names):
            if i >= j:
                continue
            fp_i = os.path.join(script_dir, groups[name_i])
            fp_j = os.path.join(script_dir, groups[name_j])
            if not (os.path.exists(fp_i) and os.path.exists(fp_j)):
                print(f"  [SKIP] Missing file(s) for {name_i} vs {name_j}.")
                report.append(f"- {name_i} vs {name_j}: file(s) not found\n")
                continue

            d, ident = between_group_distance(fp_i, fp_j)
            dist_mat[i, j] = dist_mat[j, i] = d
            iden_mat[i, j] = iden_mat[j, i] = ident

            print(f"  {name_i} vs {name_j}")
            print(f"    Mean distance : {d:.4f}")
            print(f"    Mean identity : {ident:.4f}")
            report.append(f"- **{name_i}** vs **{name_j}**")
            report.append(f"  - Mean distance : {d:.4f}")
            report.append(f"  - Mean identity : {ident:.4f}\n")

    # ---- Build DataFrames -----------------------------------------------
    dist_df = pd.DataFrame(dist_mat, index=group_names,
                           columns=group_names).round(4)
    iden_df = pd.DataFrame(iden_mat, index=group_names,
                           columns=group_names).round(4)

    # ---- Print and append matrices to report ----------------------------
    for label, df in [("Distance", dist_df), ("Identity", iden_df)]:
        header = f"## {label} matrix\n"
        report.append(header)
        print(f"\n{label} matrix:")
        if TABULATE_AVAILABLE:
            tbl = df.to_markdown()
            print(tbl)
            report.append(tbl + "\n")
        else:
            print(df.to_string())
            report.append("```\n" + df.to_string() + "\n```\n")

    if not TABULATE_AVAILABLE:
        print("\n[TIP] Install 'tabulate' for Markdown-formatted tables: "
              "pip install tabulate")

    # ---- Save outputs ---------------------------------------------------
    dist_csv = os.path.join(output_dir, f"{output_prefix}_distance_k2p.csv")
    iden_csv = os.path.join(output_dir, f"{output_prefix}_identity_k2p.csv")
    md_file  = os.path.join(output_dir, f"{output_prefix}_results_k2p.md")

    dist_df.to_csv(dist_csv)
    iden_df.to_csv(iden_csv)

    with open(md_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))

    print(f"\nOutputs saved to:")
    print(f"  {dist_csv}")
    print(f"  {iden_csv}")
    print(f"  {md_file}")


if __name__ == "__main__":
    main()
