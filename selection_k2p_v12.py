"""
selection_k2p_v12.py

Estimates pairwise dN, dS, and dN/dS for each lineage present in a
pre-aligned FASTA file using a simplified Nei-Gojobori method with
Kimura two-parameter (K2P) correction for multiple substitutions.

For each lineage the script reports mean dN, dS, dN/dS, standard
deviation, minimum, and maximum across all pairwise comparisons.
Statistical comparisons between lineages are performed using the
Mann-Whitney U test on per-pair dN/dS distributions.

Input:
  A pre-aligned FASTA file whose sequence headers follow the pipe-
  delimited format:
      ID | Isolate | Geo_loc | Year | Lineage

Output:
  - A plain-text results file  (dnds_results_k2p.txt)
  - A bar chart PNG             (dnds_plot_k2p.png)

Usage:
  1. Set `fasta_file` and optionally `output_txt` / `output_png` under
     USER CONFIGURATION at the bottom of the script.
  2. Adjust `max_sequences_per_lineage` and `max_pairs_per_lineage` if
     needed to control runtime on large datasets.
  3. Run: python selection_k2p_v12.py
"""

import random
import math
import logging
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; change to "TkAgg"
                               # if you need a pop-up window
import matplotlib.pyplot as plt
from Bio import AlignIO
from scipy.stats import mannwhitneyu

# Fix random seed for reproducibility of subsampling
random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------------------------
# Standard genetic code (NCBI table 1)
# ---------------------------------------------------------------------------

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

TRANSITIONS  = {("A","G"),("G","A"),("C","T"),("T","C")}
TRANSVERSIONS = {
    ("A","C"),("C","A"),("A","T"),("T","A"),
    ("G","C"),("C","G"),("G","T"),("T","G"),
}


# ---------------------------------------------------------------------------
# dN/dS calculation (Nei-Gojobori + K2P correction)
# ---------------------------------------------------------------------------

def calculate_dnds(seq1, seq2):
    """
    Estimate dN, dS, and dN/dS for a pair of aligned coding sequences.

    Method: simplified Nei-Gojobori (1986) site counting (1/3 synonymous,
    2/3 non-synonymous per codon) combined with Kimura two-parameter
    correction for multiple substitutions.  Proportions are rescaled when
    2P + Q approaches 1 to prevent logarithm domain errors.

    Parameters
    ----------
    seq1, seq2 : str
        Aligned nucleotide sequences of equal length divisible by 3.

    Returns
    -------
    dN, dS, dN_dS : float or np.nan
        NaN is returned for any value that cannot be computed.
    """
    if len(seq1) != len(seq2) or len(seq1) % 3 != 0:
        raise ValueError("Sequences must be aligned and of codon length.")

    syn_sites = non_syn_sites = 0
    syn_ts = syn_tv = non_syn_ts = non_syn_tv = 0

    for i in range(0, len(seq1), 3):
        codon1 = seq1[i:i+3].upper()
        codon2 = seq2[i:i+3].upper()
        if codon1 not in CODON_TABLE or codon2 not in CODON_TABLE:
            continue

        # Site counting: 1/3 synonymous, 2/3 non-synonymous per codon
        syn_sites     += 1 / 3
        non_syn_sites += 2 / 3

        if codon1 == codon2:
            continue

        aa1      = CODON_TABLE[codon1]
        aa2      = CODON_TABLE[codon2]
        n_diffs  = sum(codon1[j] != codon2[j] for j in range(3))
        weight   = 1 / n_diffs if n_diffs > 0 else 0

        for j in range(3):
            if codon1[j] == codon2[j]:
                continue
            pair = (codon1[j], codon2[j])
            is_ts = pair in TRANSITIONS
            is_tv = pair in TRANSVERSIONS
            if aa1 == aa2:           # synonymous substitution
                if is_ts: syn_ts += weight
                if is_tv: syn_tv += weight
            else:                    # non-synonymous substitution
                if is_ts: non_syn_ts += weight
                if is_tv: non_syn_tv += weight

    if syn_sites == 0 or non_syn_sites == 0:
        return np.nan, np.nan, np.nan

    # Proportions of transitions (P) and transversions (Q)
    P_s = syn_ts     / syn_sites     if syn_sites     > 0 else 0.0
    Q_s = syn_tv     / syn_sites     if syn_sites     > 0 else 0.0
    P_n = non_syn_ts / non_syn_sites if non_syn_sites > 0 else 0.0
    Q_n = non_syn_tv / non_syn_sites if non_syn_sites > 0 else 0.0

    # Rescale to prevent logarithm domain errors when 2P + Q >= 1
    for P, Q, tag in [(P_s, Q_s, "syn"), (P_n, Q_n, "nonsyn")]:
        if P + Q > 0:
            scale = min(1.0, 0.98 / (2 * P + Q))
            if tag == "syn":
                P_s, Q_s = P_s * scale, Q_s * scale
            else:
                P_n, Q_n = P_n * scale, Q_n * scale

    # K2P distance formula
    dS = (
        -0.5 * math.log(1 - 2*P_s - Q_s) - 0.25 * math.log(1 - 2*Q_s)
        if (1 - 2*P_s - Q_s) > 0 and (1 - 2*Q_s) > 0
        else np.nan
    )
    dN = (
        -0.5 * math.log(1 - 2*P_n - Q_n) - 0.25 * math.log(1 - 2*Q_n)
        if (1 - 2*P_n - Q_n) > 0 and (1 - 2*Q_n) > 0
        else np.nan
    )

    dN_dS = (
        dN / dS
        if not (np.isnan(dN) or np.isnan(dS)) and dS != 0
        else np.nan
    )

    logging.debug(
        f"P_s={P_s:.4f} Q_s={Q_s:.4f} P_n={P_n:.4f} Q_n={Q_n:.4f} "
        f"dN={dN:.4f} dS={dS:.4f} dN/dS={dN_dS:.4f}"
    )
    return dN, dS, dN_dS


# ---------------------------------------------------------------------------
# FASTA parsing — group sequences by lineage
# ---------------------------------------------------------------------------

def parse_and_group(fasta_file):
    """
    Read an aligned FASTA file and return a dict mapping lineage labels
    to lists of sequence strings.

    Expected header format (pipe-separated, 5 fields):
        >ID | Isolate | Geo_loc | Year | Lineage
    """
    try:
        alignment = AlignIO.read(fasta_file, "fasta")
        logging.info(f"Loaded {len(alignment)} sequences from '{fasta_file}'.")
    except Exception as exc:
        logging.error(f"Could not read '{fasta_file}': {exc}")
        raise

    groups  = {}
    skipped = 0

    for record in alignment:
        parts = record.description.split(" | ")
        if len(parts) != 5:
            logging.warning(
                f"[SKIP] Malformed header (expected 5 fields): "
                f"{record.description}"
            )
            skipped += 1
            continue
        lineage = parts[4].strip()
        groups.setdefault(lineage, []).append(str(record.seq))

    logging.info("Lineage groups:")
    for lin, seqs in groups.items():
        logging.info(f"  {lin}: {len(seqs)} sequences")
    if skipped:
        logging.warning(f"Skipped {skipped} record(s) due to malformed headers.")

    return groups


# ---------------------------------------------------------------------------
# Per-lineage dN/dS computation
# ---------------------------------------------------------------------------

def compute_lineage_dnds(
    lineage_groups,
    max_sequences=300,
    max_pairs=20000,
):
    """
    Compute summary dN/dS statistics for each lineage.

    Parameters
    ----------
    lineage_groups : dict  {lineage_label: [seq_str, ...]}
    max_sequences  : int   Maximum sequences per lineage (random subsampling).
    max_pairs      : int   Maximum pairwise comparisons per lineage.

    Returns
    -------
    results    : dict  {lineage: {dN, dS, dN/dS, min, max, std}}
    dnds_lists : dict  {lineage: [per-pair dN/dS values]}
    """
    results    = {}
    dnds_lists = {}

    for lineage, sequences in lineage_groups.items():
        if len(sequences) < 2:
            logging.info(
                f"Lineage {lineage}: only {len(sequences)} sequence(s) — skipped."
            )
            continue

        # Subsample if needed
        if len(sequences) > max_sequences:
            sequences = random.sample(sequences, max_sequences)
            logging.info(f"Lineage {lineage}: subsampled to {max_sequences} sequences.")

        dN_list = []
        dS_list = []
        dN_dS_list = []
        n_pairs = min(
            len(sequences) * (len(sequences) - 1) // 2,
            max_pairs
        )
        pair_count = 0

        for s1, s2 in combinations(sequences, 2):
            if pair_count >= n_pairs:
                logging.info(
                    f"Lineage {lineage}: reached {n_pairs} pair limit."
                )
                break
            try:
                dN, dS, dN_dS = calculate_dnds(s1, s2)
                if not any(np.isnan(v) for v in (dN, dS, dN_dS)):
                    dN_list.append(dN)
                    dS_list.append(dS)
                    dN_dS_list.append(dN_dS)
                    pair_count += 1
            except Exception as exc:
                logging.warning(
                    f"Lineage {lineage}: pair error — {exc}"
                )

        if not dN_list:
            logging.info(f"Lineage {lineage}: no valid dN/dS pairs computed.")
            continue

        avg_dN    = np.nanmean(dN_list)
        avg_dS    = np.nanmean(dS_list)
        avg_dNdS  = np.nanmean(dN_dS_list)
        min_dNdS  = max(0.0, np.nanmin(dN_dS_list))   # floor negatives at 0
        max_dNdS  = np.nanmax(dN_dS_list)
        std_dNdS  = np.nanstd(dN_dS_list)

        results[lineage]    = {
            "dN": avg_dN, "dS": avg_dS, "dN/dS": avg_dNdS,
            "min_dN_dS": min_dNdS, "max_dN_dS": max_dNdS,
            "std_dN_dS": std_dNdS,
        }
        dnds_lists[lineage] = dN_dS_list

        logging.info(
            f"Lineage {lineage}: dN={avg_dN:.4f}  dS={avg_dS:.4f}  "
            f"dN/dS={avg_dNdS:.4f}  std={std_dNdS:.4f}  "
            f"[{min_dNdS:.4f}, {max_dNdS:.4f}]"
        )

    return results, dnds_lists


# ---------------------------------------------------------------------------
# Save text results + Mann-Whitney U pairwise tests
# ---------------------------------------------------------------------------

def save_results(results, dnds_lists, output_txt):
    """
    Write per-lineage dN/dS summaries and all pairwise Mann-Whitney U
    test results to a plain-text file.
    """
    with open(output_txt, "w") as fh:
        for lin, res in results.items():
            fh.write(
                f"Lineage {lin}: "
                f"dN = {res['dN']:.4f}, "
                f"dS = {res['dS']:.4f}, "
                f"dN/dS (avg) = {res['dN/dS']:.4f}, "
                f"min = {res['min_dN_dS']:.4f}, "
                f"max = {res['max_dN_dS']:.4f}, "
                f"std = {res['std_dN_dS']:.4f}\n"
            )

        # Pairwise Mann-Whitney U tests on per-pair dN/dS distributions
        valid = [lin for lin in results if lin in dnds_lists]
        for lin1, lin2 in combinations(valid, 2):
            if dnds_lists[lin1] and dnds_lists[lin2]:
                u_stat, p_val = mannwhitneyu(
                    dnds_lists[lin1], dnds_lists[lin2],
                    nan_policy="omit"
                )
                fh.write(
                    f"Mann-Whitney U ({lin1} vs {lin2}): "
                    f"U = {u_stat:.4f}, p = {p_val:.4f}\n"
                )

    logging.info(f"Results saved to '{output_txt}'.")


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

# Canonical lineage display order (Yim-Im et al. 2023 classification)
LINEAGE_ORDER = [
    "L1A","L1B","L1C.1","L1C.2","L1C.3","L1C.4","L1C.5","L1C.n",
    "L1D","L1E","L1F","L1H","L1I","L1J","L2","L3","L4",
    "L5A","L5B","L6","L7","L8A","L8B","L8C","L8D","L8E",
    "L9A","L9B","L9C","L9D","L9E","L10","L11",
]

def plot_dnds(results, output_png, title="dN/dS Ratios by Lineage (K2P model)"):
    """
    Save a bar chart of mean dN/dS ± standard deviation for each lineage.

    Lineages are ordered according to LINEAGE_ORDER; any lineage not in
    that list is appended at the end in alphabetical order.
    """
    ordered = [lin for lin in LINEAGE_ORDER if lin in results]
    extras  = sorted(lin for lin in results if lin not in LINEAGE_ORDER)
    lineages = ordered + extras

    dnds_vals = [results[lin]["dN/dS"]    for lin in lineages]
    std_vals  = [results[lin]["std_dN_dS"] for lin in lineages]

    fig_width = max(10, len(lineages) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.bar(
        lineages, dnds_vals,
        color="#4CAF50", width=0.6,
        yerr=std_vals, capsize=5, ecolor="grey"
    )
    for idx, val in enumerate(dnds_vals):
        ax.text(idx, val + 0.02, f"{val:.2f}", ha="left", va="bottom", fontsize=8)

    ax.set_xlabel("Lineage")
    ax.set_ylabel("dN/dS")
    ax.set_title(title)
    ax.set_ylim(bottom=0, top=2.5)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    logging.info(f"Bar chart saved to '{output_png}'.")


# ---------------------------------------------------------------------------
# Entry point — edit the variables below to configure your analysis
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # USER CONFIGURATION
    # ------------------------------------------------------------------

    # Path to the pre-aligned FASTA file
    fasta_file = "alignment.fasta"

    # Output file paths
    output_txt = "dnds_results_k2p.txt"
    output_png = "dnds_plot_k2p.png"

    # Maximum sequences to use per lineage (random subsampling when exceeded)
    max_sequences_per_lineage = 300

    # Maximum pairwise comparisons per lineage
    max_pairs_per_lineage = 20_000

    # Chart title
    chart_title = "dN/dS Ratios by Lineage (K2P model)"

    # ------------------------------------------------------------------

    lineage_groups = parse_and_group(fasta_file)

    results, dnds_lists = compute_lineage_dnds(
        lineage_groups,
        max_sequences = max_sequences_per_lineage,
        max_pairs     = max_pairs_per_lineage,
    )

    save_results(results, dnds_lists, output_txt)
    plot_dnds(results, output_png, title=chart_title)
