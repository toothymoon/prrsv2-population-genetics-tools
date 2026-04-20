"""
Identity_Pick_Identity-like_v5.py

Extracts sequences from a pre-aligned FASTA file that meet a minimum
nucleotide identity threshold relative to a single reference sequence.
Results are sorted by identity in descending order.

Identity is calculated using pairwise deletion: positions where either
sequence contains a gap ('-') or an ambiguous base (non-ACGT) are
excluded from both the numerator and the denominator.

Input:
  A pre-aligned FASTA file (all sequences must be the same length, e.g.
  produced by MAFFT G-INS-i or equivalent).

Output:
  - A FASTA file containing sequences that pass the identity threshold.
  - A CSV file listing sequence names and their identity values.

Usage:
  1. Set the four path variables and the threshold at the bottom of the
     script under USER CONFIGURATION.
  2. Run: python Identity_Pick_Identity-like_v5.py
"""

from Bio import SeqIO
import pandas as pd


# ---------------------------------------------------------------------------
# Identity calculation
# ---------------------------------------------------------------------------

def calculate_identity(seq1, seq2):
    """
    Calculate the percent nucleotide identity between two pre-aligned
    sequences of equal length.

    Sites where either sequence has a gap ('-') or an ambiguous base
    (i.e., not in {A, C, G, T}) are excluded (pairwise deletion).

    Parameters
    ----------
    seq1, seq2 : str or Bio.Seq
        Aligned nucleotide sequences of the same length.

    Returns
    -------
    identity : float or None
        Percent identity (0–100).  Returns None if no valid sites exist.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be the same length.")

    identical   = 0
    valid_sites = 0

    for a, b in zip(seq1, seq2):
        a = a.upper()
        b = b.upper()
        # Skip positions with gaps or ambiguous bases (pairwise deletion)
        if a in "ACGT" and b in "ACGT":
            valid_sites += 1
            if a == b:
                identical += 1

    if valid_sites == 0:
        return None   # No valid sites — identity undefined

    return (identical / valid_sites) * 100


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(input_fasta, target_id, output_fasta, output_csv, threshold=90.0):
    """
    Identify and export sequences whose identity to a reference meets or
    exceeds `threshold` percent.

    Parameters
    ----------
    input_fasta  : str    Path to the pre-aligned FASTA file.
    target_id    : str    Sequence ID of the reference (must exist in the file).
    output_fasta : str    Output path for the filtered FASTA file.
    output_csv   : str    Output path for the identity summary CSV.
    threshold    : float  Minimum identity (%) for inclusion (default: 90.0).
    """
    # Load alignment
    sequences = list(SeqIO.parse(input_fasta, "fasta"))
    print(f"Loaded {len(sequences)} sequences from '{input_fasta}'.")

    # Verify uniform sequence length (required for pre-aligned input)
    lengths = {len(r.seq) for r in sequences}
    if len(lengths) > 1:
        raise ValueError(
            f"Sequences have inconsistent lengths: {lengths}. "
            "Please supply a pre-aligned (MSA) FASTA file."
        )
    print(f"Alignment length: {lengths.pop()} bp")

    # Locate reference sequence
    target_seq = next(
        (r for r in sequences if r.id == target_id), None
    )
    if target_seq is None:
        raise ValueError(
            f"Reference ID '{target_id}' not found in '{input_fasta}'. "
            "Check the target_id setting."
        )
    print(f"Reference sequence: {target_seq.description}")

    # Calculate identity for every sequence against the reference
    passing = []
    skipped = 0

    for record in sequences:
        identity = calculate_identity(str(target_seq.seq), str(record.seq))
        if identity is None:
            print(f"  [SKIP] {record.description}: no valid aligned sites.")
            skipped += 1
            continue
        if identity >= threshold:
            passing.append((record, identity))

    if skipped:
        print(f"Skipped {skipped} sequence(s) with no valid aligned sites.")

    # Sort by identity descending
    passing.sort(key=lambda x: x[1], reverse=True)

    print(f"\nFound {len(passing)} sequence(s) with >= {threshold:.1f}% "
          f"identity to '{target_id}':")
    for record, ident in passing:
        print(f"  {record.description}: {ident:.2f}%")

    # Write filtered FASTA
    if passing:
        with open(output_fasta, "w") as fh:
            SeqIO.write([r for r, _ in passing], fh, "fasta")
        print(f"\nFiltered sequences saved to: '{output_fasta}'")
    else:
        print(f"\nNo sequences met the {threshold:.1f}% threshold. "
              "FASTA file not written.")

    # Write identity CSV
    if passing:
        df = pd.DataFrame(
            [[r.description, f"{i:.2f}%"] for r, i in passing],
            columns=["Sequence Name", "Identity (%)"]
        )
        df.to_csv(output_csv, index=False)
        print(f"Identity table saved to: '{output_csv}'")
    else:
        print("No data to write to CSV.")


# ---------------------------------------------------------------------------
# Entry point — edit the variables below to configure your analysis
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # USER CONFIGURATION
    # ------------------------------------------------------------------

    # Path to the pre-aligned MSA FASTA file (all sequences same length)
    input_fasta  = "alignment.fasta"

    # Accession / ID of the reference sequence (must exist in input_fasta)
    target_id    = "reference_accession"

    # Output FASTA — sequences that pass the identity threshold
    output_fasta = "identity_filtered.fasta"

    # Output CSV — sequence names and identity values
    output_csv   = "identity_results.csv"

    # Minimum identity threshold (%)
    threshold    = 90.0

    # ------------------------------------------------------------------

    main(input_fasta, target_id, output_fasta, output_csv, threshold)
