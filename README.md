# PRRSV-2 Population Genetics Tools

Python scripts for population genetic analysis of viral ORF5 sequences.

## Scripts

| Script | Description |
|--------|-------------|
| `calculate_distance-identity_k2p.py` | Pairwise genetic distance and nucleotide identity (K2P model) |
| `calculate_fst_v7.2.5.py` | Hudson's F_ST with permutation test and Welch's t-test |
| `Identity_Pick_Identity-like_v5.py` | Extract sequences by identity threshold to a reference |
| `selection_k2p_v12.py` | dN/dS estimation (Nei-Gojobori + K2P correction) |

## Requirements
biopython>=1.79
numpy
pandas
scipy
matplotlib
tqdm

## Input format

All scripts expect pre-aligned FASTA files with pipe-delimited headers:
ID | Isolate | Country | Year | Lineage
## Citation

If you use these scripts, please cite:
He J-T, Kuo C-H, He J-L (2026) ...（Artical）... DOI: [Zenodo DOI]

## License

MIT License
