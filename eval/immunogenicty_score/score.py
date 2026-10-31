"""Tuned NetMHCIIpan/Janus immunogenicity per residue (lower is better)."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from netmhc import ALLELES, predict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import read_fasta


def janus_5mer(core):
    return ''.join(core[i] for i in (1, 2, 4, 6, 7))


def score_hits(frame, maps, length):
    if length <= 0:
        raise ValueError('Sequence length must be positive')
    rank = pd.to_numeric(frame['Rank_EL'], errors='raise')
    core_rel = pd.to_numeric(frame['Core_Rel'], errors='raise')
    hits = frame.loc[(rank <= 1.) & (core_rel >= .7) & frame['MHC'].isin(ALLELES)
                     & frame['Core'].astype(str).str.len().ge(9)]
    counts = np.fromiter(
        (maps[row.MHC].get(janus_5mer(str(row.Core)), 0) for row in hits.itertuples()),
        dtype=np.float64, count=len(hits))
    ranks = pd.to_numeric(hits['Rank_EL'], errors='raise').to_numpy(dtype=np.float64)
    contributions = np.square(-np.log10(np.maximum(ranks, 1e-3))) / np.square(1 + counts)
    return float(contributions.sum() / length)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--fasta', type=Path)
    sources.add_argument('--netmhc-csv', type=Path)
    parser.add_argument('--seq-len', type=int, help='Required with --netmhc-csv')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--netmhc-bin', default='netMHCIIpan', help='Path to the installed 4.3 executable')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.netmhc_csv and (args.seq_len is None or args.seq_len <= 0):
        parser.error('--seq-len must be positive when using --netmhc-csv')
    artifact = json.loads((args.data_dir / 'janus_counts.json').read_text())
    if artifact['rank_max'] != 1.0 or artifact['core_min'] != .7:
        raise ValueError('Janus artifact does not use the tuned reference filters')
    maps = artifact['counts']
    if set(maps) != set(ALLELES):
        raise ValueError('Janus artifact must contain all nine alleles')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    if args.netmhc_csv:
        value = score_hits(pd.read_csv(args.netmhc_csv), maps, args.seq_len)
        rows = [(args.netmhc_csv.stem, args.seq_len, value)]
    else:
        records = list(read_fasta(args.fasta))
        if not records or any(not sequence for _, sequence in records):
            raise ValueError('FASTA must contain nonempty sequences')
        rows = [(name, len(sequence), score_hits(predict(sequence, args.netmhc_bin), maps, len(sequence)))
                for name, sequence in records]
    pd.DataFrame(rows, columns=['id', 'length', 'immunogenicity_score']).to_csv(temporary, index=False)
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
