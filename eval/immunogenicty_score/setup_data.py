"""Predict human references and build tuned Janus counts."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path
import pandas as pd
from netmhc import ALLELES, predict
from score import janus_5mer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import read_references, read_fasta, verify_records


def build_counts(paths):
    counts = {a: Counter() for a in ALLELES}
    for path in paths:
        # Match the original float32 reference thresholds and (accession, allele, core) deduplication.
        frame = pd.read_csv(path, usecols=['MHC', 'Core', 'Core_Rel', 'Rank_EL'],
                            dtype={'Core_Rel': 'float32', 'Rank_EL': 'float32'})
        frame = frame.loc[(frame['Core_Rel'] >= .7) & (frame['Rank_EL'] <= 1.)
                          & frame['MHC'].isin(ALLELES) & frame['Core'].astype(str).str.len().ge(9)]
        frame = frame.drop_duplicates(['MHC', 'Core'])
        for row in frame.itertuples():
            counts[row.MHC][janus_5mer(row.Core)] += 1
    return {a: dict(c) for a, c in counts.items()}


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fasta', type=Path, default=here / 'data/references.fasta')
    parser.add_argument('--data-dir', type=Path, default=here / 'data')
    parser.add_argument('--predictions-dir', type=Path, help='Existing per-accession CSVs; all references must be present')
    parser.add_argument('--netmhc-bin', default='netMHCIIpan', help='Path to the installed NetMHCIIpan-4.3 executable')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    rows = read_references(here / 'references.tsv')
    records = dict(read_fasta(args.fasta)); verify_records(records, rows)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    directory = args.predictions_dir or args.data_dir / 'predictions'
    directory.mkdir(parents=True, exist_ok=True)
    def run(row):
        accession = row['accession']; path = directory / f'{accession}.csv'
        if not path.exists():
            if args.predictions_dir:
                raise FileNotFoundError(path)
            frame = predict(records[accession], args.netmhc_bin)
            temporary = path.with_suffix('.csv.tmp'); frame.to_csv(temporary, index=False); temporary.replace(path)
        return path
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        paths = list(executor.map(run, rows))
    result = {'rank_max': 1.0, 'core_min': .7, 'reference_count': len(rows),
              'counts': build_counts(paths)}
    temporary = args.data_dir / 'janus_counts.json.tmp'
    temporary.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    temporary.replace(args.data_dir / 'janus_counts.json')
    print(f'Built Janus counts for {len(paths)} references')


if __name__ == '__main__':
    main()
