"""Fetch UniProt reference sequences."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import fetch


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=here / 'data/references.fasta')
    parser.add_argument('--source-fasta', type=Path, help='Use a local/archived UniProt FASTA instead of downloading')
    args = parser.parse_args()
    fetch(here / 'references.tsv', args.output, args.source_fasta)


if __name__ == '__main__':
    main()
