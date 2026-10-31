"""Rebuild the frozen reference bank from ESMC embeddings; no metric fitting."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from score import embed_records, normalize
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import read_references, read_fasta, verify_records


def build_artifact(rows, config, accessions, embeddings):
    accessions = [str(a) for a in accessions]
    if len(accessions) != len(set(accessions)):
        raise ValueError('Duplicate accessions in embedding input')
    if embeddings.ndim != 2 or embeddings.shape != (len(accessions), config['shape'][2]):
        raise ValueError('Embedding shape does not match accessions and ESMC dimension')
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError('Embeddings must be finite and nonzero')
    lookup = {a: i for i, a in enumerate(accessions)}
    missing = {r['accession'] for r in rows} - lookup.keys()
    if missing:
        raise ValueError(f'Missing {len(missing)} reference embeddings')
    refs = np.empty(config['shape'], dtype=np.float16)
    names = np.empty(config['shape'][:2], dtype='<U32')
    values = normalize(embeddings.astype(np.float32))
    for row in rows:
        group = config['group_names'].index(row['group']); index = int(row['index'])
        refs[group, index] = values[lookup[row['accession']]]
        names[group, index] = row['accession']
    art = {k: np.array(v) for k, v in config.items() if k != 'shape'}
    for key in ['neighbor_tau', 'conditional_weight', 'group_tau', 'output_temperature']:
        art[key] = np.float32(config[key])
    art.update(reference_embeddings=refs, reference_accessions=names)
    return art


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fasta', type=Path, default=here / 'data/references.fasta')
    parser.add_argument('--embeddings-npz', type=Path, help='Optional existing ESMC mean-pool cache: accessions/ids, embeddings')
    parser.add_argument('--output', type=Path, default=here / 'data/reference_bank.npz')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--max-batch', type=int, default=16)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.max_batch < 1:
        parser.error('Batch limits must be positive')
    rows = read_references(here / 'references.tsv')
    config = json.loads((here / 'config.json').read_text())
    if args.embeddings_npz:
        with np.load(args.embeddings_npz, allow_pickle=False) as z:
            accessions = z['ids'] if 'ids' in z else z['accessions']
            embeddings = z['embeddings']
    else:
        records = dict(read_fasta(args.fasta)); verify_records(records, rows)
        accessions = list(dict.fromkeys(row['accession'] for row in rows))
        if any(not 1 <= len(records[a]) <= 1023 for a in accessions):
            raise ValueError('References must be 1–1023 residues')
        embeddings = embed_records([(a, records[a]) for a in accessions], args.device,
                                   args.max_tokens, args.max_batch)
    art = build_artifact(rows, config, accessions, embeddings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **art); temporary.replace(args.output)
    print(f'Wrote reference bank {art["reference_embeddings"].shape}: {args.output}')


if __name__ == '__main__':
    main()
