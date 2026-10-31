"""Design sequences from a PDB backbone using AR or MDM HumanMPNN."""
import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from models.load import CHECKPOINTS, load_model
from models.structure import parse_PDB, tied_featurize

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


@torch.inference_mode()
def design(model, protein, num_sequences=8, temperature=0.1, device='cpu',
           design_chains=None, design_positions=None):
    protein = copy.deepcopy(protein)
    chains = [k.removeprefix('seq_chain_') for k in protein if k.startswith('seq_chain_')]
    for chain in chains:
        protein[f'seq_chain_{chain}'] = protein[f'seq_chain_{chain}'].replace('-', 'X')
        if not set(protein[f'seq_chain_{chain}']) <= set(ALPHABET):
            raise ValueError(f'Unsupported amino acid in chain {chain}')
    protein['seq'] = ''.join(protein[f'seq_chain_{c}'] for c in chains)
    selected = chains if design_chains is None else design_chains
    if not chains or not set(selected) <= set(chains):
        raise ValueError(f'Design chains must belong to {chains}; got {selected}')
    if design_positions is not None and not set(design_positions) <= set(chains):
        raise ValueError('Position JSON contains an unknown chain')
    fixed, offsets, start = {}, {}, 0
    for chain in chains:
        length = len(protein[f'seq_chain_{chain}'])
        positions = (set(range(1, length + 1)) if design_positions is None
                     else set(design_positions.get(chain, [])))
        if any(type(i) is not int or i < 1 or i > length for i in positions):
            raise ValueError(f'Invalid 1-based position for chain {chain} (length {length})')
        fixed[chain] = sorted(set(range(1, length + 1)) - positions)
        offsets[chain] = np.arange(start, start + length)
        start += length
    arrays = tied_featurize(
        [copy.deepcopy(protein) for _ in range(num_sequences)], torch.device(device),
        chain_dict={protein['name']: (list(selected), [c for c in chains if c not in selected])},
        fixed_position_dict={protein['name']: fixed})
    (X, S, mask, _, chain_m, chain_enc, orders, _, _, _, positions,
     omit_mask, residx, _, _, pc, pb, po, bias, _) = arrays
    effective = mask * chain_m * positions
    omit = np.array([aa == 'X' for aa in ALPHABET], dtype=np.float32)
    if hasattr(model, 'mask_idx'):
        samples = model.sample(X, S, mask, effective, residx, chain_enc,
                               temperature=temperature, omit_AAs_np=omit)
    else:
        result = model.sample(
            X, torch.randn(chain_m.shape, device=X.device), S, chain_m, chain_enc, residx,
            mask=mask, temperature=temperature, omit_AAs_np=omit,
            bias_AAs_np=np.zeros(21, dtype=np.float32), chain_M_pos=positions,
            omit_AA_mask=omit_mask, pssm_coef=pc, pssm_bias=pb, pssm_multi=0.,
            pssm_log_odds_flag=False, pssm_log_odds_mask=(po > 0).float(),
            pssm_bias_flag=False, bias_by_res=bias)
        samples = result['S']
        # Official AR assigns token 0 at unresolved positions. Preserve native identity.
        samples = torch.where(effective.bool(), samples, S)
    if not torch.equal(samples[~effective.bool()], S[~effective.bool()]):
        raise RuntimeError('Sampler changed a fixed position')
    native = ''.join(protein[f'seq_chain_{c}'] for c in chains)
    designs = []
    for i, order in enumerate(orders):
        permutation = np.concatenate([offsets[c] for c in order])
        restored = np.empty(start, dtype=np.int64)
        restored[permutation] = samples[i, :start].cpu().numpy()
        effective_restored = np.empty(start, dtype=bool)
        effective_restored[permutation] = effective[i, :start].cpu().numpy().astype(bool)
        sequence = ''.join(ALPHABET[t] for t in restored)
        designs.append({'index': i, 'sequence': sequence,
                        'chains': {c: ''.join(sequence[j] for j in offsets[c]) for c in chains},
                        'designed_positions': (np.flatnonzero(effective_restored) + 1).tolist()})
    return {'name': protein['name'], 'chain_order': chains,
            'native_sequence': native, 'designs': designs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--pdb', type=Path, help='Full-backbone PDB file')
    inputs.add_argument('--jsonl', type=Path, help='Parsed ProteinMPNN JSONL structures')
    parser.add_argument('--model', choices=CHECKPOINTS, default='humanmpnn_ar_pretrained')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path)
    parser.add_argument('--num-sequences', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--design-chains', nargs='+', help='Other chains remain fixed backbone/sequence context')
    parser.add_argument('--design-positions', type=Path,
                        help='JSON {"A": [1, 2, 3]}; only these 1-based parsed-chain positions are redesigned')
    args = parser.parse_args()
    if args.num_sequences < 1 or not np.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--num-sequences and --temperature must be positive')
    random.seed(args.seed); np.random.seed(args.seed % (2**32)); torch.manual_seed(args.seed)
    model = load_model(args.model, args.device, args.weights_dir)
    if args.pdb:
        proteins = parse_PDB(str(args.pdb))
    else:
        proteins = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    if not proteins:
        parser.error('No protein structures found')
    positions = json.loads(args.design_positions.read_text()) if args.design_positions else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for protein in proteins:
        result = design(model, protein, args.num_sequences, args.temperature, args.device,
                        args.design_chains, positions)
        result.update(model=args.model, seed=args.seed, temperature=args.temperature)
        name = protein['name']
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Structure name must be a plain filename')
        (args.output_dir / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
        # Designs only: evaluation scripts can consume this FASTA directly.
        SeqIO.write(
            (SeqRecord(Seq(row['sequence']), id=f"{name}__{row['index']}",
                       description=f'model={args.model}') for row in result['designs']),
            args.output_dir / f'{name}.fasta', 'fasta-2line')
        print(f"Wrote {args.num_sequences} designs for {name}")


if __name__ == '__main__':
    main()
