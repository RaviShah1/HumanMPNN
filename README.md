# HumanMPNN

Code for HumanMPNN, a family of autoregressive and masked diffusion models for protein inverse folding.

## Install

Use Python 3.11 or newer.

```bash
conda create -n humanmpnn python=3.11 -y
conda activate humanmpnn
python -m pip install -r requirements.txt
```

This installs the Python dependencies for inference and evaluation.
Model weights are included in `weights/`.

ESMC weights download automatically when needed. Immunogenicity scoring requires
a separate NetMHCIIpan installation, described in the [setup guide](eval/immunogenicty_score/README.md).

## Design sequences

Run commands from the repository root.

```bash
python inference.py --pdb protein.pdb --model humanmpnn_ar_pretrained \
  --num-sequences 8 --temperature 0.1 --seed 0 --output-dir outputs/
```

Outputs are a FASTA file of designs and a JSON file with sequence and chain details.
Use `--device cpu` to run without a GPU.

To redesign part of a protein, use `--design-chains A` or
`--design-positions positions.json`. The positions file contains entries such as
`{"A": [10, 11, 12]}`. Positions start at 1 within each parsed chain, rather than
using PDB residue numbers. Unselected positions and residues with incomplete
backbone coordinates stay unchanged.

## Models

Choose a model with `--model`.

| Model | Weights |
|---|---|
| `proteinmpnn` | `weights/proteinmpnn/v_48_020.pt` |
| `solublempnn` | `weights/solublempnn/v_48_020.pt` |
| `humanmpnn_ar_pretrained` | `weights/humanmpnn/ar_pretrained.pt` |
| `humanmpnn_ar_finetuned` | `weights/humanmpnn/ar_finetuned.pt` |
| `humanmpnn_mdm_pretrained` | `weights/humanmpnn/mdm_pretrained.pt` |
| `humanmpnn_mdm_finetuned` | `weights/humanmpnn/mdm_finetuned.pt` |
| `humanmpnn-ab` | `weights/humanmpnn-ab/ar_finetuned.pt` |

`humanmpnn-ab` is the best Stage 3 checkpoint from HumanMPNN-Ab v2: autoregressive
fine-tuning on human-origin paired antibody repertoires, followed by predicted
and experimental antibody–antigen complexes. Checkpoint provenance is recorded in
[`weights/humanmpnn-ab/provenance.json`](weights/humanmpnn-ab/provenance.json).
For an antibody–antigen complex, select only the antibody chains for redesign
(for example, `--model humanmpnn-ab --design-chains H L`) to keep the antigen fixed.

## Evaluate designs

- [Immunogenicity score](eval/immunogenicty_score/README.md): lower is better.
- [Human proximity score](eval/human_proximity_score/README.md): higher is better.

Each guide covers downloading reference data, preparing it, and scoring a FASTA.
Use `--help` on any script for more options.
