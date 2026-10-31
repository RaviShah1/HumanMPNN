# Human proximity score

Compares ESMC-600M sequence embeddings with human and nonhuman protein references.
Scores range from 0 to 1. Higher values indicate greater similarity to the human
reference group.

## Setup

Install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

ESMC runs through Transformers, so no separate `esm` package is needed.
Its weights download automatically on first use and take about 2.3 GB.

## Prepare data and score

Run these commands from the repository root:

```bash
python eval/human_proximity_score/fetch_data.py
python eval/human_proximity_score/setup_data.py --device cuda
python eval/human_proximity_score/score.py --fasta outputs/protein.fasta \
  --artifact eval/human_proximity_score/data/reference_bank.npz \
  --output outputs/human_proximity.tsv --device cuda
```

Prepare the reference data once, then reuse it for later scoring. It is stored
in this directory's `data/` folder. Use `--device cpu` to run without a GPU.
Reduce `--max-tokens` or `--max-batch` if GPU memory is limited.

The output TSV contains one row per sequence, including `human_proximity_score`.
Sequences longer than 1,023 residues are marked `outside_validated_length` and
are not scored. Use `--embeddings-npz` to score cached embeddings instead of a FASTA.
