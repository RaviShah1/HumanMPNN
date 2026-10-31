# Immunogenicity score

Combines NetMHCIIpan binding predictions with similarity to human peptides.
Lower scores indicate lower predicted immunogenicity. Results are normalized
by sequence length and saved as one CSV row per sequence.

## Setup

Install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Download the NetMHCIIpan-4.3 package for your system from DTU and extract it with
its prediction data. Install `tcsh` and `awk`. In the `netMHCIIpan` launcher,
set `NMHOME` to the installation directory and `TMPDIR` to a writable directory.
Pass the launcher path with `--netmhc-bin` as shown below.

## Prepare data and score

Run these commands from the repository root:

```bash
python eval/immunogenicty_score/fetch_data.py
python eval/immunogenicty_score/setup_data.py \
  --netmhc-bin /path/to/netMHCIIpan-4.3/netMHCIIpan --workers 4
python eval/immunogenicty_score/score.py --fasta outputs/protein.fasta \
  --netmhc-bin /path/to/netMHCIIpan-4.3/netMHCIIpan \
  --output outputs/immunogenicity.csv
```

Data preparation is needed once and can take a while. Completed downloads and
reference predictions are reused on later runs. Data is stored in this directory's
`data/` folder.

To score existing NetMHCIIpan predictions, use `--netmhc-csv predictions.csv`
and `--seq-len 250` instead of `--fasta`. The reference data is still required.
