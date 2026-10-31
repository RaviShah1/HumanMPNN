"""NetMHCIIpan-4.3 invocation and parser for the paper's nine DRB1 alleles."""
import re
import io
import subprocess
import tempfile
from pathlib import Path
import pandas as pd

ALLELES = ('DRB1_0101', 'DRB1_0301', 'DRB1_0401', 'DRB1_0701', 'DRB1_0801',
           'DRB1_0901', 'DRB1_1101', 'DRB1_1301', 'DRB1_1501')
COLUMNS = ['Pos', 'MHC', 'Peptide', 'Of', 'Core', 'Core_Rel', 'Inverted',
           'Identity', 'Score_EL', 'Rank_EL', 'Exp_Bind', 'BindLevel']


def predict(sequence, executable='netMHCIIpan'):
    if len(sequence) < 15:
        return pd.DataFrame(columns=COLUMNS)
    with tempfile.TemporaryDirectory(prefix='humanmpnn_netmhc_') as directory:
        peptides = Path(directory) / 'input.pep'
        peptides.write_text('\n'.join(sequence[i:i + 15] for i in range(len(sequence) - 14)) + '\n')
        result = subprocess.run([str(executable), '-f', str(peptides), '-inptype', '1',
                                 '-a', ','.join(ALLELES)], capture_output=True, text=True, check=True)
    table = '\n'.join(re.findall(r'^\s*\d+\s+\S+.*$', result.stdout, flags=re.MULTILINE))
    frame = (pd.read_csv(io.StringIO(table), sep=r'\s+', names=[*COLUMNS, '_binding_strength'], dtype=str,
                         keep_default_na=False, engine='python').reindex(columns=COLUMNS).fillna('')
             if table else pd.DataFrame(columns=COLUMNS))
    if frame.empty or set(frame['MHC']) != set(ALLELES):
        raise RuntimeError('NetMHCIIpan did not return predictions for all nine alleles')
    for col in ['Core_Rel', 'Rank_EL', 'Score_EL']:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    return frame.loc[frame['Rank_EL'] <= 5.0].reset_index(drop=True)
