"""Download UniProt reference sequences or select them from a local FASTA."""
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import io
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


def read_fasta(source):
    for record in SeqIO.parse(source, 'fasta'):
        name = record.id.split('|')[1] if '|' in record.id else record.id
        yield name, str(record.seq).upper()


def read_references(path):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False).to_dict('records')


def verify_records(records, rows):
    missing = [r['accession'] for r in rows if not records.get(r['accession'])]
    if missing:
        raise ValueError(f'{len(missing)} missing reference sequences: {missing[:10]}. '
                         'Use --source-fasta to supply a local UniProt FASTA.')


def fetch(references, output, source_fasta=None):
    rows = read_references(references)
    wanted = {r['accession'] for r in rows}
    if not wanted:
        raise ValueError('The reference list is empty')
    records = {}
    if source_fasta:
        records = {a: s for a, s in read_fasta(source_fasta) if a in wanted}
    else:
        cache = output.parent / (output.stem + '_downloads')
        cache.mkdir(parents=True, exist_ok=True)
        # Cache by accession, so changing the reference list cannot reuse a
        # batch belonging to different proteins.
        for accession in wanted:
            path = cache / f'{accession}.fasta'
            if path.exists():
                records.update(read_fasta(path))
        missing = sorted(wanted - records.keys())
        for start in range(0, len(missing), 100):
            chunk = missing[start:start + 100]
            query = ' OR '.join('accession:' + a for a in chunk)
            url = 'https://rest.uniprot.org/uniprotkb/stream?' + urlencode({'query': query, 'format': 'fasta'})
            for attempt in range(4):
                try:
                    with urlopen(url, timeout=120) as response:
                        text = response.read().decode()
                    downloaded = list(read_fasta(io.StringIO(text)))
                    if text.strip() and not downloaded:
                        raise ValueError('UniProt returned an unexpected response')
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            for accession, sequence in downloaded:
                if accession in wanted:
                    records[accession] = sequence
                    path = cache / f'{accession}.fasta'
                    temporary = path.with_suffix('.tmp')
                    SeqIO.write(SeqRecord(Seq(sequence), id=accession, description=''),
                                temporary, 'fasta-2line')
                    temporary.replace(path)
            print(f'Downloaded {len(wanted & records.keys())}/{len(wanted)} references', flush=True)
    verify_records(records, rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    SeqIO.write((SeqRecord(Seq(records[accession]), id=accession, description='')
                 for accession in sorted(wanted)), temporary, 'fasta-2line')
    temporary.replace(output)
    print(f'Wrote {len(wanted)} references: {output}')
