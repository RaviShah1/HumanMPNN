"""Score FASTA sequences or cached ESMC embeddings against the frozen human reference bank."""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
import torch


MODEL_ID = "biohub/ESMC-600M"
EMBED_DIM = 1152

# Nine reference groups; index 0 is Human and defines the "close margin".
GROUPS = [
    "Human", "Non-human mammals", "Reptiles/Birds", "Amphibians", "Fish",
    "Invertebrates", "Fungi", "Plants", "Bacteria",
]


# ----------------------- FASTA IO -----------------------

def read_fasta(path: Path):
    return [(record.id, str(record.seq).upper()) for record in SeqIO.parse(path, 'fasta')]


def read_fasta_directory(path: Path):
    """Read .fa, .fasta, and .faa files as <stem>__<sample> records.

    The `sample=` field in the FASTA header (ProteinMPNN convention) is used
    when present; otherwise sample defaults to the running record count.
    """
    records = []
    sample_re = re.compile(r"(?:^|[, ]+)sample=([^, ]+)")
    for fasta in sorted(p for p in path.iterdir() if p.suffix.lower() in {'.fa', '.fasta', '.faa'}):
        for sample_index, record in enumerate(SeqIO.parse(fasta, 'fasta')):
            match = sample_re.search(record.description)
            sample = match.group(1) if match else str(sample_index)
            records.append((f"{fasta.stem}__{sample}", str(record.seq).upper()))
    return records


# ----------------------- ESMC embedding -----------------------

def _batches(length_sorted, max_tokens, max_batch):
    batch, max_len = [], 0
    for idx, length in length_sorted:
        tok_len = length + 2
        candidate = max(max_len, tok_len)
        if batch and (candidate * (len(batch) + 1) > max_tokens or
                      len(batch) >= max_batch):
            yield batch
            batch, max_len = [], 0
            candidate = tok_len
        batch.append(idx)
        max_len = candidate
    if batch:
        yield batch


def embed_records(records, device, max_tokens, max_batch):
    if not records:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {MODEL_ID} on {device}...", flush=True)
    model, info = AutoModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32,
        output_loading_info=True)
    if (info['missing_keys'] or info.get('mismatched_keys') or
            any(not k.startswith('lm_head.') for k in info['unexpected_keys'])):
        raise RuntimeError(f'ESMC checkpoint did not load completely: {info}')
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    order = sorted(((i, len(seq)) for i, (_n, seq) in enumerate(records)),
                   key=lambda row: row[1])
    output = np.empty((len(records), EMBED_DIM), dtype=np.float32)
    done = 0
    with torch.inference_mode():
        for idxs in _batches(order, max_tokens, max_batch):
            seqs = [records[i][1] for i in idxs]
            enc = tokenizer(seqs, return_tensors="pt", padding=True)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            for row, original in enumerate(idxs):
                length = len(records[original][1])
                output[original] = hidden[row, 1:1 + length].float().mean(0).cpu().numpy()
            done += len(idxs)
            if done % 5000 < len(idxs):
                print(f"  embedded {done:,}/{len(records):,}", flush=True)
    return output


# ----------------------- kNN scoring math -----------------------

def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def logmeanexp(values, tau, axis=-1):
    z = values / tau
    maximum = z.max(axis=axis, keepdims=True)
    return tau * (np.squeeze(maximum, axis=axis) +
                  np.log(np.exp(z - maximum).mean(axis=axis)))


def density(neighbors, k, method, tau):
    values = neighbors[:, :, :k]
    if method == "mean":
        return values.mean(axis=2)
    return logmeanexp(values, tau, axis=2)


def broad_other_density(densities, reducer, tau):
    other = densities[:, 1:]
    if reducer == "max":
        return other.max(axis=1)
    return logmeanexp(other, tau, axis=1)


def raw_human_score(densities, alpha, reducer, group_tau):
    close_margin = densities[:, 0] - densities[:, 1]
    broad_margin = densities[:, 0] - broad_other_density(
        densities, reducer, group_tau)
    return alpha * close_margin + (1 - alpha) * broad_margin


def load_artifact(path: Path):
    with np.load(path, allow_pickle=False) as z:
        art = {k: z[k] for k in z.files}
    return art


def score_embeddings(embeddings, artifact, device, batch):
    if batch < 1:
        raise ValueError('Batch size must be positive')
    x = normalize(embeddings.astype(np.float32, copy=False))
    refs = artifact["reference_embeddings"].astype(np.float32)
    ngroup, nref, dim = refs.shape
    if x.ndim != 2 or x.shape[1] != dim or ngroup != len(GROUPS) or nref == 0:
        raise ValueError('Query embeddings and reference bank have incompatible shapes')
    if not np.isfinite(x).all() or not np.isfinite(refs).all():
        raise ValueError('Embeddings must be finite')
    k = min(int(artifact["k"]), nref)
    ref_tensor = torch.from_numpy(refs.reshape(ngroup * nref, dim)).to(
        device=device, dtype=torch.float16)
    densities = np.empty((len(x), ngroup), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            stop = min(start + batch, len(x))
            q = torch.from_numpy(np.ascontiguousarray(x[start:stop])).to(
                device=device, dtype=torch.float16)
            sim = (q @ ref_tensor.T).reshape(stop - start, ngroup, nref)
            neighbors = torch.topk(sim, k, dim=2).values.float().cpu().numpy()
            densities[start:stop] = density(
                neighbors, k, str(artifact["density_method"]),
                float(artifact["neighbor_tau"]))
            if stop % 10000 < batch:
                print(f"  kNN scored {stop:,}/{len(x):,}", flush=True)
    raw = raw_human_score(
        densities, float(artifact["conditional_weight"]),
        str(artifact["other_reducer"]), float(artifact["group_tau"]))
    score = sigmoid(raw / float(artifact["output_temperature"]))
    predicted = np.argmax(densities, axis=1)
    return score, raw, densities, predicted


# ----------------------- CLI -----------------------

def cli():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--fasta", type=Path, help="Single .fa file")
    src.add_argument("--fasta-dir", type=Path,
                     help="Directory of FASTA files (ids become <stem>__<sample>)")
    src.add_argument("--embeddings-npz", type=Path,
                     help="Pre-computed mean-pooled ESMC-600M embeddings")
    p.add_argument("--artifact", type=Path, required=True,
                   help="Reference bank produced by setup_data.py")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--embedding-cache", type=Path,
                   help="If set, save the ESMC embeddings we compute to this NPZ")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-tokens", type=int, default=65536)
    p.add_argument("--max-batch", type=int, default=128)
    p.add_argument("--knn-batch", type=int, default=512)
    p.add_argument("--max-length", type=int, default=1023,
                   help="Validated length range; longer records are omitted")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if min(args.max_tokens, args.max_batch, args.knn_batch, args.max_length) < 1:
        p.error('Batch and length limits must be positive')
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    return args


def main():
    args = cli()
    artifact = load_artifact(args.artifact)

    skipped = []
    if args.fasta or args.fasta_dir:
        original = (read_fasta(args.fasta) if args.fasta
                    else read_fasta_directory(args.fasta_dir))
        if args.limit is not None:
            original = original[:args.limit]
        records = []
        for name, sequence in original:
            if not sequence or len(sequence) > args.max_length:
                skipped.append((name, len(sequence), "outside_validated_length"))
            else:
                records.append((name, sequence))
        names = np.array([n for n, _ in records])
        lengths = np.array([len(s) for _, s in records])
        embeddings = embed_records(records, args.device, args.max_tokens,
                                   args.max_batch)
        if args.embedding_cache:
            args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.embedding_cache,
                                ids=names, lengths=lengths,
                                embeddings=embeddings.astype(np.float16))
    else:
        with np.load(args.embeddings_npz) as z:
            embeddings = z["embeddings"].astype(np.float32)
            key = "ids" if "ids" in z.files else "accessions"
            names = z[key].astype(str)
            lengths = z["lengths"] if "lengths" in z.files else \
                np.full(len(names), -1)

    score, raw, densities, predicted = score_embeddings(
        embeddings, artifact, args.device, args.knn_batch)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    density_columns = [f"density_{g.lower().replace(' ', '_').replace('/', '_')}"
                       for g in GROUPS]
    frame = pd.DataFrame({
        'id': names, 'length': np.asarray(lengths, dtype=np.int64), 'human_proximity_score': score,
        'raw_margin': raw, 'predicted_taxonomy': np.asarray(GROUPS)[predicted],
        **dict(zip(density_columns, densities.T)), 'status': 'ok',
    })
    if skipped:
        skipped_frame = pd.DataFrame(skipped, columns=['id', 'length', 'status'])
        frame = pd.concat([frame, skipped_frame], ignore_index=True)
    frame.to_csv(args.output, sep='\t', index=False, float_format='%.8f')
    print(f"Wrote {len(names):,} scores ({len(skipped):,} skipped) to {args.output}")


if __name__ == "__main__":
    main()
