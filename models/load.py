"""Loading of exported inference tensors."""
from pathlib import Path
import torch
from .ar import ProteinMPNN
from .mdm import HumanMPNNMDM

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    'proteinmpnn': 'proteinmpnn/v_48_020.pt',
    'solublempnn': 'solublempnn/v_48_020.pt',
    'humanmpnn_ar_pretrained': 'humanmpnn/ar_pretrained.pt',
    'humanmpnn_ar_finetuned': 'humanmpnn/ar_finetuned.pt',
    'humanmpnn_mdm_pretrained': 'humanmpnn/mdm_pretrained.pt',
    'humanmpnn_mdm_finetuned': 'humanmpnn/mdm_finetuned.pt',
}


def load_model(name, device='cpu', weights_dir=None):
    path = Path(weights_dir or ROOT / 'weights') / CHECKPOINTS[name]
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['architecture'] == 'mdm':
        model = HumanMPNNMDM(k_neighbors=checkpoint['num_edges'])
    elif checkpoint['architecture'] == 'ar':
        model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                            hidden_dim=128, k_neighbors=checkpoint['num_edges'], augment_eps=0.)
    else:
        raise ValueError(f"Unsupported architecture in {path}")
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.to(device).eval()
