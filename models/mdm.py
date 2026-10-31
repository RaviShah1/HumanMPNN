"""Filtered-edge GAT masked-diffusion model"""
import torch
from torch import nn
from .graph import ProteinFeatures, EncLayer, DecLayer, gather_nodes, cat_neighbors_nodes


class MDMDecoder(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3):
        super().__init__()
        self.W_s = nn.Embedding(22, hidden_dim)
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=0.1,
                     layer_type='gat', num_heads=1) for _ in range(num_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, 21)

    def forward(self, h_V, h_E, E_idx, S, mask, mask_attend):
        h_ES = cat_neighbors_nodes(self.W_s(S), h_E, E_idx)
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_V = layer(h_V, h_ESV, mask, mask_attend)
        return self.W_out(h_V)


class HumanMPNNMDM(nn.Module):
    def __init__(self, k_neighbors=48):
        super().__init__()
        self.features = ProteinFeatures(128, 128, top_k=k_neighbors, augment_eps=0.)
        self.W_e = nn.Linear(128, 128)
        self.encoder_layers = nn.ModuleList([
            EncLayer(128, 256, dropout=0.1, layer_type='gat', num_heads=1)
            for _ in range(3)
        ])
        self.decoder = MDMDecoder()
        self.mask_idx = 21

    def encode(self, X, mask, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((*E.shape[:2], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        mask_attend = mask.unsqueeze(-1) * gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        return h_V, h_E, E_idx, mask_attend

    def forward(self, X, S, mask, residue_idx, chain_encoding_all):
        h_V, h_E, E_idx, mask_attend = self.encode(X, mask, residue_idx, chain_encoding_all)
        return self.decoder(h_V, h_E, E_idx, S, mask, mask_attend).log_softmax(-1)

    @torch.inference_mode()
    def sample(self, X, S_true, mask, chain_M, residue_idx, chain_encoding_all,
               temperature=0.1, omit_AAs_np=None):
        """Original length-step random-reveal schedule, including its RNG draws."""
        h_V, h_E, E_idx, mask_attend = self.encode(X, mask, residue_idx, chain_encoding_all)
        designed = (chain_M * mask).bool()
        S = S_true.masked_fill(designed, self.mask_idx)
        omit = (torch.as_tensor(omit_AAs_np, device=X.device, dtype=torch.float32)
                if omit_AAs_np is not None else None)
        for n in range(X.shape[1]):
            masked = S == self.mask_idx
            logits = self.decoder(h_V, h_E, E_idx, S, mask, mask_attend)
            if omit is not None:
                logits = logits - omit.view(1, 1, -1) * 1e8
            logits = logits.double()
            if temperature != 0:
                logits = logits / temperature
                logits = logits - torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            tokens = logits.log_softmax(-1).max(-1).indices
            scores = torch.rand_like(logits[..., 0])
            scores = scores.masked_fill(~designed, float('inf'))
            scores[designed & ~masked] = 1.0
            cutoff = scores.sort(-1).values[:, X.shape[1] - 1 - n].unsqueeze(-1)
            remask = scores < cutoff
            S[remask] = self.mask_idx
            S = torch.where(masked & ~remask, tokens, S)
            if not (S == self.mask_idx).any():
                break
        return S
