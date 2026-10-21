# -*- coding: utf-8 -*-
"""
ONE model: a GRU over the chronological structured-state + note event
sequence, queried at whatever landmark time T the Dataset sliced the
sequence at. The GRU's initial hidden state comes from the static features;
its final state is fused with them to give one logit per outcome.

Note: BatchNorm1d layers require batch_size > 1 in train mode. Keep batch_size > 1
for training; eval/inference is fine with any batch size including 1
(BatchNorm uses running statistics in eval mode).
"""

import torch
import torch.nn as nn

from common import config as cfg

MODALITY_PAD = 0
MODALITY_VITAL = 1  # "vital" event slot now covers the full vitals+labs structured state vector
MODALITY_NOTE = 2
NUM_VITAL_CHANNELS = len(cfg.STRUCTURED_CHANNELS)  # 6 vitals + 16 labs, see config.py STRUCTURED_CHANNELS


class EventSequenceModel(nn.Module):
    def __init__(self, num_static_features, num_labels, event_hidden=128, gru_hidden=128, note_hidden=768):
        super().__init__()
        self.event_hidden = event_hidden
        self.gru_hidden = gru_hidden

        self.vital_proj = nn.Linear(NUM_VITAL_CHANNELS, event_hidden)
        self.note_proj = nn.Linear(note_hidden, event_hidden)
        # padding_idx=MODALITY_PAD (0) -> padding positions contribute a
        # zero embedding here; they are excluded from the GRU entirely via
        # pack_padded_sequence anyway, this is just defensive.
        self.modality_embed = nn.Embedding(3, event_hidden, padding_idx=MODALITY_PAD)
        self.time_embed = nn.Linear(1, event_hidden)

        self.static_encoder = nn.Sequential(
            nn.Linear(num_static_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, gru_hidden),
        )

        self.gru = nn.GRU(input_size=event_hidden, hidden_size=gru_hidden, num_layers=1, batch_first=True)

        self.fusion_classifier = nn.Sequential(
            nn.Linear(gru_hidden + gru_hidden, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, num_labels),
        )

    def forward(self, content, modality, delta_t, lengths, static, return_embeds=False):
        """
        content: (B, L, 768) float
        modality: (B, L) long, 0=pad/1=vital/2=note
        delta_t: (B, L) float
        lengths: (B,) long, true (unpadded) sequence length per sample -- must be >= 1
        static: (B, F_static) float
        """
        vital_out = self.vital_proj(content[..., :NUM_VITAL_CHANNELS])
        note_out = self.note_proj(content)
        is_vital = (modality == MODALITY_VITAL).unsqueeze(-1)
        combined = torch.where(is_vital, vital_out, note_out)

        mod_emb = self.modality_embed(modality)
        time_emb = self.time_embed(delta_t.unsqueeze(-1))
        event_repr = combined + mod_emb + time_emb

        static_feat = self.static_encoder(static)
        h0 = static_feat.unsqueeze(0)  # (num_layers=1, B, gru_hidden)

        packed = nn.utils.rnn.pack_padded_sequence(
            event_repr, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed, h0)
        h_seq = h_n[-1]  # (B, gru_hidden) -- final hidden state of the last GRU layer

        fused = torch.cat([h_seq, static_feat], dim=1)
        logits = self.fusion_classifier(fused)

        if return_embeds:
            return logits, h_seq, static_feat, fused
        return logits
