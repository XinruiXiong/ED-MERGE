# -*- coding: utf-8 -*-
"""
Phase 1 core: the actual "dynamic" dataset. Each item is one encounter's
chronological event sequence (vitals state updates + cached note embeddings,
merged and sorted by time) truncated at a landmark time T.

Content width is unified at 768 (the BERT hidden size) so both modalities can
share one padded batch tensor: a vital event's 6-d state vector occupies the
first 6 slots and the rest are zero; a note event's cached 768-d embedding
fills the whole slot. The model (sequence_model.py) applies separate
modality-specific projections and selects per-event by modality tag. This dataset does NOT run BERT; it only reads
already-cached embeddings from precompute_note_embeddings.py.

Landmark sampling: continuous uniform in [0, upper_bound] per encounter,
resampled every epoch for train, sampled once (fixed) for val/test so
validation metrics are comparable across epochs. See
common/event_utils.sample_landmark_minutes.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from common import config as cfg
from common.event_utils import EncounterEventIndex, NoteEmbeddingIndex, sample_landmark_minutes

MODALITY_PAD = 0
MODALITY_VITAL = 1
MODALITY_NOTE = 2

# Inference-time ablation components (see evaluate_ablation.py).
ABLATION_COMPONENTS = ("static", "text", "vital", "lab")
_VITAL_IDX = [cfg.STRUCTURED_CHANNELS.index(c) for c in cfg.VITAL_CHANNELS]
_LAB_IDX = [cfg.STRUCTURED_CHANNELS.index(c) for c in cfg.LAB_CHANNELS]


def drop_structured_channels(times, states, drop_vital, drop_lab):
    """Remove vital and/or lab information from a forward-filled state stream.
    Dropped channels are set to 0 (post-scaling == train mean, i.e. exactly how
    a never-observed channel looks), then events that no longer change any
    remaining channel are removed, so the timing of the dropped modality's
    measurements is not left behind either. Dropping both removes all
    structured events."""
    if drop_vital and drop_lab:
        return times[:0], states[:0]
    states = states.copy()
    states[:, _VITAL_IDX if drop_vital else _LAB_IDX] = 0.0
    if len(states) <= 1:
        return times, states
    keep = np.ones(len(states), dtype=bool)
    keep[1:] = np.any(states[1:] != states[:-1], axis=1)
    return times[keep], states[keep]


class EventSequenceDataset(Dataset):
    def __init__(self, static_df, vitals_index: EncounterEventIndex, note_embed_index: NoteEmbeddingIndex,
                 outcome_cols, static_cols, max_landmark_minutes, seed, resample_each_epoch: bool,
                 fixed_landmark_minutes=None, drop_modalities=()):
        """
        fixed_landmark_minutes: if set (a single float), EVERY item uses this
        exact landmark instead of a sampled one. Used by evaluate_landmarks.py
        to score the fixed landmark grid from one checkpoint.
        Mutually exclusive with resample_each_epoch=True.

        drop_modalities: subset of ABLATION_COMPONENTS removed at inference
        (evaluate_ablation.py). "text" drops note events, "vital"/"lab" see
        drop_structured_channels, "static" sets the static vector to 0 (train
        mean). Empty for training and normal evaluation.
        """
        unknown = set(drop_modalities) - set(ABLATION_COMPONENTS)
        assert not unknown, f"unknown ablation components: {unknown}"
        self.drop_modalities = set(drop_modalities)
        self.static_df = static_df.reset_index(drop=True)
        self.vitals_index = vitals_index
        self.note_embed_index = note_embed_index
        self.outcome_cols = outcome_cols
        self.static_cols = static_cols
        self.max_landmark_minutes = max_landmark_minutes
        self.resample_each_epoch = resample_each_epoch
        self.fixed_landmark_minutes = fixed_landmark_minutes
        self.rng = np.random.default_rng(seed)

        # Content width is driven by whatever the cached note embeddings
        # actually are (768 for real BioClinicalBERT), never hardcoded --
        # this also lets tests substitute a smaller stand-in encoder. Falls
        # back to the config default only if the cache is empty (e.g. a
        # split with zero notes) and note_embed_index.hidden_size is 0.
        self.content_width = note_embed_index.hidden_size or 768
        assert self.content_width >= len(cfg.STRUCTURED_CHANNELS), (
            "note embedding hidden size must be >= number of structured (vitals+labs) channels "
            f"(got {self.content_width} < {len(cfg.STRUCTURED_CHANNELS)})"
        )

        self.landmarks = np.zeros(len(self.static_df), dtype=np.float64)
        self.resample_landmarks()

    def resample_landmarks(self):
        data_boundary = self.static_df["data_boundary_minutes"].to_numpy()

        if self.fixed_landmark_minutes is not None:
            self.landmarks[:] = float(self.fixed_landmark_minutes)
            # Still respect each encounter's own data_boundary_minutes ceiling
            # even in fixed-landmark eval mode -- a patient whose data stream
            # is cut off (by dx_time or ED departure) at 15 min has no events
            # past 15 min regardless of the reporting grid point.
            from common.event_utils import landmark_upper_bound
            for i in range(len(self.static_df)):
                upper = landmark_upper_bound(data_boundary[i], self.max_landmark_minutes)
                self.landmarks[i] = min(self.landmarks[i], upper) if upper > 0 else 0.0
            return

        for i in range(len(self.static_df)):
            self.landmarks[i] = sample_landmark_minutes(data_boundary[i], self.max_landmark_minutes, self.rng)

    def __len__(self):
        return len(self.static_df)

    def __getitem__(self, idx):
        row = self.static_df.iloc[idx]
        stay_id = str(row["stay_id"])
        T = float(self.landmarks[idx])

        v_times, v_states = self.vitals_index.vitals_events_upto(stay_id, T)
        n_times, n_embeds = self.note_embed_index.embeddings_upto(stay_id, T)

        drop = self.drop_modalities
        if "text" in drop:
            n_times, n_embeds = n_times[:0], n_embeds[:0]
        if "vital" in drop or "lab" in drop:
            v_times, v_states = drop_structured_channels(v_times, v_states, "vital" in drop, "lab" in drop)

        n_vital = v_times.shape[0]
        n_note = n_times.shape[0]
        n_total = n_vital + n_note

        content = np.zeros((max(n_total, 1), self.content_width), dtype=np.float32)
        modality = np.zeros(max(n_total, 1), dtype=np.int64)
        times = np.zeros(max(n_total, 1), dtype=np.float64)

        if n_vital > 0:
            content[:n_vital, :len(cfg.STRUCTURED_CHANNELS)] = v_states
            modality[:n_vital] = MODALITY_VITAL
            times[:n_vital] = v_times
        if n_note > 0:
            content[n_vital:n_vital + n_note, :] = n_embeds
            modality[n_vital:n_vital + n_note] = MODALITY_NOTE
            times[n_vital:n_vital + n_note] = n_times

        if n_total == 0:
            # Should not happen (Phase 0 always seeds a t=0 vitals event),
            # but guard anyway: emit a single zero "vital" placeholder event.
            modality[0] = MODALITY_VITAL

        order = np.argsort(times[:max(n_total, 1)], kind="stable")
        content = content[order]
        modality = modality[order]
        times = times[order]

        delta_t = np.zeros_like(times)
        if len(times) > 1:
            delta_t[1:] = np.diff(times)
        delta_t = np.log1p(np.clip(delta_t, 0, None)).astype(np.float32)

        static_vals = row[self.static_cols].to_numpy(dtype=np.float32)
        if "static" in drop:
            static_vals = np.zeros_like(static_vals)
        labels = row[self.outcome_cols].to_numpy(dtype=np.float32)

        return {
            "content": content.astype(np.float32),
            "modality": modality.astype(np.int64),
            "delta_t": delta_t,
            "static": static_vals,
            "labels": labels,
            "stay_id": stay_id,
            "landmark_T": T,
            "n_vital": n_vital,
            "n_note": n_note,
        }


def collate_sequence_batch(batch):
    batch_size = len(batch)
    lengths = [item["content"].shape[0] for item in batch]
    max_len = max(lengths)
    hidden = batch[0]["content"].shape[1]

    content = torch.zeros((batch_size, max_len, hidden), dtype=torch.float32)
    modality = torch.zeros((batch_size, max_len), dtype=torch.long)
    delta_t = torch.zeros((batch_size, max_len), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        L = lengths[i]
        content[i, :L] = torch.from_numpy(item["content"])
        modality[i, :L] = torch.from_numpy(item["modality"])
        delta_t[i, :L] = torch.from_numpy(item["delta_t"])
        mask[i, :L] = True

    static = torch.stack([torch.tensor(item["static"]) for item in batch], dim=0)
    labels = torch.stack([torch.tensor(item["labels"]) for item in batch], dim=0)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    stay_ids = [item["stay_id"] for item in batch]
    landmarks = torch.tensor([item["landmark_T"] for item in batch], dtype=torch.float32)
    n_vitals_seen = torch.tensor([item["n_vital"] for item in batch], dtype=torch.long)
    n_notes_seen = torch.tensor([item["n_note"] for item in batch], dtype=torch.long)

    return {
        "content": content, "modality": modality, "delta_t": delta_t, "mask": mask,
        "lengths": lengths_t, "static": static, "labels": labels,
        "stay_ids": stay_ids, "landmark_T": landmarks,
        "n_vitals_seen": n_vitals_seen, "n_notes_seen": n_notes_seen,
    }
