# -*- coding: utf-8 -*-
"""
Shared event-stream lookup utilities, used by both:
  - phase1_model/stage1_finetune_bert.py (needs a "snapshot at landmark T" view)
  - phase1_model/sequence_dataset.py (needs the full ordered event sequence up to T)

Built once per split at Dataset-construction time (groups events by stay_id into
sorted numpy arrays), so per-sample lookups at training time are O(log n)
binary searches instead of repeated pandas filtering -- matters at multi-million
row scale over many epochs.
"""

import numpy as np
import pandas as pd

from . import config as cfg


class EncounterEventIndex:
    """Fast per-encounter lookup over the Phase 0 output tables."""

    def __init__(self, static_df: pd.DataFrame, vitals_df: pd.DataFrame, notes_df: pd.DataFrame):
        """vitals_df is actually the combined vitals+labs STRUCTURED event
        stream (see build_event_streams.py's build_structured_event_stream)
        -- kept as "vitals_df" / "_vitals_by_stay" naming since it is still
        conceptually the same forward-filled state-vector stream, just wider
        (cfg.STRUCTURED_CHANNELS = 6 vitals + 16 labs): labs are extra
        channels of the same state vector, not a separate event modality.
        """
        self.static_df = static_df.reset_index(drop=True)
        self.stay_ids = self.static_df["stay_id"].astype(str).tolist()

        self._vitals_by_stay = {}
        for stay_id, g in vitals_df.groupby("stay_id"):
            g = g.sort_values("t_minutes")
            self._vitals_by_stay[str(stay_id)] = (
                g["t_minutes"].to_numpy(dtype=np.float64),
                g[cfg.STRUCTURED_CHANNELS].to_numpy(dtype=np.float32),
            )

        self._notes_by_stay = {}
        for stay_id, g in notes_df.groupby("stay_id"):
            g = g.sort_values("t_minutes")
            self._notes_by_stay[str(stay_id)] = (
                g["t_minutes"].to_numpy(dtype=np.float64),
                g["note_text"].tolist(),
            )

    def vitals_state_at(self, stay_id: str, t_minutes: float) -> np.ndarray:
        """Most recently known STRUCTURED_CHANNELS-wide state (vitals+labs)
        at or before t_minutes. Falls back to zeros (post-scaling, zero ==
        train-population mean) if this stay has no events at all -- should
        not happen for vitals given Phase 0 always seeds a t=0 triage row,
        but handled defensively (labs channels legitimately have no seed).
        """
        entry = self._vitals_by_stay.get(str(stay_id))
        if entry is None:
            return np.zeros(len(cfg.STRUCTURED_CHANNELS), dtype=np.float32)
        times, values = entry
        idx = np.searchsorted(times, t_minutes, side="right") - 1
        if idx < 0:
            return np.zeros(len(cfg.STRUCTURED_CHANNELS), dtype=np.float32)
        return values[idx]

    def vitals_events_upto(self, stay_id: str, t_minutes: float):
        """All (t, state_vector) pairs with t <= t_minutes, chronological.
        Used by the sequence model (Phase 1 proper), not by Stage 1.
        """
        entry = self._vitals_by_stay.get(str(stay_id))
        if entry is None:
            return np.array([]), np.zeros((0, len(cfg.STRUCTURED_CHANNELS)), dtype=np.float32)
        times, values = entry
        idx = np.searchsorted(times, t_minutes, side="right")
        return times[:idx], values[:idx]

    def notes_upto(self, stay_id: str, t_minutes: float):
        """List[str] of note texts with t <= t_minutes, chronological."""
        entry = self._notes_by_stay.get(str(stay_id))
        if entry is None:
            return []
        times, texts = entry
        idx = np.searchsorted(times, t_minutes, side="right")
        return texts[:idx]

    def notes_events_upto(self, stay_id: str, t_minutes: float):
        """(times[:idx], texts[:idx]) -- used by the sequence model to pair
        each note with its Δt for the event-stream GRU input.
        """
        entry = self._notes_by_stay.get(str(stay_id))
        if entry is None:
            return np.array([]), []
        times, texts = entry
        idx = np.searchsorted(times, t_minutes, side="right")
        return times[:idx], texts[:idx]


class NoteEmbeddingIndex:
    """Looks up cached per-note BERT embeddings (from
    precompute_note_embeddings.py) by (stay_id, t_minutes). Built once per
    split from the cached .npy + index parquet.
    """

    def __init__(self, embeddings: np.ndarray, index_df: pd.DataFrame):
        self.embeddings = embeddings
        # shape[1] (embedding dim) is meaningful even for a (0, hidden) empty
        # array -- precompute_note_embeddings.py preserves it on purpose for
        # exactly this reason (a split can legitimately have zero notes, e.g.
        # a tiny val set, without losing track of what hidden size the rest
        # of the pipeline must agree on).
        self.hidden_size = embeddings.shape[1] if embeddings.ndim == 2 else 0
        self._by_stay = {}
        for stay_id, g in index_df.groupby("stay_id"):
            g = g.sort_values("t_minutes")
            self._by_stay[str(stay_id)] = (
                g["t_minutes"].to_numpy(dtype=np.float64),
                g["row_idx"].to_numpy(dtype=np.int64),
            )

    @classmethod
    def from_cache_dir(cls, cache_dir: str, split: str):
        embeddings = np.load(f"{cache_dir}/{split}_note_embeddings.npy")
        index_df = pd.read_parquet(f"{cache_dir}/{split}_note_embeddings_index.parquet")
        return cls(embeddings, index_df)

    def embeddings_upto(self, stay_id: str, t_minutes: float):
        """(times[:idx], embeddings[:idx, hidden]) for events at/before t_minutes."""
        entry = self._by_stay.get(str(stay_id))
        if entry is None:
            return np.array([]), np.zeros((0, self.hidden_size), dtype=np.float32)
        times, row_idx = entry
        idx = np.searchsorted(times, t_minutes, side="right")
        return times[:idx], self.embeddings[row_idx[:idx]]


def landmark_upper_bound(data_boundary_minutes, max_landmark_minutes):
    """Highest minute value a landmark T may take for one encounter.

    data_boundary_minutes is Phase 0's min(dx_boundary_minutes,
    outtime_minutes) -- already NaN-safely combined there (see
    build_event_streams.py). Kept as a single argument here (rather than
    taking dx/outtime separately and re-combining) so there is exactly one
    place in the whole codebase that decides how those two get merged;
    Phase 0's event-stream truncation and Phase 1's landmark sampling
    ceiling can never drift out of sync as a result. NaN means "no
    constraint from dx_time or outtime for this encounter".
    """
    bounds = [max_landmark_minutes]
    if data_boundary_minutes is not None and not np.isnan(data_boundary_minutes):
        bounds.append(max(0.0, data_boundary_minutes))
    return float(min(bounds))


def sample_landmark_minutes(data_boundary_minutes, max_landmark_minutes, rng: np.random.Generator):
    """Continuous uniform sample in [0, upper_bound]."""
    upper = landmark_upper_bound(data_boundary_minutes, max_landmark_minutes)
    if upper <= 0:
        return 0.0
    return float(rng.uniform(0.0, upper))
