import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base import BaseStep


logger = logging.getLogger(__name__)


@dataclass
class MarioAlignmentResult:
    events_behavior: np.ndarray
    event_id_behavior: Dict[str, int]
    stim_channel: Optional[str]
    fps_est_eeg: Optional[float]
    fps_expected_meta: Optional[float]
    offset_s: Optional[float]
    drift_a: float
    drift_b: float
    n_segment_starts: int


def _summarize_events(ev: Optional[np.ndarray]) -> Dict[str, object]:
    if ev is None or len(ev) == 0:
        return {"n": 0, "codes": {}, "iei_median": None}
    codes, counts = np.unique(ev[:, 2], return_counts=True)
    return {
        "n": int(len(ev)),
        "codes": {int(c): int(n) for c, n in zip(codes, counts)},
        "iei_median": float(np.median(np.diff(ev[:, 0]))) if len(ev) > 1 else None,
    }


def _find_stim_channel(raw: mne.io.BaseRaw) -> Optional[str]:
    # Prefer explicit 'Trigger' then any stim
    if 'Trigger' in raw.ch_names:
        return 'Trigger'
    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    for ch, t in types.items():
        if t == 'stim':
            return ch
    return None


def _detect_frame_pulses(raw: mne.io.BaseRaw, stim_ch: str, mask_bit: int = 2) -> Tuple[np.ndarray, Optional[float]]:
    ev = mne.find_events(
        raw,
        stim_channel=stim_ch,
        mask=mask_bit,
        mask_type='and',
        shortest_event=1,
        uint_cast=True,
        consecutive=False,
        verbose=False,
    )
    fps_est = None
    if ev is not None and len(ev) > 1:
        dt = np.median(np.diff(ev[:, 0])) / raw.info['sfreq']
        if dt > 0:
            fps_est = 1.0 / dt
    return ev, fps_est


def _detect_bit_events(raw: mne.io.BaseRaw, stim_ch: str, mask_bit: int) -> np.ndarray:
    """Detect events on a given digital bit of the stim channel.
    Returns MNE-style events array (n, 3) with sample indices in column 0.
    """
    try:
        ev = mne.find_events(
            raw,
            stim_channel=stim_ch,
            mask=mask_bit,
            mask_type='and',
            shortest_event=1,
            uint_cast=True,
            consecutive=False,
            verbose=False,
        )
        return ev if ev is not None else np.empty((0, 3), dtype=int)
    except Exception:
        return np.empty((0, 3), dtype=int)


def _expected_fps_from_metadata(df: pd.DataFrame) -> Optional[float]:
    try:
        meta = []
        if 'game' in df.columns:
            meta.extend(df['game'].dropna().astype(str).unique().tolist())
        if 'stim_file' in df.columns:
            meta.extend(df['stim_file'].dropna().astype(str).unique().tolist())
        blob = " ".join(meta)
        if any(tok in blob for tok in ('Nes', 'NES', 'nes')):
            return 60.0988  # NTSC NES
    except Exception:
        pass
    return None


def _derive_behavior_onsets(df: pd.DataFrame, fps_guess: float) -> Tuple[np.ndarray, np.ndarray, str]:
    # Pick a label column: prefer first string-like
    label_col = df.columns[0]
    if df[label_col].dtype.kind in 'biufc':
        cand = [c for c in df.columns if df[c].dtype == object]
        if cand:
            label_col = cand[0]
    labels = df[label_col].astype(str).values
    # Prefer 'onset'; else fall back to frame-like columns
    cols_lower = {c.lower(): c for c in df.columns}
    onsets = None
    if 'onset' in cols_lower:
        c = cols_lower['onset']
        try:
            onsets = pd.to_numeric(df[c], errors='coerce').values
        except Exception:
            onsets = None
    if onsets is None:
        for alt in ('frame', 'sample', 'nframes'):
            if alt in cols_lower:
                c = cols_lower[alt]
                vals = pd.to_numeric(df[c], errors='coerce').values
                onsets = vals / fps_guess
                break
    if onsets is None:
        onsets = np.array([], dtype=float)
    return onsets, labels, label_col


def _segment_starts_from_pulses(ev: np.ndarray, sfreq: float, gap_thr_s: float = 0.5) -> np.ndarray:
    t = ev[:, 0] / sfreq
    if t.size == 0:
        return np.array([], dtype=float)
    diffs = np.diff(t)
    gap_idx = np.where(diffs > gap_thr_s)[0]
    seg_starts_idx = np.r_[0, gap_idx + 1]
    return t[seg_starts_idx]


def _behavior_level_onsets(df: pd.DataFrame, fps_guess: float) -> np.ndarray:
    """Extract onset times (seconds) for level changes from behavior TSV.
    Returns empty array if no level info is present.
    """
    if 'level' not in df.columns:
        return np.array([], dtype=float)

    lvl_raw = df['level']
    lvl_str = lvl_raw.astype(str).str.strip()
    invalid_tokens = {"", "nan", "none", "null"}
    valid = (~lvl_raw.isna()) & (~lvl_str.str.lower().isin(invalid_tokens))

    cols_lower = {c.lower(): c for c in df.columns}
    onset_vals = None
    if 'onset' in cols_lower:
        onset_vals = pd.to_numeric(df[cols_lower['onset']], errors='coerce')
    else:
        for alt in ('frame', 'sample', 'nframes'):
            if alt in cols_lower:
                onset_vals = pd.to_numeric(df[cols_lower[alt]], errors='coerce') / max(fps_guess, 1e-6)
                break
    if onset_vals is None:
        return np.array([], dtype=float)

    lvl_clean = lvl_str.where(valid, other=np.nan)
    lvl_changes = pd.Series(lvl_clean).ne(pd.Series(lvl_clean).shift(1)) & valid
    t = onset_vals[lvl_changes].values
    t = t[~np.isnan(t)]
    return np.asarray(t, dtype=float)


def _robust_offset(seg_starts_t: np.ndarray, labels: np.ndarray, onsets: np.ndarray,
                   candidates: List[str], max_dist_s: float = 3.0) -> Tuple[Optional[float], str, int, Optional[float]]:
    best = (None, '', 0, None)
    for lab in candidates:
        mask = np.array([(str(x) == lab) for x in labels])
        beh_t = onsets[mask]
        if seg_starts_t.size == 0 or beh_t.size == 0:
            continue
        deltas = []
        for bt in beh_t:
            j = np.argmin(np.abs(seg_starts_t - bt))
            dt = seg_starts_t[j] - bt
            if abs(dt) <= max_dist_s:
                deltas.append(dt)
        if not deltas:
            continue
        deltas = np.array(deltas)
        off = float(np.median(deltas))
        std = float(np.std(deltas))
        if len(deltas) > best[2] or (len(deltas) == best[2] and (best[3] is None or std < (best[3] or std+1))):
            best = (off, lab, len(deltas), std)
    return best


def _snap_to_frames(times_s: np.ndarray, frame_times: np.ndarray, frame_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if times_s.size == 0:
        return np.array([], dtype=int), np.array([])
    idx = np.searchsorted(frame_times, times_s)
    idx = np.clip(idx, 1, len(frame_times) - 1)
    prev = frame_times[idx - 1]
    nextt = frame_times[idx]
    choose_prev = (times_s - prev) <= (nextt - times_s)
    nearest_idx = np.where(choose_prev, idx - 1, idx)
    nearest_samples = frame_samples[nearest_idx]
    nearest_times = frame_times[nearest_idx]
    return nearest_samples.astype(int), nearest_times


def _fit_drift(aligned: np.ndarray, snapped: np.ndarray) -> Tuple[float, float, float]:
    # Fit delta = snapped - aligned ≈ a + b*t
    if aligned.size == 0 or snapped.size == 0:
        return 0.0, 0.0, 0.0
    n = min(aligned.size, snapped.size)
    aligned = aligned[:n]
    snapped = snapped[:n]
    delta = snapped - aligned
    try:
        b, a = np.polyfit(aligned, delta, 1)
        rate_ms_min = b * 60000.0
        return float(a), float(b), float(rate_ms_min)
    except Exception:
        return 0.0, 0.0, 0.0


def _build_behavior_events(
    df: pd.DataFrame,
    offset_s: float,
    drift_a: float,
    drift_b: float,
    fps_guess: float,
    frame_times: np.ndarray,
    frame_samples: np.ndarray,
    shift_level_starts_frames: int = 0,
) -> Tuple[np.ndarray, Dict[str, int]]:
    labels = df[df.columns[0]].astype(str) if df[df.columns[0]].dtype == object else df.select_dtypes(include='object').iloc[:, 0]
    labels = labels.astype(str).values

    def beh_times(mask: np.ndarray) -> np.ndarray:
        if mask is None or (not mask.any()):
            return np.array([])
        cols_lower = {c.lower(): c for c in df.columns}
        if 'onset' in cols_lower:
            vals = pd.to_numeric(df.loc[mask, cols_lower['onset']], errors='coerce').values
            return vals[~np.isnan(vals)]
        for alt in ('frame', 'sample', 'nframes'):
            if alt in cols_lower:
                vals = pd.to_numeric(df.loc[mask, cols_lower[alt]], errors='coerce').values
                vals = vals[~np.isnan(vals)] / fps_guess
                return vals
        return np.array([])

    events_list: List[np.ndarray] = []
    event_id: Dict[str, int] = {}

    # Keypresses (per-key if available)
    key_col = None
    cols_lower = {c.lower(): c for c in df.columns}
    for cname in ('key', 'button', 'response'):
        if cname in cols_lower:
            key_col = cols_lower[cname]
            break
    # Include explicit 'keypress' labels, or anything containing 'key'/'press',
    # but exclude releases (e.g., 'key_release', 'release', 'keyup')
    def _is_press_label(s: str) -> bool:
        s = s.lower()
        if 'release' in s or 'keyup' in s or 'key_up' in s:
            return False
        return ('key' in s) or ('press' in s)
    kp_mask = (labels == 'keypress') | np.array([_is_press_label(str(s)) for s in labels])
    kp_t = beh_times(kp_mask)
    kp_t_al = kp_t + offset_s
    if drift_a or drift_b:
        kp_t_al = kp_t_al + (drift_a + drift_b * kp_t_al)
    kp_samp, _ = _snap_to_frames(kp_t_al, frame_times, frame_samples)

    if kp_samp.size:
        if key_col is not None:
            kp_df = df.loc[kp_mask].copy()
            time_vals = None
            if 'onset' in cols_lower:
                time_vals = pd.to_numeric(kp_df[cols_lower['onset']], errors='coerce')
            else:
                for alt in ('frame', 'sample', 'nframes'):
                    if alt in cols_lower:
                        time_vals = pd.to_numeric(kp_df[cols_lower[alt]], errors='coerce') / fps_guess
                        break
            time_vals = (time_vals.fillna(np.nan).values + offset_s)
            if drift_a or drift_b:
                time_vals = time_vals + (drift_a + drift_b * time_vals)
            inb = (time_vals >= 0) & (time_vals <= frame_times[-1])
            key_vals = kp_df[key_col].fillna('UNKNOWN').astype(str).str.strip().str.upper().str.replace(' ', '_').values
            key_vals = key_vals[inb]
            kp_samp2, _ = _snap_to_frames(time_vals[inb], frame_times, frame_samples)
            uniq = sorted(np.unique(key_vals).tolist())
            base = 930
            key_to_code = {k: (i + base) for i, k in enumerate(uniq)}
            for s, k in zip(kp_samp2, key_vals):
                events_list.append(np.array([int(s), 0, int(key_to_code.get(k, 999))]))
            event_id.update({f"kp_{k}": code for k, code in key_to_code.items()})
        else:
            events_list.append(np.c_[kp_samp, np.zeros_like(kp_samp), np.full_like(kp_samp, 900)])
            event_id['beh_keypress'] = 900

    # Block starts
    for name, code in (('fixation_dot', 910), ('gym-retro_game', 920)):
        m = (labels == name)
        t = beh_times(m)
        t_al = t + offset_s
        if drift_a or drift_b:
            t_al = t_al + (drift_a + drift_b * t_al)
        s, _ = _snap_to_frames(t_al, frame_times, frame_samples)
        if s.size:
            events_list.append(np.c_[s, np.zeros_like(s), np.full_like(s, code)])
            event_id[name.replace('-', '_')] = code

    # Level starts from 'level' column changes (ignore NaN/empty/None)
    if 'level' in df.columns:
        lvl_raw = df['level']
        lvl_str = lvl_raw.astype(str).str.strip()
        # Treat true NaN/None/empty and literal 'nan'/'none'/'null' as missing
        invalid_tokens = {"", "nan", "none", "null"}
        valid = (~lvl_raw.isna()) & (~lvl_str.str.lower().isin(invalid_tokens))

        onset_vals = None
        if 'onset' in cols_lower:
            onset_vals = pd.to_numeric(df[cols_lower['onset']], errors='coerce')
        else:
            for alt in ('frame', 'sample', 'nframes'):
                if alt in cols_lower:
                    onset_vals = pd.to_numeric(df[cols_lower[alt]], errors='coerce') / fps_guess
                    break
        if onset_vals is not None:
            # Only consider changes among valid level labels
            lvl_clean = lvl_str.where(valid, other=np.nan)
            lvl_changes = pd.Series(lvl_clean).ne(pd.Series(lvl_clean).shift(1)) & valid
            if lvl_changes.any():
                t = onset_vals[lvl_changes].values + offset_s
                names = lvl_str[lvl_changes].values
                if drift_a or drift_b:
                    t = t + (drift_a + drift_b * t)
                s, _ = _snap_to_frames(t, frame_times, frame_samples)
                # Optional cosmetic shift to avoid visual overlap with keypresses
                if shift_level_starts_frames and s.size:
                    if frame_samples.size > 1:
                        step = int(round(np.median(np.diff(frame_samples))))
                    else:
                        # Fall back to fps_guess
                        step = max(1, int(round((1.0 / max(fps_guess, 1e-6)) * (frame_times.shape[0] and (frame_samples[-1] / frame_times[-1]) or 1))))
                    s = (s + shift_level_starts_frames * step).astype(int)
                    s = np.clip(s, 0, None)
                base = 1000
                lvl_map: Dict[str, int] = {}
                for samp, nm in zip(s, names):
                    key = f"lvl_{str(nm).strip()}"
                    if key not in lvl_map:
                        lvl_map[key] = base + len(lvl_map)
                    events_list.append(np.array([int(samp), 0, int(lvl_map[key])]))
                if lvl_map:
                    event_id.update(lvl_map)

    if not events_list:
        return np.empty((0, 3), dtype=int), {}
    ev = np.vstack(events_list).astype(int)
    ev = np.unique(ev, axis=0)
    ev = ev[np.argsort(ev[:, 0])]
    return ev, event_id


class MarioEventAlignmentStep(BaseStep):
    """
    Detects frame-sync triggers in the EEG, aligns behavior events to EEG time,
    applies robust offset and optional drift correction, and builds MNE-style
    events for keypresses, blocks, and level starts. Results are stored in
    raw.info['temp'] under 'behavior_events' and 'behavior_event_id'.

    Parameters (self.params):
    - behav_tsv_path: str (required)
    - stim_channel: str = 'Trigger'
    - frame_bit_mask: int = 2
    - gap_threshold_s: float = 0.5
    - max_match_dist_s: float = 3.0
    - drift_threshold_ms_per_min: float = 1.0
    - prefer_meta_fps: bool = True  # NES metadata (≈60.099 Hz) over EEG pulses
    - plot: bool = True             # save static plots
    - summary_dir: Optional[str]    # where to save plots/TSV
    """

    def run(self, data: mne.io.BaseRaw) -> mne.io.BaseRaw:
        if data is None:
            raise ValueError("[MarioEventAlignmentStep] No Raw provided")

        raw = data
        params = self.params
        beh_path = Path(params.get('behav_tsv_path', ''))
        if not beh_path.exists():
            raise FileNotFoundError(f"Behavior TSV not found: {beh_path}")

        stim_ch = params.get('stim_channel') or _find_stim_channel(raw) or 'Trigger'
        mask_bit = int(params.get('frame_bit_mask', 2))
        gap_thr = float(params.get('gap_threshold_s', 0.5))
        max_dist = float(params.get('max_match_dist_s', 3.0))
        drift_thr_ms_min = float(params.get('drift_threshold_ms_per_min', 1.0))
        prefer_meta_fps = bool(params.get('prefer_meta_fps', True))
        do_plot = bool(params.get('plot', True))
        summary_dir = Path(params.get('summary_dir', 'figures/summary'))
        require_frame_pulses = bool(params.get('require_frame_pulses', True))
        shift_level_frames = int(params.get('shift_level_starts_frames', 0))
        align_first_event = bool(params.get('align_first_event', False))
        disable_drift = bool(params.get('disable_drift', params.get('ignore_drift', False)))
        force_zero_offset = bool(params.get('force_zero_offset', False))
        summary_dir.mkdir(parents=True, exist_ok=True)

        # 1) Detect frame pulses and estimate FPS
        events_frame, fps_est = _detect_frame_pulses(raw, stim_ch, mask_bit)
        n_pulses = 0 if (events_frame is None) else int(len(events_frame))
        logger.info(f"[Mario] stim={stim_ch}, pulses={n_pulses}, fps_est={fps_est}")

        # 2) Load behavior and infer expected FPS
        beh_df = pd.read_csv(beh_path, sep='\t')
        fps_meta = _expected_fps_from_metadata(beh_df)
        fps_guess = (fps_meta if (prefer_meta_fps and fps_meta is not None) else (fps_est or 60.0))
        if fps_meta is not None and prefer_meta_fps:
            logger.info(f"[Mario] Using metadata FPS {fps_meta:.3f} over EEG-estimated {fps_est}")
        else:
            logger.info(f"[Mario] Using EEG-estimated FPS {fps_est} (metadata FPS={fps_meta})")

        # If no frame pulses and soft mode allowed, synthesize frame grid
        sf = float(raw.info['sfreq'])
        synth_frames = False
        if (events_frame is None or len(events_frame) == 0):
            if require_frame_pulses:
                raise RuntimeError(
                    f"[Mario] No frame-sync pulses detected on stim_channel='{stim_ch}' with mask_bit={mask_bit}. "
                    f"Verify channel typing and bit mask; alignment requires frame pulses."
                )
            synth_frames = True
            logger.warning(
                f"[Mario] No frame pulses; synthesizing frame grid at {fps_guess:.3f} Hz. "
                f"Alignment will be coarse (anchored to EEG start)."
            )
            step = max(1, int(round(sf / max(fps_guess, 1e-6))))
            frame_samples = np.arange(0, raw.n_times, step, dtype=int)
            frame_times = frame_samples / sf
            events_frame = np.c_[frame_samples, np.zeros_like(frame_samples), np.ones_like(frame_samples)]
        else:
            frame_samples = events_frame[:, 0].astype(int)
            frame_times = frame_samples / sf

        # 3) Behavior onsets + labels
        beh_onsets_s, beh_labels, label_col = _derive_behavior_onsets(beh_df, fps_guess)
        # Precompute segment starts from frame pulses (used for robust offset and summary)
        seg_starts = _segment_starts_from_pulses(events_frame, sf, gap_thr)

        # Optional: use photodiode level anchors to estimate offset/drift
        level_bit_mask = params.get('level_bit_mask', None)
        offset_from_anchors = None
        drift_from_anchors: Tuple[float, float] = (0.0, 0.0)
        anchors_info = {"n": 0, "bit": None}
        if level_bit_mask is not None:
            try:
                level_bit_mask = int(level_bit_mask)
            except Exception:
                level_bit_mask = None
        if level_bit_mask is not None and level_bit_mask > 0:
            ev_level = _detect_bit_events(raw, stim_ch, level_bit_mask)
            lvl_eeg_t = (ev_level[:, 0] / sf) if ev_level.size else np.array([], dtype=float)
            lvl_beh_t = _behavior_level_onsets(beh_df, fps_guess)
            if lvl_eeg_t.size and lvl_beh_t.size:
                pairs_eeg: List[float] = []
                pairs_beh: List[float] = []
                for bt in lvl_beh_t:
                    j = int(np.argmin(np.abs(lvl_eeg_t - bt)))
                    dt = float(lvl_eeg_t[j] - bt)
                    if abs(dt) <= max_dist:
                        pairs_eeg.append(float(lvl_eeg_t[j]))
                        pairs_beh.append(float(bt))
                if pairs_eeg:
                    pairs_eeg_np = np.asarray(pairs_eeg, dtype=float)
                    pairs_beh_np = np.asarray(pairs_beh, dtype=float)
                    diffs = pairs_eeg_np - pairs_beh_np
                    offset_from_anchors = float(np.median(diffs))
                    # Drift: fit delta = a + b * t_aligned, t_aligned = beh + offset
                    t_al = pairs_beh_np + offset_from_anchors
                    delta = pairs_eeg_np - t_al
                    try:
                        b, a = np.polyfit(t_al, delta, 1)
                        drift_from_anchors = (float(a), float(b))
                    except Exception:
                        drift_from_anchors = (0.0, 0.0)
                    anchors_info = {"n": int(len(pairs_eeg)), "bit": int(level_bit_mask)}
                    logger.info(f"[Mario] Using level anchors (bit={level_bit_mask}) for alignment: offset={offset_from_anchors:.3f}s, anchors={anchors_info['n']}")

        # 4) Offset estimation
        if offset_from_anchors is not None:
            offset_s = offset_from_anchors
            logger.info(f"[Mario] Offset from level anchors: {offset_s:.3f}s")
        elif align_first_event:
            # Anchor first behavior onset to first EEG frame (or t=0 if synthesized)
            ev_t0 = (events_frame[0, 0] / sf) if (events_frame is not None and len(events_frame)) else 0.0
            b0 = float(beh_onsets_s[0]) if beh_onsets_s.size else 0.0
            offset_s = ev_t0 - b0
            logger.info(f"[Mario] Using first-event anchor; offset={offset_s:.3f}s (no robust search)")
        else:
            # Robust offset via segment starts
            offset_s, chosen_lab, nmatch, std = _robust_offset(
                seg_starts, beh_labels, beh_onsets_s, ['gym-retro_game', 'fixation_dot'], max_dist
            )
            if offset_s is None:
                # Fallback: anchor first behavior onset to first frame time (0 if synthesized)
                ev_t0 = (events_frame[0, 0] / sf) if (events_frame is not None and len(events_frame)) else 0.0
                b0 = float(beh_onsets_s[0]) if beh_onsets_s.size else 0.0
                offset_s = ev_t0 - b0
                logger.info(f"[Mario] Fallback offset={offset_s:.3f}s (first-anchor)")
            else:
                logger.info(f"[Mario] Robust offset={offset_s:.3f}s using '{chosen_lab}' (matches={nmatch}, std={std:.3f}s)")
                if nmatch < 3:
                    logger.warning(f"[Mario] Low anchor matches for offset (n={nmatch}); alignment confidence reduced.")

        # Optionally force zero-reported offset while still applying the anchor internally
        offset_used = offset_s
        if force_zero_offset:
            logger.info("[Mario] Force-zero offset enabled; reporting 0.000 s")
            offset_s = 0.0

        # 5) Drift estimate using keypresses (or anchors if provided)
        # frame_samples/times already prepared above
        # Try initial drift fit
        drift_a = 0.0
        drift_b = 0.0
        anchor_drift_applied = False
        if offset_from_anchors is not None and (not disable_drift):
            # Use anchor-based drift estimate
            drift_a, drift_b = drift_from_anchors
            rate_ms_min = drift_b * 60000.0
            if abs(rate_ms_min) > drift_thr_ms_min:
                anchor_drift_applied = True
                logger.info(f"[Mario] Anchor-based drift: a={drift_a:.4f}, b={drift_b:.6f} (ms/min={rate_ms_min:.2f}); applying.")
            else:
                drift_a = 0.0; drift_b = 0.0
                logger.info(f"[Mario] Anchor-based drift small (ms/min={rate_ms_min:.2f}); not applying.")

        if disable_drift or anchor_drift_applied:
            kp_t = np.array([])
            kp_al = np.array([])
            kp_samp_pre = np.array([])
            kp_snap_pre = np.array([])
        else:
            kp_mask = (beh_labels == 'keypress') | np.array(['key' in s.lower() or 'press' in s.lower() for s in beh_labels])
            kp_t = beh_onsets_s[kp_mask]
            kp_al = kp_t + (offset_used if 'offset_used' in locals() else offset_s)
            kp_samp_pre, kp_snap_pre = _snap_to_frames(kp_al, frame_times, frame_samples)
        # QC: pre-drift keypress snapping error (in ms)
        try:
            if kp_snap_pre.size and kp_al.size:
                n_err = min(kp_snap_pre.size, kp_al.size)
                err_ms = (kp_snap_pre[:n_err] - kp_al[:n_err]) * 1000.0
                err_abs = np.abs(err_ms)
                p90 = float(np.percentile(err_abs, 90)) if err_abs.size else None
                logger.info(
                    f"[Mario][QC] Pre-drift keypress snap error: median={np.median(err_abs):.2f} ms, "
                    f"p90={p90:.2f} ms, n={n_err}"
                )
        except Exception:
            pass
        drift_a, drift_b, rate_ms_min = _fit_drift(kp_al[:kp_snap_pre.size], kp_snap_pre)
        apply_drift = abs(rate_ms_min) > drift_thr_ms_min
        if disable_drift:
            apply_drift = False
        if apply_drift:
            logger.info(f"[Mario] Drift delta ≈ {drift_a:.4f} + {drift_b:.6f}*t (ms/min={rate_ms_min:.2f}); applying.")
        else:
            drift_a = 0.0; drift_b = 0.0
            logger.info("[Mario] Drift correction disabled by parameter; skipping." if disable_drift else f"[Mario] Drift small (ms/min={rate_ms_min:.2f}); not applying.")

        # QC: post-drift snapping error on keypresses (in ms)
        try:
            if kp_al.size:
                kp_al_post = kp_al + (drift_a + drift_b * kp_al)
                _, kp_snap_post = _snap_to_frames(kp_al_post, frame_times, frame_samples)
                n_err2 = min(kp_snap_post.size, kp_al_post.size)
                if n_err2:
                    err_ms2 = (kp_snap_post[:n_err2] - kp_al_post[:n_err2]) * 1000.0
                    err_abs2 = np.abs(err_ms2)
                    p90_2 = float(np.percentile(err_abs2, 90)) if err_abs2.size else None
                    logger.info(
                        f"[Mario][QC] Post-drift keypress snap error: median={np.median(err_abs2):.2f} ms, "
                        f"p90={p90_2:.2f} ms, n={n_err2}"
                    )
        except Exception:
            pass
        
        # If anchor drift was selected, ensure it is used
        if 'anchor_drift_applied' in locals() and anchor_drift_applied:
            drift_a, drift_b = drift_from_anchors

        # 6) Build behavior events aligned to EEG and snap to frames
        events_behavior, event_id_behavior = _build_behavior_events(
            beh_df, (offset_used if 'offset_used' in locals() else offset_s), drift_a, drift_b, fps_guess, frame_times, frame_samples,
            shift_level_starts_frames=shift_level_frames
        )
        logger.info(f"[Mario] Built {len(events_behavior)} behavior events; ids={list(event_id_behavior.keys())}")

        # 7) Save TSV and plots
        try:
            out_tsv = summary_dir / "aligned_behavior_events.tsv"
            with out_tsv.open('w', newline='', encoding='utf-8') as f:
                import csv
                w = csv.writer(f, delimiter='\t')
                w.writerow(["sample", "code", "label", "aligned_time_s"])
                inv = {v: k for k, v in event_id_behavior.items()}
                for s, _, c in events_behavior:
                    w.writerow([int(s), int(c), inv.get(int(c), str(int(c))), s / raw.info['sfreq']])
            logger.info(f"[Mario] Saved TSV: {out_tsv}")
        except Exception:
            logger.exception("[Mario] Failed to save aligned behavior TSV")

        if do_plot:
            try:
                # Stim overview (first 30s)
                dur = 30.0
                sf = raw.info['sfreq']
                s0, s1 = 0, min(raw.n_times, int(dur * sf))
                t = np.arange(s0, s1) / sf
                if stim_ch in raw.ch_names:
                    y = raw.get_data(picks=[stim_ch], start=s0, stop=s1)[0]
                    plt.figure(figsize=(10, 3))
                    plt.step(t, np.round(y).astype(int), where='post', linewidth=1.0)
                    plt.title(f"Stim {stim_ch} (first {dur:.0f}s)")
                    plt.xlabel("Time (s)"); plt.ylabel("Level")
                    if events_frame is not None and len(events_frame):
                        ev_t = events_frame[:, 0] / sf
                        mask = (ev_t >= 0) & (ev_t <= dur)
                        for tt in ev_t[mask]:
                            plt.axvline(tt, color='red', alpha=0.25, linewidth=0.7)
                    plt.tight_layout(); plt.savefig(summary_dir / f"stim_{stim_ch}_overview.png", dpi=150); plt.close()
                # Overlay first 20s with behaviors
                dur2 = 20.0
                s1b = min(raw.n_times, int(dur2 * sf))
                t2 = np.arange(0, s1b) / sf
                if stim_ch in raw.ch_names:
                    y2 = raw.get_data(picks=[stim_ch], start=0, stop=s1b)[0]
                    plt.figure(figsize=(10, 3))
                    plt.step(t2, np.round(y2).astype(int), where='post', linewidth=1.0)
                    plt.title(f"{stim_ch} with overlays (first {dur2:.0f}s)")
                    plt.xlabel("Time (s)"); plt.ylabel("Level")
                    inv = {v: k for k, v in event_id_behavior.items()}
                    for s, _, c in events_behavior:
                        tt = s / sf
                        if 0 <= tt <= dur2:
                            name = inv.get(int(c), '')
                            if isinstance(name, str) and name.startswith('lvl_'):
                                plt.axvline(tt, color='green', alpha=0.5, linewidth=0.8)
                            elif name == 'gym_retro_game':
                                plt.axvline(tt, color='purple', alpha=0.4, linewidth=0.8)
                            elif name == 'fixation_dot':
                                plt.axvline(tt, color='blue', alpha=0.4, linewidth=0.8)
                            elif name.startswith('kp_') or name == 'beh_keypress':
                                plt.axvline(tt, color='orange', alpha=0.4, linewidth=0.6)
                    plt.tight_layout(); plt.savefig(summary_dir / f"{stim_ch}_with_overlays_first20s.png", dpi=150); plt.close()
            except Exception:
                logger.exception("[Mario] Plotting failed")

        # 8) Store in info['temp'] and return
        raw.info.setdefault('temp', {})
        raw.info['temp']['behavior_events'] = events_behavior
        raw.info['temp']['behavior_event_id'] = event_id_behavior
        raw.info['temp']['mario_alignment'] = {
            'stim_channel': stim_ch,
            'fps_est_eeg': fps_est,
            'fps_expected_meta': fps_meta,
            'offset_s': offset_s,
            'offset_used_s': (offset_used if 'offset_used' in locals() else offset_s),
            'drift_a': drift_a,
            'drift_b': drift_b,
            'level_anchor_bit': (int(params.get('level_bit_mask')) if params.get('level_bit_mask') is not None else None),
            'n_level_anchors': (anchors_info.get('n', 0) if 'anchors_info' in locals() else 0),
            'n_segment_starts': int(len(seg_starts)),
            'synth_frames': bool(synth_frames),
        }
        return raw


__all__ = [
    'MarioEventAlignmentStep',
]
