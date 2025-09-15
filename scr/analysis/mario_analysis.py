"""
Mario Task EEG Analysis

Loads the final cleaned epochs produced by the Mario preprocessing pipeline
and generates baseline-corrected ERPs, topomaps, ERP images, and TFRs.
Optionally applies a band-pass filter (default 1–40 Hz) after loading epochs.
Adds DSI-24 motor ROI helpers (C3/C4 Laplacians) and an LRP trace.
Figures are saved into the BIDS-derivatives tree under the same pipeline.

Usage examples
- Provide epochs path directly:
    python -m scr.analysis.mario_analysis --epochs \
      derivatives/mario-preproc/sub-01/ses-001/eeg/sub-01_ses-001_run-01_task-mario_desc-clean_epo.fif

- Or let it infer paths from subject/session/run/task and a derivatives root:
    python -m scr.analysis.mario_analysis --deriv-root derivatives/mario-preproc \
      --subject 01 --session 001 --run 01 --task mario
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import mne
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import csv


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mario EEG analysis: ERP, topomap, ERP-image, TFR")
    p.add_argument("--epochs", type=str, default=None, help="Path to cleaned epochs FIF (_desc-clean_epo.fif)")
    p.add_argument("--deriv-root", type=str, default=str(Path("derivatives") / "mario-preproc"), help="Derivatives root")
    p.add_argument("--subject", default="01", help="Subject (e.g., 01)")
    p.add_argument("--session", default="001", help="Session (e.g., 001)")
    p.add_argument("--run", default="01", help="Run (e.g., 01)")
    p.add_argument("--task", default="mario", help="Task label (default mario)")
    p.add_argument("--bandpass", type=str, default="1.0,40.0", help="Band-pass in Hz as 'l,h' (use 'none' to skip)")
    p.add_argument("--baseline", type=str, default="-0.6,-0.4", help="Baseline window in seconds, e.g., -0.2,0.0")
    p.add_argument("--tmin", type=float, default=-1.2, help="Epoch start for plots")
    p.add_argument("--tmax", type=float, default=0.6, help="Epoch end for plots")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Log level")
    return p


def infer_epochs_path(deriv_root: Path, sub: str, ses: str, run: str, task: str) -> Path:
    eeg_dir = deriv_root / f"sub-{sub}" / f"ses-{ses}" / "eeg"
    # Preferred naming produced by preprocessing
    cand = eeg_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_desc-clean_epo.fif"
    if cand.exists():
        return cand
    # Fallback older ordering
    cand2 = eeg_dir / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-clean_epo.fif"
    if cand2.exists():
        return cand2
    # Fallback to any clean epochs
    for p in eeg_dir.glob("*_desc-clean_epo.fif"):
        return p
    raise FileNotFoundError(f"No cleaned epochs found under {eeg_dir}")


def ensure_fig_dirs(deriv_root: Path, sub: str, ses: str, run: str, task: str) -> Path:
    fig_dir = deriv_root / f"sub-{sub}" / f"ses-{ses}" / "figures" / f"task-{task}" / f"run-{run}" / "analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def pick_first_available(epochs: mne.Epochs, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in epochs.ch_names:
            return name
    return None


def run_analysis(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(levelname)s - %(message)s')
    sub, ses, run, task = args.subject, args.session, args.run, args.task
    deriv_root = Path(args.deriv_root).resolve()

    if args.epochs is None:
        epochs_path = infer_epochs_path(deriv_root, sub, ses, run, task)
    else:
        epochs_path = Path(args.epochs).resolve()
    logging.info(f"Loading epochs: {epochs_path}")
    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose=False)

    # Optional band-pass filtering (e.g., 1–40 Hz)
    bp = None
    try:
        if isinstance(args.bandpass, str) and args.bandpass.strip().lower() != 'none':
            l_s, h_s = [x.strip() for x in args.bandpass.split(',')]
            l_f = float(l_s) if l_s else None
            h_f = float(h_s) if h_s else None
            bp = (l_f, h_f)
            epochs.filter(l_freq=l_f, h_freq=h_f, picks='eeg', verbose=False)
            logging.info(f"Applied band-pass filter: {bp} Hz")
    except Exception as e:
        logging.warning(f"Band-pass parsing/apply failed for '{args.bandpass}': {e}")

    # Baseline correction
    try:
        b0, b1 = [float(x.strip()) for x in args.baseline.split(",")]
        baseline = (b0, b1)
    except Exception:
        baseline = (-0.2, 0.0)
    epochs.apply_baseline(baseline)
    logging.info(f"Applied baseline: {baseline}")

    # Limit time window for plotting if desired
    epochs.crop(tmin=args.tmin, tmax=args.tmax)

    # Prepare figure dir
    fig_dir = ensure_fig_dirs(deriv_root, sub, ses, run, task)

    # Determine condition key: prefer 'kp_all' if present, otherwise first
    cond_keys = list(epochs.event_id.keys())
    if not cond_keys:
        raise RuntimeError("No event_id found in epochs.")
    cond = 'kp_all' if 'kp_all' in cond_keys else cond_keys[0]
    logging.info(f"Using condition for analysis: {cond}")

    # Subset epochs (check event_id, not channel types)
    epochs_c = epochs[cond] if cond in epochs.event_id else epochs

    # 1) Evoked average and save
    evk = epochs_c.average()
    fig_evk = evk.plot(spatial_colors=True, gfp=True, show=False, time_unit='s')
    fig_path = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_cond-{cond}_evoked.png"
    fig_evk.savefig(fig_path, dpi=200)
    plt.close(fig_evk)
    logging.info(f"Saved evoked: {fig_path}")

    # Save evoked fif
    evk_fif = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_cond-{cond}_evoked-ave.fif"
    evk.save(str(evk_fif), overwrite=True)

    # 2) Topomaps at key time points
    topo_times = np.array([-0.1, 0.0, 0.1, 0.2, 0.3])
    try:
        fig_topo = evk.plot_topomap(times=topo_times, ch_type='eeg', time_unit='s', show=False)
        fig_topo_path = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_cond-{cond}_topomaps.png"
        fig_topo.savefig(fig_topo_path, dpi=200)
        plt.close(fig_topo)
        logging.info(f"Saved topomaps: {fig_topo_path}")
    except Exception as e:
        logging.warning(f"Topomap plotting failed: {e}")

    # 3) ERP image for a representative channel (prefer Cz/FCz/CPz/Pz)
    ch = pick_first_available(epochs, ["Cz","FCz","CPz","Pz","Fz","Oz"]) or epochs.ch_names[len(epochs.ch_names)//2]
    try:
        fig_img = mne.viz.plot_epochs_image(epochs_c, picks=[ch], sigma=2.0, vmin=None, vmax=None, show=False)
        if isinstance(fig_img, list):
            fig_img = fig_img[0]
        fig_img_path = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_cond-{cond}_erpimage_{ch}.png"
        fig_img.savefig(fig_img_path, dpi=200)
        plt.close(fig_img)
        logging.info(f"Saved ERP image: {fig_img_path}")
    except Exception as e:
        logging.warning(f"ERP image failed for {ch}: {e}")

    # 3b) DSI-24 ROI overlays and LRP (C3/C4 based)
    roi_figs = {}
    try:
        # Define Mario key groups for thumbs (available subset only)
        right_keys_all = ["kp_B", "kp_X", "kp_Y"]
        left_keys_all  = ["kp_L", "kp_R", "kp_U", "kp_D"]
        ev_keys = set(epochs.event_id.keys())
        right_keys = [k for k in right_keys_all if k in ev_keys]
        left_keys  = [k for k in left_keys_all if k in ev_keys]

        if right_keys and left_keys:
            ev_right = epochs[right_keys].average()
            ev_left  = epochs[left_keys].average()

            # Simple overlays at C3 and C4 if present
            for roi_ch in ["C3", "C4"]:
                if roi_ch in epochs.ch_names:
                    fig_roi, ax = plt.subplots(figsize=(8, 3.2))
                    ax.plot(ev_right.times, ev_right.copy().pick(roi_ch).data[0], label=f"Right ({'+'.join(right_keys)})", color='C1')
                    ax.plot(ev_left.times,  ev_left.copy().pick(roi_ch).data[0],  label=f"Left ({'+'.join(left_keys)})",  color='C0')
                    ax.axvline(0, color='k', lw=0.8, alpha=0.6)
                    ax.set_title(f"{roi_ch}: Left vs Right")
                    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
                    ax.legend(loc='best', fontsize=8)
                    out_roi = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_roi_{roi_ch}_left_vs_right.png"
                    fig_roi.savefig(out_roi, dpi=200); plt.close(fig_roi)
                    roi_figs[f"left_vs_right_{roi_ch}"] = str(out_roi)

            # Surface Laplacians using DSI-24 neighbors (fallback to available subset)
            neighbors = {
                "C3": ["F3", "P3", "T7", "Cz"],
                "C4": ["F4", "P4", "T8", "Cz"],
            }
            def lap_from_evoked(ev: mne.Evoked, center: str, neighs: list[str]) -> Optional[np.ndarray]:
                if center not in epochs.ch_names:
                    return None
                use_nei = [n for n in neighs if n in epochs.ch_names]
                if not use_nei:
                    return None
                c = ev.copy().pick(center).data[0]
                n_stack = [ev.copy().pick(n).data[0] for n in use_nei]
                n_mean = np.mean(np.vstack(n_stack), axis=0)
                return c - n_mean

            for center, neighs in neighbors.items():
                lap_r = lap_from_evoked(ev_right, center, neighs)
                lap_l = lap_from_evoked(ev_left,  center, neighs)
                if lap_r is not None and lap_l is not None:
                    fig_lap, ax = plt.subplots(figsize=(8, 3.2))
                    ax.plot(ev_right.times, lap_r, label=f"Right lap({center})", color='C1')
                    ax.plot(ev_left.times,  lap_l, label=f"Left lap({center})",  color='C0')
                    ax.axvline(0, color='k', lw=0.8, alpha=0.6)
                    ax.set_title(f"Laplacian at {center}: Left vs Right")
                    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
                    ax.legend(loc='best', fontsize=8)
                    out_lap = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_roi_laplacian_{center}_left_vs_right.png"
                    fig_lap.savefig(out_lap, dpi=200); plt.close(fig_lap)
                    roi_figs[f"laplacian_{center}"] = str(out_lap)

            # LRP using standard formula: 0.5*((C3_R - C4_R) + (C4_L - C3_L))
            lrp_path = None
            if ("C3" in epochs.ch_names) and ("C4" in epochs.ch_names):
                c3_r = ev_right.copy().pick("C3").data[0]
                c4_r = ev_right.copy().pick("C4").data[0]
                c3_l = ev_left.copy().pick("C3").data[0]
                c4_l = ev_left.copy().pick("C4").data[0]
                lrp = 0.5 * ((c3_r - c4_r) + (c4_l - c3_l))
                fig_lrp, ax = plt.subplots(figsize=(8, 3.0))
                ax.plot(ev_right.times, lrp, color='C3')
                ax.axvline(0, color='k', lw=0.8, alpha=0.6)
                ax.set_title("LRP (C3/C4)")
                ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
                lrp_path = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_lrp_C3C4.png"
                fig_lrp.savefig(lrp_path, dpi=200); plt.close(fig_lrp)
                roi_figs["lrp_C3C4"] = str(lrp_path)
            else:
                logging.info("Skipping LRP: C3/C4 not found among channels.")
        else:
            logging.info("Skipping ROI/LRP: left/right key groups not both present.")
    except Exception as e:
        logging.warning(f"ROI/LRP section failed: {e}")

    # 3c) Extra ERP images at C3/C4 and Laplacian ERP images
    erp_images_extra = {}
    laplacian_erp_images = {}
    try:
        for ch_extra in ["C3", "C4"]:
            if ch_extra in epochs.ch_names:
                try:
                    fig_img_ex = mne.viz.plot_epochs_image(epochs_c, picks=[ch_extra], sigma=2.0, vmin=None, vmax=None, show=False)
                    if isinstance(fig_img_ex, list):
                        fig_img_ex = fig_img_ex[0]
                    out_img_ex = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_cond-{cond}_erpimage_{ch_extra}.png"
                    fig_img_ex.savefig(out_img_ex, dpi=200)
                    plt.close(fig_img_ex)
                    erp_images_extra[ch_extra] = str(out_img_ex)
                    logging.info(f"Saved ERP image at {ch_extra}: {out_img_ex}")
                except Exception as e:
                    logging.warning(f"ERP image failed for {ch_extra}: {e}")

        # Laplacian ERP images for C3/C4 using DSI-24 neighbors
        neighbors = {
            "C3": ["F3", "P3", "T7", "Cz"],
            "C4": ["F4", "P4", "T8", "Cz"],
        }
        def compute_laplacian_epochs(ep: mne.Epochs, center: str, neighs: list[str]) -> Optional[np.ndarray]:
            if center not in ep.ch_names:
                return None
            use_nei = [n for n in neighs if n in ep.ch_names]
            if not use_nei:
                return None
            cen = ep.copy().pick(center).get_data()[:, 0, :]  # (n_epochs, n_times)
            nei = [ep.copy().pick(n).get_data()[:, 0, :] for n in use_nei]
            n_mean = np.mean(np.stack(nei, axis=0), axis=0)   # (n_epochs, n_times)
            return cen - n_mean

        for center, neighs in neighbors.items():
            try:
                lap_dat = compute_laplacian_epochs(epochs_c, center, neighs)
                if lap_dat is None:
                    continue
                vmax = float(np.nanmax(np.abs(lap_dat))) or 1.0
                fig_lap_img, ax = plt.subplots(figsize=(7.5, 4.2))
                im = ax.imshow(lap_dat, aspect='auto', origin='lower',
                               extent=[epochs_c.times[0], epochs_c.times[-1], 0, lap_dat.shape[0]],
                               cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                ax.axvline(0, color='k', lw=0.8, alpha=0.6)
                ax.set_title(f"ERP image (Laplacian {center})")
                ax.set_xlabel('Time (s)'); ax.set_ylabel('Trial')
                cbar = plt.colorbar(im, ax=ax, pad=0.02)
                cbar.set_label('Amplitude')
                out_lap_img = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_cond-{cond}_erpimage_lap_{center}.png"
                fig_lap_img.savefig(out_lap_img, dpi=200)
                plt.close(fig_lap_img)
                laplacian_erp_images[f"lap_{center}"] = str(out_lap_img)
                logging.info(f"Saved Laplacian ERP image for {center}: {out_lap_img}")
            except Exception as e:
                logging.warning(f"Laplacian ERP image failed for {center}: {e}")
    except Exception as e:
        logging.warning(f"Extra ERP image section failed: {e}")

    # 4) Time-Frequency (Morlet)
    try:
        freqs = np.logspace(np.log10(4), np.log10(40), 20)
        n_cycles = freqs / 2.0
        power = mne.time_frequency.tfr_morlet(
            epochs_c, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, decim=2, n_jobs=1)
        power.apply_baseline(baseline=baseline, mode='logratio')
        fig_tfr = power.plot_topo(baseline=None, mode=None, title=f"TFR {cond}", show=False)
        fig_tfr_path = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_cond-{cond}_tfr_topo.png"
        fig_tfr.savefig(fig_tfr_path, dpi=200)
        plt.close(fig_tfr)
        logging.info(f"Saved TFR topography: {fig_tfr_path}")
    except Exception as e:
        logging.warning(f"TFR computation/plotting failed: {e}")

    # 5) Optional: per-key evokeds if labels like 'kp_*' exist
    try:
        kp_labels = [k for k in epochs.event_id.keys() if str(k).startswith('kp_')]
        if kp_labels:
            for lbl in kp_labels:
                try:
                    ev_lbl = epochs[lbl].average()
                    fig_lbl = ev_lbl.plot(spatial_colors=True, gfp=True, show=False, time_unit='s')
                    fig_lbl_path = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_cond-{lbl}_evoked.png"
                    fig_lbl.savefig(fig_lbl_path, dpi=200)
                    plt.close(fig_lbl)
                    logging.info(f"Saved evoked for {lbl}: {fig_lbl_path}")
                except Exception as e:
                    logging.warning(f"Evoked plotting failed for {lbl}: {e}")
    except Exception:
        logging.debug("Skipping per-key evokeds.")

    # 5b) Pairwise comparisons and difference waves (e.g., kp_L vs kp_R)
    try:
        kp_set = set(kp_labels) if 'kp_labels' in locals() else set()
        # Define plausible pairs
        desired_pairs = [("kp_L", "kp_R"), ("kp_U", "kp_D"), ("kp_X", "kp_Y")]
        pairs = [(a, b) for (a, b) in desired_pairs if a in kp_set and b in kp_set]
        if pairs:
            # Helper to plot compare evokeds at a single channel
            for a, b in pairs:
                try:
                    ev_a = epochs[a].average(); ev_b = epochs[b].average()
                    diff = mne.combine_evoked([ev_a, ev_b], [1.0, -1.0])
                    # Overlay at channel of interest
                    try:
                        from mne.viz import plot_compare_evokeds
                        fig_cmp = plot_compare_evokeds({a: ev_a, b: ev_b, f"{a}-{b}": diff}, picks=ch, show=False, colors={a: 'C0', b: 'C1', f"{a}-{b}": 'C3'})
                        # plot_compare_evokeds returns Axes or list of Axes depending on version/args
                        # Normalize to a Figure for saving
                        if isinstance(fig_cmp, list):
                            ax0 = fig_cmp[0]
                            fig_norm = ax0.figure if hasattr(ax0, 'figure') else plt.gcf()
                        elif hasattr(fig_cmp, 'figure') and not hasattr(fig_cmp, 'savefig'):
                            # Single Axes
                            fig_norm = fig_cmp.figure
                        else:
                            # Already a Figure
                            fig_norm = fig_cmp
                    except Exception:
                        # Fallback: plot evoked and diff separately
                        fig_norm, ax = plt.subplots(figsize=(8, 4))
                        for e, lab, col in [(ev_a, a, 'C0'), (ev_b, b, 'C1'), (diff, f"{a}-{b}", 'C3')]:
                            dat = e.copy().pick(ch).data[0]
                            ax.plot(e.times, dat, label=lab, color=col)
                        ax.axvline(0, color='k', lw=0.8, alpha=0.6)
                        ax.legend(); ax.set_title(f"{a} vs {b} at {ch}"); ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
                    out_cmp = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_cond-{a}_vs_{b}_overlay_{ch}.png"
                    fig_norm.savefig(out_cmp, dpi=200); plt.close(fig_norm)

                    # Difference at channel (save as separate figure)
                    fig_diff, axd = plt.subplots(figsize=(8, 3))
                    axd.plot(diff.times, diff.copy().pick(ch).data[0], color='C3')
                    axd.axvline(0, color='k', lw=0.8, alpha=0.6)
                    axd.set_title(f"Difference {a}-{b} at {ch}")
                    axd.set_xlabel('Time (s)'); axd.set_ylabel('Amplitude')
                    out_diff = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_cond-{a}_minus_{b}_diff_{ch}.png"
                    fig_diff.savefig(out_diff, dpi=200); plt.close(fig_diff)

                except Exception as e:
                    logging.warning(f"Pairwise evoked failed for {a} vs {b}: {e}")

            # Peak table (simple components at ch)
            peak_rows = []
            comp_windows = {
                'N1': (0.08, 0.14, 'neg'),
                'P2': (0.15, 0.25, 'pos'),
                'P3': (0.25, 0.5, 'pos'),
            }
            def find_peak(ev, ch_name, t0, t1, polarity):
                e = ev.copy().pick(ch_name)
                times = e.times
                data = e.data[0]
                m = (times >= t0) & (times <= t1)
                if not np.any(m):
                    return np.nan, np.nan
                idx = np.argmax(data[m]) if polarity == 'pos' else np.argmin(data[m])
                tsel = times[m]
                dsel = data[m]
                return float(tsel[idx]), float(dsel[idx])

            for a, b in pairs:
                try:
                    ev_a = epochs[a].average(); ev_b = epochs[b].average()
                    diff = mne.combine_evoked([ev_a, ev_b], [1.0, -1.0])
                    for comp, (t0, t1, pol) in comp_windows.items():
                        ta, ya = find_peak(ev_a, ch, t0, t1, pol)
                        tb, yb = find_peak(ev_b, ch, t0, t1, pol)
                        td, yd = find_peak(diff, ch, t0, t1, 'pos' if pol=='pos' else 'neg')
                        peak_rows.extend([
                            {'pair': f'{a}_vs_{b}', 'condition': a, 'component': comp, 'channel': ch, 't_peak_s': ta, 'amp': ya},
                            {'pair': f'{a}_vs_{b}', 'condition': b, 'component': comp, 'channel': ch, 't_peak_s': tb, 'amp': yb},
                            {'pair': f'{a}_vs_{b}', 'condition': f'{a}-{b}', 'component': comp, 'channel': ch, 't_peak_s': td, 'amp': yd},
                        ])
                except Exception as e:
                    logging.warning(f"Peak extraction failed for {a} vs {b}: {e}")

            if peak_rows:
                csv_path = fig_dir / f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_erp_peaks_{ch}.csv"
                try:
                    with csv_path.open('w', newline='', encoding='utf-8') as f:
                        w = csv.DictWriter(f, fieldnames=['pair','condition','component','channel','t_peak_s','amp'])
                        w.writeheader(); w.writerows(peak_rows)
                    logging.info(f"Saved ERP peaks table: {csv_path}")
                except Exception as e:
                    logging.warning(f"Failed writing peaks CSV: {e}")
    except Exception as e:
        logging.debug(f"Skipping pairwise comparisons: {e}")

    # 6) Save a small JSON summary
    try:
        summary = {
            "epochs_path": str(epochs_path),
            "n_epochs": int(len(epochs_c)),
            "event_id": epochs.event_id,
            "bandpass": list(bp) if bp is not None else None,
            "baseline": list(baseline),
            "tmin": float(args.tmin),
            "tmax": float(args.tmax),
            "channel_erpimage": ch,
            "figures": {
                "evoked": str(fig_path),
                "topomaps": str(fig_topo_path) if 'fig_topo_path' in locals() else None,
                "erp_image": str(fig_img_path) if 'fig_img_path' in locals() else None,
                "tfr_topo": str(fig_tfr_path) if 'fig_tfr_path' in locals() else None,
                "roi": roi_figs if 'roi_figs' in locals() else {},
                "erp_images_extra": erp_images_extra,
                "laplacian_erp_images": laplacian_erp_images,
            }
        }
        with (fig_dir / "analysis_summary.json").open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    except Exception:
        logging.warning("Failed to write analysis summary JSON")

    logging.info(f"Analysis complete. Figures in: {fig_dir}")
    return 0


def main():
    args = build_argparser().parse_args()
    raise SystemExit(run_analysis(args))


if __name__ == "__main__":
    main()
