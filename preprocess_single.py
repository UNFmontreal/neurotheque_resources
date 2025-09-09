#!/usr/bin/env python3
"""
Preprocess a single BIDS run (JSON-driven). Saves intermediates, metrics, and an MNE HTML report.

Major steps (all optional / config-driven):
- set montage/reference
- resample
- band-pass and notch
- ICA (with optional ECG/EOG aids) and component exclusion
- optional autoreject
- epoching via event mapping from JSON (same schema as BIDSify)
- compute metrics (PSDs, rejection summary)
- save cleaned raw, epochs/evokeds, figures
- write MNE Report to derivatives/reports

Python 3.10+
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mne
import numpy as np
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne_bids import BIDSPath, read_raw_bids

from event_mapping import map_events_from_config


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _bids_derivative_paths(
    bids_root: Path, derivatives_root: Path, project_version: str,
    bp: BIDSPath, task: str
) -> Dict[str, Path]:
    base = derivatives_root / project_version / bp.subject
    if bp.session:
        base = base / bp.session
    eeg = base / "eeg"
    eeg.mkdir(parents=True, exist_ok=True)

    stem = f"sub-{bp.subject}"
    if bp.session:
        stem += f"_ses-{bp.session}"
    stem += f"_task-{task}"
    if bp.run:
        stem += f"_run-{bp.run}"

    out = {
        "raw_clean": eeg / f"{stem}_desc-preproc_raw.fif",
        "ica_fif": eeg / f"{stem}_desc-ica.fif",
        "epochs": eeg / f"{stem}_desc-epo.fif",
        "evoked": eeg / f"{stem}_desc-ave.fif",
        "report": derivatives_root / "reports" / f"{stem}_desc-preproc_report.html",
        "log": derivatives_root / "logs" / f"{stem}_desc-preproc_log.json",
    }
    return out


def _load_cfg(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_bids_raw(bids_root: Path, sub: str, ses: Optional[str], task: str, run: Optional[str]) -> Tuple[mne.io.BaseRaw, BIDSPath]:
    bp = BIDSPath(subject=sub, session=ses, task=task, run=run, datatype="eeg", root=bids_root)
    raw = read_raw_bids(bp, verbose=False)
    raw.load_data()
    return raw, bp


def _maybe_resample(raw: mne.io.BaseRaw, cfg: Dict[str, object]) -> None:
    sfreq = cfg.get("resample_sfreq")
    if sfreq:
        raw.resample(float(sfreq))


def _maybe_filter(raw: mne.io.BaseRaw, cfg: Dict[str, object]) -> None:
    fcfg = cfg.get("filter")
    if fcfg:
        raw.filter(l_freq=fcfg.get("l_freq"), h_freq=fcfg.get("h_freq"))


def _maybe_notch(raw: mne.io.BaseRaw, cfg: Dict[str, object], power_line_freq: Optional[int]) -> None:
    ncfg = cfg.get("notch")
    if ncfg:
        freqs = ncfg.get("freqs")
        if not freqs and power_line_freq:
            freqs = [power_line_freq, 2 * power_line_freq]
        if freqs:
            raw.notch_filter(freqs=freqs)


def _maybe_reref(raw: mne.io.BaseRaw, cfg: Dict[str, object]) -> None:
    reref = cfg.get("reref", "average")
    if reref == "average":
        raw.set_eeg_reference("average", projection=False)
    elif isinstance(reref, str) and "," in reref:
        l, r = reref.split(",", 1)
        raw.set_eeg_reference(ref_channels=[l.strip(), r.strip()], projection=False)
    elif reref == "linked_mastoids":
        raw.set_eeg_reference(ref_channels=["M1", "M2"], projection=False)


def _maybe_ica(raw: mne.io.BaseRaw, cfg: Dict[str, object], report: mne.Report):
    icfg = cfg.get("ica")
    if not icfg:
        return None
    method = str(icfg.get("method", "fastica"))
    n_comp = icfg.get("n_components", "auto")
    if n_comp == "auto":
        n_components = 0.99  # variance explained heuristic
    else:
        n_components = int(n_comp)

    ica = ICA(method=method, n_components=n_components, random_state=97)
    ica.fit(raw)

    # help find artifact components
    if icfg.get("eog", False):
        try:
            eog_epochs = create_eog_epochs(raw, reject_by_annotation=True)
            eog_inds, _ = ica.find_bads_eog(eog_epochs)
            ica.exclude.extend(eog_inds)
        except Exception:
            pass
    if icfg.get("ecg", False):
        try:
            ecg_epochs = create_ecg_epochs(raw, reject_by_annotation=True)
            ecg_inds, _ = ica.find_bads_ecg(ecg_epochs)
            ica.exclude.extend(ecg_inds)
        except Exception:
            pass

    # plot to report
    try:
        fig = ica.plot_components(show=False)
        report.add_figure(fig, title="ICA components", section="ica")
        fig = ica.plot_sources(raw, show=False)
        report.add_figure(fig, title="ICA sources", section="ica")
    except Exception:
        pass

    raw_clean = ica.apply(raw.copy())
    report.add_html("<p>ICA excluded components: "
                    + ", ".join(map(str, sorted(set(ica.exclude)))) + "</p>", title="ICA excluded", section="ica")
    return ica, raw_clean


def _maybe_autoreject(epochs: mne.Epochs, cfg: Dict[str, object], report: mne.Report) -> mne.Epochs:
    ar_cfg = cfg.get("autoreject") or {}
    if not ar_cfg.get("enabled", False):
        return epochs
    from autoreject import AutoReject
    ar = AutoReject(**({"method": ar_cfg.get("method", "local")}))
    cleaned = ar.fit_transform(epochs)
    try:
        fig = ar.get_reject_log(epochs).plot()
        report.add_figure(fig, title="AutoReject log", section="epochs")
    except Exception:
        pass
    return cleaned


def _compute_psd(raw: mne.io.BaseRaw) -> Optional[object]:
    try:
        fig = raw.plot_psd(show=False)
        return fig
    except Exception:
        return None


def preprocess_one(
    cfg: Dict[str, object],
    bids_root: Path,
    sub: str, ses: Optional[str], task: str, run: Optional[str],
) -> Dict[str, object]:
    t0 = time.time()
    prj = cfg.get("project", {"version": "0.1.0"})
    preprocessing = cfg.get("preprocessing", {})
    report_cfg = cfg.get("report", {"enabled": True, "title": "Neurothèque Preprocessing Report"})

    derivatives_root = Path(cfg["paths"].get("derivatives_root") or (bids_root / "derivatives" / "neurotheque-preproc"))
    out = {}

    raw, bp = _read_bids_raw(bids_root, sub, ses, task, run)
    paths = _bids_derivative_paths(bids_root, derivatives_root, prj.get("version", "0.1.0"), bp, task)

    # Prepare report
    report = mne.Report(title=report_cfg.get("title", "Neurothèque Preprocessing Report"))
    try:
        report.add_raw(raw=raw, title="Raw", psd=False)
    except Exception:
        pass

    # Montage/reference
    try:
        # montage application already in BIDSify; skip here unless specified
        pass
    except Exception:
        pass
    _maybe_reref(raw, preprocessing)

    # Optional resample/filter/notch
    _maybe_resample(raw, preprocessing)
    _maybe_filter(raw, preprocessing)
    _maybe_notch(raw, preprocessing, power_line_freq=cfg.get("bids", {}).get("line_freq"))

    # PSD before cleaning
    if (cfg.get("preprocessing", {}).get("metrics", {}).get("psd", True)):
        fig = _compute_psd(raw)
        if fig is not None:
            report.add_figure(fig, title="PSD (pre-clean)", section="psd")

    # ICA
    ica_result = _maybe_ica(raw, preprocessing, report)
    if ica_result:
        ica, raw_clean = ica_result
    else:
        ica = None
        raw_clean = raw

    # Save cleaned raw if requested
    if preprocessing.get("save_intermediates", True):
        _ensure_dir(paths["raw_clean"])
        raw_clean.save(paths["raw_clean"], overwrite=True)

    # Events and epoching
    evs_global = cfg["events"]
    task_cfg = evs_global.get("tasks", {}).get(task, evs_global.get("tasks", {}).get(evs_global.get("default_task")))
    ev = mne.find_events(raw_clean, stim_channel=evs_global.get("trigger_channel", "Trigger"),
                         shortest_event=1, consecutive=False, uint_cast=True, verbose=False)
    ev_mapped, event_id = map_events_from_config(raw_clean, ev, task_cfg or {})

    epochs = None
    if task_cfg and task_cfg.get("epoching"):
        ep = task_cfg["epoching"]
        if ev_mapped.size and event_id:
            epochs = mne.Epochs(
                raw_clean, ev_mapped, event_id=event_id,
                tmin=float(ep.get("tmin", -0.2)), tmax=float(ep.get("tmax", 0.8)),
                baseline=tuple(ep.get("baseline", (-0.2, 0.0))),
                preload=True, reject_by_annotation=bool(ep.get("reject_by_annotation", True)),
            )
            # optional autoreject
            if cfg.get("preprocessing", {}).get("autoreject", {}).get("enabled", False):
                epochs = _maybe_autoreject(epochs, cfg.get("preprocessing", {}), report)

            # metrics
            try:
                report.add_epochs(epochs=epochs, title="Epochs", psd=False)
            except Exception:
                pass

            # evoked
            try:
                evoked = {k: epochs[k].average() for k in event_id}
                figs = [evoked[k].plot_joint(show=False) for k in evoked]
                for k, fig in zip(evoked, figs):
                    report.add_figure(fig, title=f"Evoked: {k}", section="evoked")
            except Exception:
                pass

    # Report and logs
    _ensure_dir(paths["report"])
    try:
        report.save(str(paths["report"]), overwrite=True, open_browser=False)
    except Exception:
        pass

    log = {
        "subject": bp.subject, "session": bp.session, "task": task, "run": bp.run,
        "paths": {k: str(v) for k, v in paths.items()},
        "events_mapped": int(ev_mapped.shape[0]) if 'ev_mapped' in locals() and ev_mapped is not None and ev_mapped.size else 0,
        "duration_sec": round(time.time() - t0, 2),
        "version": prj.get("version", "0.1.0"),
    }
    _ensure_dir(paths["log"])
    _save_json(log, paths["log"])
    return log


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Preprocess a single BIDS run (JSON-driven).")
    ap.add_argument("--config", required=True, help="JSON config (see config_schema.json)")
    ap.add_argument("--bids-root", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--ses")
    ap.add_argument("--task", required=True)
    ap.add_argument("--run")
    args = ap.parse_args(argv)

    cfg = _load_cfg(Path(args.config))
    bids_root = Path(args.bids_root).resolve()

    preprocess_one(cfg, bids_root, args.sub, args.ses, args.task, args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

