"""
Neurotheque QC report step.

Generates an HTML report and stores derived QC metrics for the current subject/session/run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from html import escape

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.report import Report

from .base import BaseStep


class NeurothequeQCReportStep(BaseStep):
    """Create a lightweight QC report that aggregates key artefact metrics and figures."""

    def run(self, data):
        if data is None:
            raise ValueError("[NeurothequeQCReportStep] No data provided.")

        subject_id = self.params.get("subject_id")
        session_id = self.params.get("session_id")
        task_id = self.params.get("task_id")
        run_id = self.params.get("run_id")
        paths = self.params.get("paths")

        if paths is None:
            raise ValueError("[NeurothequeQCReportStep] 'paths' parameter is required.")
        if subject_id is None or session_id is None:
            raise ValueError("[NeurothequeQCReportStep] Subject and session identifiers are required.")

        report_type = self.params.get("report_type", "neurotheque-qc")
        report_filename = self.params.get("report_filename", "neurotheque_qc_report.html")
        report_dir = Path(paths.get_report_path(report_type, subject_id, session_id, task_id, run_id))
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = paths.get_report_path(
            report_type,
            subject_id,
            session_id,
            task_id,
            run_id,
            name=report_filename,
        )
        metrics_path = report_dir / "qc_metrics.json"

        logging.info(
            "[NeurothequeQCReportStep] Saving QC report to %s",
            report_path,
        )

        report = Report(title=f"Neurotheque QC - sub-{subject_id} ses-{session_id}")

        temp_info = data.info.get("temp", {}) if hasattr(data.info, "get") else {}
        autoreject_results = temp_info.get("autoreject_results", {})
        signal_metrics = temp_info.get("signal_metrics", {})
        autoreject_info = temp_info.get("autoreject")
        ica_obj = temp_info.get("ica")
        ica_labels = temp_info.get("ica_labeled", {})

        channel_types = dict(zip(data.ch_names, data.get_channel_types()))
        eeg_channels = [name for name, kind in channel_types.items() if kind == "eeg"]

        # --- Helper utilities -------------------------------------------------
        def pick_autoreject_pass(candidates: tuple[str, ...]) -> tuple[str | None, dict | None]:
            for candidate in candidates:
                if candidate in autoreject_results:
                    return candidate, autoreject_results[candidate]
            return None, None

        def render_dict_table(title: str, rows: dict) -> str:
            cells = "".join(f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>" for k, v in rows.items())
            return (
                f"<h3>{escape(title)}</h3>"
                "<table style='border-collapse:collapse;'>"
                f"<tbody>{cells}</tbody></table>"
            )

        def add_images_from(directory: Path, title_prefix: str, limit: int = 6, section: str | None = None):
            if not directory.exists():
                return
            images = sorted(directory.glob("*.png"))
            for idx, image_path in enumerate(images[:limit]):
                try:
                    report.add_image(image_path, title=f"{title_prefix} - {image_path.stem}", section=section)
                except Exception as exc:
                    logging.warning("[NeurothequeQCReportStep] Failed to add %s: %s", image_path, exc)
            if len(images) > limit:
                report.add_html(
                    f"<p>Showing {limit} of {len(images)} figures saved to {escape(str(directory))}</p>",
                    title=f"{title_prefix} (truncated)",
                    section=section,
                )

        def canon_label(label: str) -> str:
            s = str(label).strip().lower().replace("-", " ").replace("_", " ")
            if "blink" in s or "eye" in s:
                return "eye"
            if "muscle" in s:
                return "muscle"
            if "heart" in s or "ecg" in s or "cardiac" in s:
                return "heart"
            if "line noise" in s or ("line" in s and "noise" in s):
                return "line_noise"
            if "channel noise" in s or ("channel" in s and "noise" in s):
                return "channel_noise"
            if "brain" in s:
                return "brain"
            if "other" in s:
                return "other"
            return s.replace(" ", "_")

        # --- Determine pass ordering -----------------------------------------
        pass1_key, pass1 = pick_autoreject_pass(("pass1", "initial", "fit", "pre_ica"))
        pass2_key, pass2 = pick_autoreject_pass(("pass2", "final", "fit_transform", "post_ica"))

        if not pass1 and autoreject_results:
            sorted_keys = sorted(autoreject_results.keys())
            pass1_key, pass1 = sorted_keys[0], autoreject_results[sorted_keys[0]]
        if not pass2 and autoreject_results:
            sorted_keys = sorted(autoreject_results.keys())
            pass2_key, pass2 = sorted_keys[-1], autoreject_results[sorted_keys[-1]]

        # --- Raw overview -----------------------------------------------------
        try:
            report.add_raw(data, title="Raw Overview", psd=True, section="01 Raw Overview")
        except Exception as exc:
            logging.warning("[NeurothequeQCReportStep] Could not add raw overview: %s", exc)

        # --- Filtering summary ------------------------------------------------
        try:
            eeg = data.copy().pick_types(eeg=True, exclude="bads")
            if len(eeg.ch_names) > 0:
                fmax = float(min(80.0, eeg.info["sfreq"] / 2.0))
                spectrum = eeg.compute_psd(method="welch", fmin=0.5, fmax=fmax)
                fig_psd = spectrum.plot(average=True, show=False)
                report.add_figure(fig_psd, title="Post-filter EEG PSD", section="02 Filtering")
                plt.close(fig_psd)

                eeg_data = eeg.get_data()
                if isinstance(eeg, mne.io.BaseRaw):
                    data_uV = eeg_data * 1e6
                elif isinstance(eeg, mne.BaseEpochs):
                    data_uV = eeg_data.reshape(-1, eeg_data.shape[-1]) * 1e6
                else:
                    data_uV = np.asarray(eeg_data).reshape(-1, eeg_data.shape[-1]) * 1e6
                ptp = np.ptp(data_uV, axis=-1)
                fig_amp, ax_amp = plt.subplots(figsize=(6, 4))
                ax_amp.hist(ptp, bins=40, color="steelblue", alpha=0.8)
                ax_amp.set_title("Peak-to-peak distribution (uV)")
                ax_amp.set_xlabel("uV")
                ax_amp.set_ylabel("Count")
                ax_amp.grid(alpha=0.3)
                report.add_figure(fig_amp, title="Amplitude Distribution", section="02 Filtering")
                plt.close(fig_amp)
        except Exception as exc:
            logging.warning("[NeurothequeQCReportStep] Filtering summary unavailable: %s", exc)

        # --- AutoReject summaries --------------------------------------------
        def build_ar_summary(label: str, key: str, result: dict | None):
            if not result:
                return
            params = result.get("ar_params", {})
            summary = {
                "Epochs (total)": result.get("n_epochs_total"),
                "Epochs marked bad": result.get("n_epochs_bad"),
                "Mode": result.get("mode"),
                "Epoch duration (s)": result.get("epoch_duration"),
                "Epoch overlap (s)": result.get("epoch_overlap"),
                "n_interpolate": params.get("n_interpolate"),
                "consensus": params.get("consensus"),
                "cv": params.get("cv"),
                "thresh_method": params.get("thresh_method"),
            }
            report.add_html(
                render_dict_table(f"AutoReject {label} summary", summary),
                title=f"AutoReject {label} Summary",
                section=f"02 AutoReject/{label}",
            )
            plot_root = Path(
                paths.get_report_path("autoreject", subject_id, session_id, task_id, run_id)
            ) / key
            add_images_from(plot_root, f"AutoReject {label}", section=f"02 AutoReject/{label}")

        if pass1_key and pass1:
            build_ar_summary("Pass 1", pass1_key, pass1)
        if pass2_key and pass2 and pass2 is not pass1:
            build_ar_summary("Pass 2", pass2_key, pass2)

        # --- ICA overview -----------------------------------------------------
        if ica_obj is not None:
            try:
                report.add_ica(ica=ica_obj, inst=data, title="ICA Overview", section="03 ICA/Overview")
            except Exception as exc:
                logging.warning("[NeurothequeQCReportStep] Could not embed ICA overview: %s", exc)

        if ica_labels:
            suggested = ica_labels.get("suggested_exclude", [])
            labeled = ica_labels.get("labeled_components", {})
            label_rows = {
                "Suggested exclusions": ", ".join(str(idx) for idx in suggested) or "None",
                "Label sources": ", ".join(sorted(labeled.keys())) or "N/A",
            }
            report.add_html(
                render_dict_table("ICLabel Summary", label_rows),
                title="ICLabel Summary",
                section="03 ICA/ICLabel",
            )

            ic_info = labeled.get("iclabel", {}) if isinstance(labeled, dict) else {}
            labels = ic_info.get("labels", [])
            probs = None
            for key in ("y_pred_proba", "proba", "probabilities", "y_pred_probas"):
                if key in ic_info:
                    probs = np.asarray(ic_info[key])
                    break
            classes = None
            for key in ("labels_set", "classes", "class_names"):
                if key in ic_info:
                    classes = ic_info[key]
                    break

            rows = []
            for idx, raw_label in enumerate(labels):
                top_label = raw_label
                top_prob = None
                if probs is not None:
                    try:
                        arr = np.asarray(probs[idx])
                        best_idx = int(arr.argmax())
                        top_prob = float(arr[best_idx])
                        if classes is not None and best_idx < len(classes):
                            top_label = classes[best_idx]
                    except Exception:
                        top_prob = None
                canonical = canon_label(top_label)
                excluded = idx in suggested
                styles = []
                if canonical == "eye":
                    styles.append("background-color:#ffe5e5")
                if excluded:
                    styles.append("border:2px solid red")
                prob_text = f"{top_prob:.2%}" if top_prob is not None else "n/a"
                rows.append(
                    "<tr style='{style}'>"
                    "<td>{idx}</td>"
                    "<td>{label}</td>"
                    "<td>{prob}</td>"
                    "<td>{canonical}</td>"
                    "<td>{excluded}</td>"
                    "</tr>".format(
                        style=";".join(styles),
                        idx=idx,
                        label=escape(str(top_label)),
                        prob=escape(prob_text),
                        canonical=escape(canonical),
                        excluded="yes" if excluded else "no",
                    )
                )

            if rows:
                table_html = (
                    "<h4>ICLabel Component Detail</h4>"
                    "<table style='border-collapse:collapse;'>"
                    "<thead>"
                    "<tr><th>Component</th><th>Top label</th><th>Probability</th><th>Canonical</th><th>Excluded</th></tr>"
                    "</thead><tbody>"
                    f"{''.join(rows)}"
                    "</tbody></table>"
                )
                report.add_html(table_html, title="ICLabel Components", section="03 ICA/ICLabel")

            ica_dir = Path(paths.get_ica_report_dir(subject_id, session_id, task_id, run_id))
            add_images_from(ica_dir, "ICA Figures", section="03 ICA/Figures")
            add_images_from(ica_dir / "before_after", "ICA Before/After", section="03 ICA/Before-After")
            add_images_from(ica_dir / "labeled", "ICA Labeled", section="03 ICA/Labeled Components")

        # --- Metrics & signal quality ----------------------------------------
        metrics = {
            "subject": subject_id,
            "session": session_id,
            "task": task_id,
            "run": run_id,
            "sfreq": float(data.info.get("sfreq", 0.0)),
            "n_channels_total": len(data.ch_names),
            "n_channels_eeg": len(eeg_channels),
            "duration_s": float(data.n_times / data.info.get("sfreq", 1.0)),
        }
        hp = data.info.get("highpass")
        lp = data.info.get("lowpass")
        if hp is not None:
            metrics["highpass_hz"] = float(hp)
        if lp is not None:
            metrics["lowpass_hz"] = float(lp)
        if ica_obj is not None and hasattr(ica_obj, "n_components_"):
            metrics["ica_components"] = int(getattr(ica_obj, "n_components_", 0))
            metrics["ica_components_excluded"] = int(len(getattr(ica_obj, "exclude", []) or []))
        if pass1:
            metrics["pass1_bad_epochs"] = int(pass1.get("n_epochs_bad", 0))
            total = max(1, int(pass1.get("n_epochs_total", 1)))
            metrics["pass1_bad_fraction"] = metrics["pass1_bad_epochs"] / total
        if pass2 and pass2 is not pass1:
            metrics["pass2_bad_epochs"] = int(pass2.get("n_epochs_bad", 0))
            total = max(1, int(pass2.get("n_epochs_total", 1)))
            metrics["pass2_bad_fraction"] = metrics["pass2_bad_epochs"] / total

        report.add_html(
            render_dict_table("Global Metrics", metrics),
            title="Global Metrics",
            section="04 Metrics/Overview",
        )

        if signal_metrics:
            ordered = []
            if pass1_key and pass1_key in signal_metrics:
                ordered.append(("AutoReject Pass 1", signal_metrics[pass1_key]))
            if "ica_label" in signal_metrics:
                ordered.append(("Post ICLabel", signal_metrics["ica_label"]))
            if pass2_key and pass2_key in signal_metrics:
                ordered.append(("AutoReject Pass 2", signal_metrics[pass2_key]))
            if not ordered:
                ordered = [(label, metrics) for label, metrics in signal_metrics.items()]

            rows = []
            for label, stage_metrics in ordered:
                cells = "".join(
                    f"<tr><th>{escape(str(metric))}</th><td>{escape(f'{value:.3f}' if isinstance(value, float) else str(value))}</td></tr>"
                    for metric, value in stage_metrics.items()
                )
                rows.append(
                    f"<h4>{escape(label)}</h4>"
                    "<table style='border-collapse:collapse;'>"
                    f"<tbody>{cells}</tbody></table>"
                )
            report.add_html(
                "<h3>Signal Quality Metrics</h3>" + "".join(rows),
                title="Signal Quality Metrics",
                section="04 Metrics/Signal Quality",
            )

        # --- Quality checklist ------------------------------------------------
        quality_checks: list[tuple[str, bool | None, str]] = []

        if pass1 and pass2 and pass2 is not pass1:
            before = pass1.get("n_epochs_bad")
            after = pass2.get("n_epochs_bad")
            if before is not None and after is not None:
                detail = f"{before}  {after}"
                quality_checks.append(("AutoReject pass 2 reduces bad epochs", after <= before, detail))

        def channels_subset_ok(ar_result) -> bool | None:
            names = ar_result.get("ch_names") if ar_result else None
            if not names:
                return None
            return set(names).issubset(set(eeg_channels))

        if pass1:
            subset = channels_subset_ok(pass1)
            if subset is not None:
                detail = f"{len(pass1.get('ch_names', []))} EEG channels evaluated"
                quality_checks.append(("AutoReject pass 1 EEG-only", subset, detail))
        if pass2 and pass2 is not pass1:
            subset = channels_subset_ok(pass2)
            if subset is not None:
                detail = f"{len(pass2.get('ch_names', []))} EEG channels evaluated"
                quality_checks.append(("AutoReject pass 2 EEG-only", subset, detail))

        first_samp = getattr(data, "first_samp", None)
        last_samp = getattr(data, "last_samp", None)
        n_times = getattr(data, "n_times", None)
        if isinstance(first_samp, int) and isinstance(last_samp, int) and isinstance(n_times, int):
            expected_last = first_samp + n_times - 1
            quality_checks.append(("Recording starts at frame 0", first_samp == 0, f"first_samp={first_samp}"))
            quality_checks.append(
                ("Recording ends at expected frame", last_samp == expected_last, f"last_samp={last_samp}")
            )

        filter_ok = (
            hp is not None
            and lp is not None
            and abs(hp - 1.0) <= 0.2
            and abs(lp - 40.0) <= 1.0
        )
        quality_checks.append(("Band-pass within 1-40 Hz", filter_ok, f"{hp}-{lp} Hz"))

        if ica_obj is not None and hasattr(ica_obj, "exclude"):
            excluded = getattr(ica_obj, "exclude") or []
            quality_checks.append(("ICA reviewed/cleaned", len(excluded) > 0, f"{len(excluded)} components excluded"))

        if pass2_key and pass2_key in signal_metrics and pass1_key in signal_metrics:
            ptp_before = signal_metrics[pass1_key].get("peak_to_peak_uV")
            ptp_after = signal_metrics[pass2_key].get("peak_to_peak_uV")
            if ptp_before and ptp_after:
                quality_checks.append(
                    (
                        "Peak-to-peak reduction after ICA",
                        ptp_after <= ptp_before,
                        f"{ptp_before:.2f} -> {ptp_after:.2f} uV",
                    )
                )

        if quality_checks:
            rows = []
            for label, outcome, detail in quality_checks:
                if outcome is None:
                    symbol = "&mdash;"
                    status_text = "N/A"
                else:
                    symbol = "&#10003;" if outcome else "&#10007;"
                    status_text = "PASS" if outcome else "FAIL"
                rows.append(
                    "<tr>"
                    f"<td>{escape(label)}</td>"
                    f"<td style='text-align:center'>{symbol}<br><small>{status_text}</small></td>"
                    f"<td>{escape(detail)}</td>"
                    "</tr>"
                )
            checklist_html = (
                "<h3>Quality Checklist</h3>"
                "<table style='border-collapse:collapse;'>"
                "<thead>"
                "<tr><th>Check</th><th>Status</th><th>Details</th></tr>"
                "</thead><tbody>"
                f"{''.join(rows)}"
                "</tbody></table>"
            )
            report.add_html(checklist_html, title="Quality Checklist", section="05 Quality Checklist")

        try:
            report.save(report_path, overwrite=True, open_browser=False)
        except Exception as exc:
            logging.error("[NeurothequeQCReportStep] Failed to save HTML report: %s", exc)
        else:
            logging.info("[NeurothequeQCReportStep] Report written to %s", report_path)

        try:
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        except Exception as exc:
            logging.warning("[NeurothequeQCReportStep] Could not write metrics JSON: %s", exc)

        return data
