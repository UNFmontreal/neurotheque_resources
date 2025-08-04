from .base import BaseStep
from pathlib import Path
import logging
import mne
import glob

class GenerateReport(BaseStep):
    def __init__(self, params):
        super().__init__(params)

    def run(self, data):
        logging.info("Generating HTML report using mne.Report...")

        context = self.params
        subject_id = context.get("subject_id")
        session_id = context.get("session_id")
        task_id = context.get("task_id")
        run_id = context.get("run_id")

        if not all([subject_id, session_id]):
            logging.error("Missing subject or session ID for report generation.")
            return data

        report = mne.Report(title=f"Report for sub-{subject_id}, ses-{session_id}")

        # Add raw data info
        if data is not None:
            report.add_raw(data, title="Raw Data", psd=True)

        # Add metrics
        metrics = context.get("metrics", {})
        if metrics:
            metrics_html = "<h2>Metrics</h2><ul>"
            for key, value in metrics.items():
                metrics_html += f"<li><b>{key}:</b> {value}</li>"
            metrics_html += "</ul>"
            report.add_html(metrics_html, title="Metrics", section="Processing Summary")

        # Add plots
        plots = self.find_plots(subject_id, session_id)
        for plot_path in plots:
            try:
                title = Path(plot_path).stem.replace('_', ' ').title()
                report.add_image(image=plot_path, title=title, section="Figures")
            except Exception as e:
                logging.warning(f"Could not add plot {plot_path} to report: {e}")


        report_path = self.params["paths"].get_report_path(
            "summary", subject_id, session_id, task_id, run_id, name="mne_report.html"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report.save(report_path, overwrite=True, open_browser=False)
            
        logging.info(f"Report generated at: {report_path}")
        return data

    def find_plots(self, subject_id, session_id):
        """Find all .png plot files for the given subject and session."""
        
        reports_dir = self.params["paths"].reports_dir
        
        # Search for plots in the subject's report directory
        search_pattern = f"{reports_dir}/**/sub-{subject_id}/**/*ses-{session_id}/**/*.png"
        
        plot_files = glob.glob(search_pattern, recursive=True)
        
        logging.info(f"Found {len(plot_files)} plots for the report.")
        
        return plot_files
