import unittest
import numpy as np
import mne

from scr.utils.dsi24_bids import (
    rename_dsi_channels,
    set_channel_types,
    find_stim_channel,
    extract_events_and_id,
)


class TestDSI24BIDSUtils(unittest.TestCase):
    def _make_raw(self, ch_names, ch_types, sfreq=100.0, n_samples=200):
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        data = np.zeros((len(ch_names), n_samples), dtype=float)
        return mne.io.RawArray(data, info, verbose="ERROR")

    def test_rename_dsi_channels_legacy_map(self):
        ch_names = ["T3", "T5", "A1", "Fp1"]
        ch_types = ["eeg", "eeg", "eeg", "eeg"]
        raw = self._make_raw(ch_names, ch_types)

        mapping = rename_dsi_channels(raw)

        self.assertIn("T3", mapping)
        self.assertIn("T5", mapping)
        self.assertIn("A1", mapping)
        self.assertEqual(raw.ch_names[0], "T7")
        self.assertEqual(raw.ch_names[1], "P7")
        self.assertEqual(raw.ch_names[2], "M1")
        self.assertEqual(raw.ch_names[3], "Fp1")

    def test_set_channel_types_detects_stim_and_eog(self):
        ch_names = ["Fp1", "STATUS", "HEOG"]
        ch_types = ["eeg", "eeg", "eeg"]
        raw = self._make_raw(ch_names, ch_types)

        mapping = set_channel_types(raw)

        # STATUS should be set to stim, HEOG to eog
        self.assertEqual(mapping.get("STATUS"), "stim")
        self.assertEqual(mapping.get("HEOG"), "eog")
        types = dict(zip(raw.ch_names, raw.get_channel_types()))
        self.assertEqual(types["STATUS"], "stim")
        self.assertEqual(types["HEOG"], "eog")

    def test_find_stim_channel(self):
        ch_names = ["Fp1", "STATUS", "Cz"]
        ch_types = ["eeg", "eeg", "eeg"]
        raw = self._make_raw(ch_names, ch_types)
        set_channel_types(raw)

        stim = find_stim_channel(raw)
        self.assertEqual(stim, "STATUS")

    def test_extract_events_and_id_with_stim(self):
        ch_names = ["Fp1", "STATUS", "Cz"]
        ch_types = ["eeg", "eeg", "eeg"]
        raw = self._make_raw(ch_names, ch_types, sfreq=100.0, n_samples=200)

        # Set channel types appropriately
        set_channel_types(raw)

        # Create digital pulses on the STATUS channel
        status_idx = raw.ch_names.index("STATUS")
        raw._data[status_idx, 10] = 3  # event code 3
        raw._data[status_idx, 50] = 7  # event code 7

        events, event_id = extract_events_and_id(raw, stim_ch="STATUS")
        self.assertIsNotNone(events)
        self.assertGreater(len(events), 0)
        self.assertIsNotNone(event_id)
        self.assertIn("stim_3", event_id)
        self.assertIn("stim_7", event_id)

    def test_extract_events_and_id_with_annotations(self):
        ch_names = ["Fp1", "Cz"]
        ch_types = ["eeg", "eeg"]
        raw = self._make_raw(ch_names, ch_types, sfreq=100.0, n_samples=200)

        # Add a couple of annotations that will be converted to events
        ann = mne.Annotations(onset=[0.10, 0.50], duration=[0.0, 0.0],
                              description=["A", "B"])
        raw.set_annotations(ann)

        events, event_id = extract_events_and_id(raw, stim_ch=None)
        self.assertIsNotNone(events)
        self.assertGreater(len(events), 0)
        self.assertIsNotNone(event_id)


if __name__ == "__main__":
    unittest.main()


