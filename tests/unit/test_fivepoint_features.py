import numpy as np
import mne

from scr.analysis.fivepoint_analysis import (
    extract_5pt_events_by_codes,
    compute_mrcp_lrp,
)


def _make_synthetic_raw_with_events(sfreq: float = 250.0, duration_s: float = 5.0):
    rng = np.random.RandomState(42)
    n_samples = int(sfreq * duration_s)
    ch_names = ['O1', 'O2', 'Cz', 'C3', 'C4', 'Trigger']
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'stim']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    t = np.arange(n_samples) / sfreq
    # Alpha (10 Hz) on O1/O2 mostly during baseline
    alpha = 10.0
    o1 = 2e-6 * np.sin(2 * np.pi * alpha * t)
    o2 = 1.5e-6 * np.sin(2 * np.pi * alpha * t + 0.5)

    # MRCP-like negative ramp on Cz before response
    cz = np.zeros_like(t)
    # LRP: C3 negative, C4 positive pre-response
    c3 = np.zeros_like(t)
    c4 = np.zeros_like(t)

    trig = np.zeros_like(t)

    # Place one onset at 1.0 s and response at 2.0 s
    onset_samp = int(1.0 * sfreq)
    resp_samp = int(2.0 * sfreq)
    onset_code, resp_code = 10, 20
    trig[onset_samp] = onset_code
    trig[resp_samp] = resp_code

    # Create a Cz negative ramp from -1.5 s to response
    ramp_start = int((2.0 - 1.5) * sfreq)
    ramp_idx = np.arange(ramp_start, resp_samp)
    if ramp_idx.size:
        cz[ramp_idx] = -3e-6 * np.linspace(0.0, 1.0, ramp_idx.size)

    # LRP difference near -100 ms
    lrp_start = int((2.0 - 0.5) * sfreq)
    lrp_idx = np.arange(lrp_start, resp_samp)
    if lrp_idx.size:
        c3[lrp_idx] = -1.0e-6
        c4[lrp_idx] = +1.0e-6

    data = np.vstack([o1, o2, cz, c3, c4, trig])
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw, onset_code, resp_code


def test_extract_5pt_events_by_codes_pairs_correctly():
    raw, onset_code, resp_code = _make_synthetic_raw_with_events()
    on, rp, rts = extract_5pt_events_by_codes(
        raw, stim_channel='Trigger', onset_codes=[onset_code], resp_codes=[resp_code]
    )
    assert on.shape[0] == 1 and rp.shape[0] == 1
    # Event codes are recoded to 801/802
    assert on[0, 2] == 801 and rp[0, 2] == 802
    # RT should be ~1.0 s
    assert np.isclose(rts[0], 1.0, atol=1 / raw.info['sfreq'])


def test_compute_prestim_alpha_returns_vector_and_positive_values():
    from scr.analysis.fivepoint_analysis import compute_prestim_alpha

    raw, onset_code, resp_code = _make_synthetic_raw_with_events()
    # Build events for stimulus epochs
    events = mne.find_events(raw, stim_channel='Trigger', shortest_event=1, verbose=False)
    on_events = events[events[:, 2] == onset_code]
    # Epoch around onset to include -0.6..0.1 s
    e_stim = mne.Epochs(raw, on_events, event_id={'onset': onset_code}, tmin=-0.6, tmax=0.1,
                        baseline=None, preload=True, reject_by_annotation=False, verbose=False)
    alpha_vals = compute_prestim_alpha(e_stim, roi=('O1', 'O2'), tmin=-0.5, tmax=0.0)
    assert alpha_vals.shape == (len(e_stim),)
    assert np.all(np.isfinite(alpha_vals))
    # Envelope should be non-negative
    assert np.all(alpha_vals >= 0.0)


def test_compute_mrcp_lrp_on_synthetic_has_expected_signs():
    raw, onset_code, resp_code = _make_synthetic_raw_with_events()
    events = mne.find_events(raw, stim_channel='Trigger', shortest_event=1, verbose=False)
    rp_events = events[events[:, 2] == resp_code]
    e_resp = mne.Epochs(raw, rp_events, event_id={'resp': resp_code}, tmin=-1.8, tmax=0.5,
                        baseline=None, preload=True, reject_by_annotation=False, verbose=False)
    mrcp_mean, lrp_at_m100, lrp_onset = compute_mrcp_lrp(e_resp)
    assert mrcp_mean.shape == (len(e_resp),)
    assert lrp_at_m100.shape == (len(e_resp),)
    assert lrp_onset.shape == (len(e_resp),)
    # MRCP mean over [-0.8,-0.2] should be negative for our ramp
    assert mrcp_mean[0] < 0.0
    # LRP at -100 ms window should be negative (C3-C4 = -2e-6)
    assert lrp_at_m100[0] < 0.0


