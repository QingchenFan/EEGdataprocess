import signal

import mne
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy import signal
raw = mne.io.read_raw_gdf('/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/A01T.gdf')

raw.info
raw.get_data().shape

raw.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})
raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

#Filter
raw.load_data()
raw.filter(l_freq=8, h_freq=30, method='iir')

events = mne.events_from_annotations(raw)

epoch = mne.Epochs(raw, events[0], event_id=[7, 8, 9, 10], tmin=-0.2, tmax=0.7, preload=True)
#epoch.plot(events=events[0], block=True)

def morlet_wavelet_transform(X, fs=250, freq_range=(1, 15), freq_bins=100, w=5):
    '''
    Discrete continous wavelet transform of eeg data convolved with complex morlet wavelet
    INPUTS:
    X - EEG data (num_trials, num_eeg_electrodes, time_bins,1)
    fs - sampling rate in Hz
    freq_range - tuple containing min and max freq range to perform analysis within
    freq_bins - number of points between freq range being analyzed
    w - Omega0 for complex morlet wavelet
    OUTPUTS:
    X_cwt - Wavlet transformed eeg data (num_trials, num_eeg_electrodes,freq_bins,time_bins)
    '''

    N_trials, N_eegs, time_bins= X.shape


    # values for cwt
    freq = np.linspace(freq_range[0], freq_range[1], freq_bins)
    widths = w * fs / (2 * freq * np.pi)
    X_cwt = np.zeros((N_trials, N_eegs, widths.shape[0], time_bins))

    print('Performing discrete CWT convolutions...')
    for trial in tqdm.tqdm_notebook(range(N_trials), desc='Trials'):
        for eeg in tqdm.tqdm_notebook(range(N_eegs), desc='EEG Channel', leave=False):
            X_cwt[trial, eeg, :, :] = np.abs(signal.cwt(np.squeeze(X[trial, eeg, :, ]), signal.morlet2, widths, w=w))

    return X_cwt

source_data = epoch.get_data()
print('-source_data-',source_data.shape)
X_cwt = morlet_wavelet_transform(source_data)
print('X_cwt', X_cwt)  # (288, 22, 100, 226)
# num_trials, num_eeg_electrodes,freq_bins,time_bins