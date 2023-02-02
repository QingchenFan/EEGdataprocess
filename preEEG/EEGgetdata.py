import mne
import glob
import signal

import numpy as np
import tqdm
from scipy import signal

def read_data(path):
    raw = mne.io.read_raw_gdf(path, eog=['EOG-left', 'EOG-central', 'EOG-right'], preload=True)
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    raw.filter(l_freq=8, h_freq=30, method='iir')
    raw.set_eeg_reference()
    events = mne.events_from_annotations(raw)
    epoch = mne.Epochs(raw, events[0], event_id=[7, 8, 9, 10], tmin=-0.1, tmax=0.7,on_missing='warn')
    labels = epoch.events[:, -1]
    features = epoch.get_data()
    cwtfeatures = morlet_wavelet_transform(features)
    print('-cwtfeatures-', cwtfeatures.shape)
    return labels, cwtfeatures

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
if __name__ == '__main__':
    path = '/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/*T.gdf'
    for dataPath in glob.glob(path):
         print('datapath-', dataPath)
         labels, features = read_data(dataPath)
         print(labels.shape)
         print(features.shape)
         #np.save('./cwtlabel_'+dataPath[-8:-4], labels)
         #np.save('./cwtfeatures_'+dataPath[-8:-4], features)