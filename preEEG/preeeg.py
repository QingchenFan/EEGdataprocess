import mne
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from scipy import signal
# Load EEG Data
raw = mne.io.read_raw_gdf('/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/A01T.gdf')
print('-raw.info-\n', raw.info)
##raw.plot(title='raw-info')


# drop channels
raw.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})
raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

print('-drop channels-\n', raw.info)
# Filter
raw.load_data()
raw.filter(l_freq=8, h_freq=30, method='iir')

##raw.plot(title='2')
#plt.show()
# Events
events = mne.events_from_annotations(raw)
print('-events-\n', events)
print('-20 events-\n', events[0][0:20])

# Epoch
epoch = mne.Epochs(raw, events[0], event_id=[7, 8, 9, 10], tmin=-0.2, tmax=0.7, preload=True)
print('-epoch-\n', epoch)

print('-epoch_data-\n', epoch.get_data(), epoch.get_data().shape)

print('-epoch 1 dim-\n', epoch.get_data()[0][1], epoch.get_data()[0][1].shape)
print('-epoch 2 dim-\n', epoch.get_data()[1], epoch.get_data()[1].shape)
print('-epoch 3 dim-\n', epoch.get_data()[287], epoch.get_data()[287].shape)
#epoch (288,22,226)
#epoch.plot(events=events[0], block=True)


# Label
labels = epoch.events[:, -1]
print('-Label-\n', labels)
print('-Label-length-\n', labels.shape)

evoked_1 = epoch['7'].average()
evoked_2 = epoch['8'].average()
evoked_3 = epoch['9'].average()
evoked_4 = epoch['10'].average()

dicts = {'left': evoked_1, 'right': evoked_2, 'foot': evoked_3, 'tongue': evoked_4}
##mne.viz.plot_compare_evokeds(dicts)

##evoked_1.plot()
left_data = evoked_1.get_data()
print('-left_data-\n', left_data, left_data.shape, type(left_data))

#np.savetxt('./A01T_left.txt', left_data)



# def read_data(path):
#     raw = mne.io.read_raw_gdf(path, eog=['EOG-left', 'EOG-central', 'EOG-right'],preload=True)
#     raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
#     raw.filter(l_freq=8, h_freq=30, method='iir')
#     raw.set_eeg_reference()
#     events = mne.events_from_annotations(raw)
#     epoch = mne.Epochs(raw, events[0], event_id=[7, 8, 9, 10], tmin=-0.1, tmax=0.7)
#     labels = epoch.events[:, -1]
#     features = epoch.get_data()
#     return labels, features

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

    N_trials, N_eegs, time_bins, _ = X.shape
    print('N_trials', N_trials)
    print('N_eegs', N_eegs)
    print('time_bins', time_bins)

    # values for cwt
    freq = np.linspace(freq_range[0], freq_range[1], freq_bins)
    widths = w * fs / (2 * freq * np.pi)
    X_cwt = np.zeros((N_trials, N_eegs, widths.shape[0], time_bins))

    print('Performing discrete CWT convolutions...')
    for trial in tqdm_notebook(range(N_trials), desc='Trials'):
        for eeg in tqdm_notebook(range(N_eegs), desc='EEG Channel', leave=False):
            X_cwt[trial, eeg, :, :] = np.abs(signal.cwt(np.squeeze(X[trial, eeg, :, ]), signal.morlet2, widths, w=w))

    return X_cwt
aa = epoch.get_data()

res = morlet_wavelet_transform(aa)

def stft_data(X, window_size=64, stride=24, freq=(0, 30), draw=False):
    '''
    Short-time-Fourier transform (STFT) function
    INPUTS:
    X - EEG data (num_trials, num_eeg_electrodes, time_bins,1)
    window_size - fixed # of time bins to analyze frequency components within
    stride - distance between starting position of adjacent windows
    freq - frequency range to obtain spectral power components
    OUTPUTS:
    X_stft - STFT transformed eeg data (num_trials, num_eeg_electrodes*freq_bins,time_bins,1)
    num_freq - number of frequency bins
    num_time - number of time bins
    '''
    fs = 250
    num_trials, num_eegs, N, _ = X.shape
    assert (N-window_size)/stride % 1 == 0, 'Window size and stride not valid for length of data'

    f, t, Zxx = signal.stft(X[0, 0, :, :], fs=fs, axis=-2, nperseg=window_size, noverlap=window_size-stride)

    wanted_freq = np.where(np.logical_and(f >= freq[0], f <= freq[1]))[0]
    num_freq = wanted_freq.shape[0]
    num_time = t.shape[0]

    X_stft = np.empty((int(num_trials), int(num_freq*num_eegs),int(num_time),1))
    for i in range(num_trials):
        #for j in range(num_eegs):
        f, t, Zxx = signal.stft(X[i, :, :, :], fs=fs, axis=-2, nperseg=window_size, noverlap=window_size-stride)
        wanted_Zxx = Zxx[:, wanted_freq, :, :]
        wanted_Zxx = np.reshape(np.transpose(wanted_Zxx, (0, 1, 3, 2)), (num_freq*num_eegs, num_time, 1))
        if draw==True:
            draw_f = np.repeat(np.expand_dims(f[wanted_freq], axis=1), num_eegs, axis=1).T.flatten()
            plt.pcolormesh(t, range(draw_f.shape[0]), np.abs(np.squeeze(wanted_Zxx)))
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()
        X_stft[i] = wanted_Zxx

    return X_stft, num_freq, num_time