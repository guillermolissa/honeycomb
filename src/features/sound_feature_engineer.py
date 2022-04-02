import os
import time
import numpy as np
import pandas as pd
from scipy.io import wavfile as wv
from scipy import signal
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.fftpack import fft
from scipy.signal import welch
from scipy import stats
from detect_peaks import detect_peaks
from collections import defaultdict, Counter
import pywt
import librosa



def calculate_statistics(raw_signal):
    n5 = np.nanpercentile(raw_signal, 5)
    n25 = np.nanpercentile(raw_signal, 25)
    n75 = np.nanpercentile(raw_signal, 75)
    n95 = np.nanpercentile(raw_signal, 95)
    median = np.nanpercentile(raw_signal, 50)
    mean = np.nanmean(raw_signal)
    std = np.nanstd(raw_signal)
    var = np.nanvar(raw_signal)
    rms = np.nanmean(np.sqrt(raw_signal**2))
    maxv =np.max(raw_signal)
    minv = np.min(raw_signal)
    skew = stats.skew(raw_signal)
    kurtosis = stats.kurtosis(raw_signal)
    absmean = np.abs(raw_signal).mean()
    absstd = np.abs(raw_signal).std()
    
    
    return [n5, n25, n75, n95, median, mean, std, var, rms, maxv, minv, skew, kurtosis, absmean, absstd]



def calculate_signal_features(raw_signal):
    relamp = np.max(raw_signal) / np.abs(np.min(raw_signal))
    amp = np.max(raw_signal) - np.abs(np.min(raw_signal))
    ssum = np.sum(raw_signal)
    diffmean = np.mean(np.diff(raw_signal))
    
    return [relamp, amp, ssum, diffmean]


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]

def get_autocorr_values(y_values, f_s):
    N = len(y_values)
    T = 1/f_s
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values

def get_fft_values(y_values, f_s):
    T = 1/f_s
    N = len(y_values)
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values


def get_psd_values(y_values, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values


def get_first_n_peaks(x,y,no_peaks=5):
    x_, y_ = list(x), list(y)
    if len(x_) > no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks

    
def get_peaks(x_values, y_values, mph):
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    return peaks_x + peaks_y


def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy = stats.entropy(probabilities)
    return entropy


def mel(wave, sr, n_fft, hop_length, n_mels, max_pad_len, normalize, scaling, padding_mode, **kwargs):

    if normalize:
        wave = librosa.util.normalize(wave) # normalizing data before mel

    # making melspect from signal
    S = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=sr//2, **kwargs)

    # scaling
    if scaling:
        S = preprocessing.scale(S, axis=1)

    if max_pad_len:
        if S.shape[1] > max_pad_len:
            S = S[:,:max_pad_len]
        else:
            pad_width = max_pad_len - S.shape[1]
            S = np.pad(S, pad_width=((0, 0), (0, pad_width)), mode=padding_mode)
    
    S = librosa.power_to_db(S)
    S = S.astype(np.float32)
   
    return S


def mfcc(wave, sr, n_fft, hop_length, n_mfcc, max_pad_len, normalize, padding_mode, **kwargs):

    #wave, sr = librosa.load(wav_file, mono=True)
    
    #if 0 < len(wave): # workaround: 0 length causes error
    #    wave, _ = librosa.effects.trim(wave) # trim, top_db=default(60)

    if normalize:
        wave = librosa.util.normalize(wave) # normalizing data before mfcc

    # making mfcc from signal
    S = librosa.feature.mfcc(y=wave, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc, **kwargs)

    if max_pad_len:
        if S.shape[1] > max_pad_len:
            S = S[:,:max_pad_len]
        else:
            pad_width = max_pad_len - S.shape[1]
            S = np.pad(S, pad_width=((0, 0), (0, pad_width)), mode=padding_mode)
   
    return S