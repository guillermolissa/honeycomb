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
import pywt
from collections import defaultdict, Counter


sample_freq = 22050 





def preprocessing(raw_signal, sample_freq = 22050, cutoff_freq = 5):

    # Rectification 
    rectified_signal = np.abs(raw_signal)

    # Low-pass filtering -> envelope
    norm_cutoff_freq = cutoff_freq / (sample_freq / 2.0)
    b, a = signal.butter(2, norm_cutoff_freq, 'low')
    envelope = signal.filtfilt(b, a, rectified_signal, axis=0)

    return envelope, rectified_signal