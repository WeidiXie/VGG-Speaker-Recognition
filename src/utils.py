# Third Party
import librosa
import numpy as np
import time as timelib
import scipy
import soundfile as sf
import scipy.signal as sps
from scipy import interpolate
# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(vid_path, sr, mode='train'):

    #t1=timelib.time()
    #print("start loading wav")
    #print(sr)
    #wav, sr_ret = librosa.load(vid_path, sr=sr)
    #sr_ret, wav = scipy.io.wavfile.read(vid_path)

    wav, sr_ret = sf.read(vid_path)
    #sr_ret, old_audio = scipy.io.wavfile.read(vid_path)
    #if sr_ret != sr:
    #    new_rate = sr
    #    number_of_samples = round(len(old_audio) * float(new_rate) / sr_ret)
    #    wav = sps.resample(old_audio, number_of_samples)


    #assert sr_ret == 16000, "we need same samplerate as librosa originally provided but is: " +str(sr_ret)
    #print("finish loading wav", timelib.time()-t1)
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    #print("starting loading a datum")
    #t1 = timelib.time()
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        if time > spec_len:
            randtime = np.random.randint(0, time-spec_len)
            spec_mag = mag_T[:, randtime:randtime+spec_len]
        else:
            spec_mag = np.pad(mag_T, ((0, 0), (0, spec_len - time)), 'constant')
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    #print("finished loading a datum", timelib.time() - t1)
    return (spec_mag - mu) / (std + 1e-5)


