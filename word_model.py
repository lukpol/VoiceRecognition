import numpy as np
import parameterization as param
from scipy import signal as sig
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture


def resampling(x, fs_old: int, fs_new: int):
    x_len = len(x)c
    out_len = int(np.floor(x_len * fs_new / fs_old))
    out = sig.resample(x, out_len)
    return out


def framing(x, frame_len: int, overlap_len: int):
    start = 0
    signal_len = len(x)
    n_frames = int(np.floor(signal_len / (frame_len - overlap_len))) - 2
    frames = np.zeros((n_frames, frame_len))

    for idx1 in range(n_frames):
        frames[idx1, :] = x[start:start + frame_len]
        start += (frame_len - overlap_len)
    return frames


def frames_parametrization(x, n_coeff: int):
    n_frames = len(x)
    l_p = np.zeros(n_coeff)
    l_d = np.zeros(n_coeff)

    out_coeffs = np.zeros((n_frames, 3 * n_coeff))
    for idx1 in range(n_frames):
        out_coeffs[idx1] = param.parametrization(x[idx1, :], l_p, l_d)
        l_p = out_coeffs[idx1, 0:n_coeff]
        l_d = out_coeffs[idx1, n_coeff + 1:2 * n_coeff + 1]
    return out_coeffs


def vad(frames): 
    energy = np.mean(np.abs(frames))
    return frames[np.mean(np.abs(frames), 1) >= 0.9*energy,:]
          
    
def vad_rt(frame, thr = 25):
    if 20*np.log10(np.std(frame)) > thr:
        return 1 
    return 0
        
def Gaussian(coeffs):
    return np.array((np.mean(coeffs, axis = 0), np.std(coeffs, axis = 0)))


class WordModel:
    name: str
    # jakaś struktura modelu

    def __init__(self, name: str):
        self.name = name

    def save(self, path: str):
        np.save()
        return ValueError

    def load(self, path: str):
        return ValueError

    def train(self, path: str):
        # trzeba zrobić jakiś model danego słowa (nie wiem, HMM, za bardzo nie pamiętam jak to się robiło)
        # 1) Wczytanie wszystkich sygnałów danego słowa
        # 2) Resampling do 8kHz
        # 3) Ramkowanie (25-30ms + nakładka 25%)
        # 4) VAD
        # 5) Parametryzacja (26 elementów -> 13 elementów (energia+12MFCC) + delta
        # 6) Stworzenie modelu GMM

        coeffs = np.zeros((0,39))
        for k in range(1, 7):
            fs, signal = wavfile.read(path + self.name + "_" + str(k) + ".wav")
            signal = signal[:, 0]
            signal = param.preemphasis(signal, 0.95)
            if fs != 8000:
                signal = resampling(signal, fs, 8000)
            frames = framing(signal, 200, 40)
            frames = vad(frames)
            coeffs = np.append(coeffs, frames_parametrization(frames, 13), axis = 0)
        model = Gaussian(coeffs)
        print("GMM model of \"" + self.name + "\" DONE")


        return ValueError
