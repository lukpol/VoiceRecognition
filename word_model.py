import numpy as np
import parameterization as param
from scipy import signal as sig
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
import pickle

def resampling(x, fs_old: int, fs_new: int):
    x_len = len(x)
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


    out_coeffs = np.zeros((n_frames,n_coeff))
    for idx1 in range(n_frames):
        out_coeffs[idx1] = param.parametrization(x[idx1, :])
    return out_coeffs


def vad(frames): 
    energy = np.mean(np.abs(frames))
    return frames[np.mean(np.abs(frames), 1) >= 0.9*energy,:]
          
    
def vad_rt(frame, thr = 25):
    if 20*np.log10(np.std(frame)) > thr:
        return 1 
    return 0
        

class WordModel:
    name: str
    # jakaś struktura modelu
    komendy = ['wylacz', 'wstecz', 'wozek', 'wlacz', 'swiatlo', 'stop', 'start', 'prawo', 'naprzod', 'lewo', 'gora', 'dol']
    
    def __init__(self, name: str):
        self.name = name

    def save(self, path: str):
        with open(path + self.name, 'wb') as fd:
            pickle.dump(self.model, fd)

    def load(self, path: str, names=komendy):
        model = {}
        for name in names:
            with open(path + name, 'rb') as fd:
                tmp = pickle.load(fd)
                model[name] = tmp[name]
        return model
    
    def gaussian(self, coeffs):
#         coeffs = (coeffs.T - np.min(coeffs,1)).T
        N, C = coeffs.shape
        gauss = np.zeros((2,C))
        bounds = np.linspace(-1,0.9,20)
        hist = np.zeros((20, C))
        for ind, bound in enumerate(bounds):
            hist[ind,:] = np.sum(np.logical_and(coeffs>=bound,coeffs<bound+0.1),0)/N
        gauss[0,:] = np.mean((hist.T*(bounds+0.05)).T,0)
        gauss[1,:] = np.std((hist.T*(bounds+0.05)).T,0)
        return gauss
    
    def discriminate(self, path, models=None):
        if models == None:
            models = self.load('Models/', names=self.komendy)
        fs, signal = wavfile.read(path)
        signal = signal[:, 0]
        signal = param.preemphasis(signal, 0.95)
        if fs != 8000:
            signal = resampling(signal, fs, 8000)
        frames = framing(signal, 200, 40)
        frames = vad(frames)
        coeff = frames_parametrization(frames, 13)
        params = self.gaussian(coeff)
        
        dist = np.zeros((len(self.komendy)))
        for ind, name in enumerate(self.komendy):
            dist[ind] = np.sum( 0.25 * np.log(0.25 * (params[1,:]**2 / models[name][1,:]**2 + 
                                          models[name][1,:]**2 / params[1,:]**2 + 2)) + 0.25 * 
                                            ((params[0,:] - models[name][0,:])**2 / 
                                             (params[0,:]**2 / models[name][0,:]**2)) )
            
            
        print('znaleziono', self.komendy[np.argmin(dist)])
        return self.komendy[np.argmin(dist)]
    
    def train(self, path: str):
        # trzeba zrobić jakiś model danego słowa (nie wiem, HMM, za bardzo nie pamiętam jak to się robiło)
        # 1) Wczytanie wszystkich sygnałów danego słowa
        # 2) Resampling do 8kHz
        # 3) Ramkowanie (25-30ms + nakładka 25%)
        # 4) VAD
        # 5) Parametryzacja (26 elementów -> 13 elementów (energia+12MFCC) + delta
        # 6) Stworzenie modelu GMM

        coeffs = np.zeros((0,13))
        for k in range(1, 7):
            fs, signal = wavfile.read(path + self.name + "_" + str(k) + ".wav")
            signal = signal[:, 0]
            signal = param.preemphasis(signal, 0.95)
            if fs != 8000:
                signal = resampling(signal, fs, 8000)
            frames = framing(signal, 200, 40)
            frames = vad(frames)
            coeffs = np.append(coeffs, frames_parametrization(frames, 13), axis = 0)
        self.model = {self.name: self.gaussian(coeffs)}
        
        print("GMM model of \"" + self.name + "\" DONE")


        return ValueError
