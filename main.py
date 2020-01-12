import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import parameterization as param
from parameterization import parametrization

# test na przykładowym sygnale
frame_len = 200
overlap_len = 40

fs, signal = wavfile.read('./Komendy/prawo/prawo_2.wav')
signal = signal[:, 0]
signal_float = signal / np.iinfo(np.int16).max
signal_len = len(signal)
signal = param.preemphasis(signal, 0.95)  # filtr preemfazy
start = 0

n_frames = int(np.floor(signal_len / (frame_len - overlap_len))) - 2
frames = np.zeros((n_frames, frame_len))

for idx1 in range(n_frames):
    frames[idx1, :] = signal[start:start + frame_len]
    start += (frame_len - overlap_len)

l_p = np.zeros(13)
l_d = np.zeros(13)

param = np.zeros((n_frames, 39))
for idx1 in range(n_frames):
    param[idx1] = parametrization(frames[idx1, :], l_p, l_d)
    l_p = param[idx1, 0:13]
    l_d = param[idx1, 14:27]
    print(idx1)

fig, axs = plt.subplots(2)

fig.suptitle('Vertically stacked subplots')
axs[0].plot(signal_float)
axs[1] = plt.imshow(np.transpose(param), origin='lowest', aspect='auto')
plt.colorbar(orientation='horizontal')
plt.show()
