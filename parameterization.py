import math
import numpy as np

mfcc_len = 0
mfcc_filterBank = 0


def hanning_window(x: list) -> list:
    signal_len = len(x)
    window = np.zeros(signal_len)
    for n in range(signal_len):
        window[n] = 0.5 * (1 - math.cos((2 * math.pi * n) / (signal_len - 1)))
    out = x * window

    return out


def hamming_window(x: list) -> list:
    signal_len = len(x)
    window = np.zeros(signal_len)
    for n in range(signal_len):
        window[n] = 0.54 - 0.46 * math.cos((2 * math.pi * n) / (signal_len - 1))
    out = x * window

    return out


def preemphasis(x: list, a: float) -> list:
    signal_len = len(x)
    out = np.zeros(signal_len)

    out[0] = x[0]
    for k in range(signal_len - 1):
        out[k + 1] = x[k] - a * x[k]

    return list(out)


def dft(x: list) -> list:
    signal_len = len(x)
    wk = np.exp(2j * np.pi / signal_len)
    out_tmp = np.zeros(signal_len, dtype=np.complex_)

    for k in range(signal_len):
        for n in range(signal_len):
            out_tmp[k] += x[n] * pow(wk, -k * n)

    out_tmp = pow(np.absolute(out_tmp), 2)
    out = list(out_tmp[0:math.floor(signal_len / 2)])
    return out


def fft(x: list) -> list:
    fft_tmp = np.absolute(np.fft.rfft(x))  # Magnitude of the FFT
    out = ((1.0 / len(x)) * (fft_tmp ** 2))
    return out


def dct(x: list) -> list:
    signal_len = len(x)
    out = np.zeros(signal_len)

    for k in range(signal_len):
        for n in range(signal_len):
            out[k] += x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * signal_len))
        out[k] = 2 * out[k]
    return list(out)


def energy(x: list):
    sum_tmp = sum([pow(el, 2) for el in x])
    if sum_tmp > 0:
        out = 20 * np.log10(sum_tmp) / len(x)
    else:
        out = -120 / len(x)
    return out


def mfcc_filtering(x: list, nfilt: int) -> list:
    global mfcc_filterBank

    mfc = np.zeros(nfilt)
    for n in range(nfilt):
        mfc_tmp = np.dot(x, mfcc_filterBank[n])
        if mfc_tmp > 0:
            mfc[n] = 20 * np.log10(mfc_tmp)
        else:
            mfc[n] = -120

    return list(mfc)


def make_filterBank(n: int, nfilt: int) -> list:
    low_mel = 0
    high_mel = (2595 * np.log10(1 + 4000 / 700))
    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    idx = np.floor((2 * n - 1) * hz_points / 8000)

    bank = np.zeros((nfilt, n))
    for m in range(1, nfilt + 1):
        f_m_minus = int(idx[m - 1])  # left
        f_m = int(idx[m])  # center
        f_m_plus = int(idx[m + 1])  # right

        for k in range(f_m_minus, f_m):
            bank[m - 1, k] = (k - idx[m - 1]) / (idx[m] - idx[m - 1])
        for k in range(f_m, f_m_plus):
            bank[m - 1, k] = (idx[m + 1] - k) / (idx[m + 1] - idx[m])

    return list(bank)


def delta(x: list, y: list) -> list:
    out = np.zeros(len(x))
    for n in range(len(x)):
        out[n] = x[n] - y[n]
    return list(out)


def parametrization(x, last_param, last_delta):
    global mfcc_filterBank
    global mfcc_len

    x_normalized = [el / max(abs(x)) for el in x]  # normalizacja sygnału

    x_windowed = hamming_window(x_normalized)  # okienkowanie (Hamming)

    spectrum = fft(x_windowed)  # rzeczywista część widma mocy TODO: FFT

    if mfcc_len != len(spectrum):  # sprawdzanie dostępności banku filtrów
        mfcc_filterBank = make_filterBank(len(spectrum), 40)
        mfcc_len = len(spectrum)

    MFC = mfcc_filtering(spectrum, 40)  # filtracja filtrami melowymi + energia każdego pasma

    MFCC = dct(MFC)  # dekorelacja DCT

    final_param = MFCC[1:13]  # skrócenie do do 12 współczynników

    n = np.arange(12)
    cep_lifter = 23
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    final_param *= lift
    final_param = list(final_param)

    final_param.insert(0, energy(x))  # dodanie energii na początek

    delta1 = delta(final_param, last_param)  # dodanie pierwszej pochodnej
    final_param.extend(delta1)

    delta2 = delta(delta1, last_delta)  # dodanie drugiej pochodnej
    final_param.extend(delta2)

    return final_param
