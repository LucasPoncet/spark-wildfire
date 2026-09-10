"""
dataset folder
|_ sound track folder
   |_ sound track
   |_ measures
"""
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy import signal


path = "data/raw_recordings/fsc22/ESC-50-master/ESC-50-master/audio/1-977-A-39.wav"
path1 = "data/raw_recordings/fsc22/ESC-50-master/ESC-50-master/audio/1-977-A-39.wav"
path2 = "data/raw_recordings/fsc22/ESC-50-master/ESC-50-master/audio/1-1791-A-26.wav"

def cut_into_instance(
        path: str,
        nb_instance: int
                    ):
    y, sr = librosa.load(path, sr= None)
    instance_size = int(len(y)/((nb_instance+1)//2))
    instance_sample = np.array([y[max(0,(i-1)*(instance_size//2)):
                               min((i+1)*(instance_size//2), len(y))
                            ] 
                                   for i in range(1,nb_instance +1)
                            ]
                            )
    return instance_sample

def add_signal(
        path1: str,
        path2: str
        ):
    y1, sr1 = librosa.load(path1, sr= None)
    y2, sr2 = librosa.load(path2, sr= None)
    if len(y2) == len(y1):
        return (np.array(y1) + np.array(y2))/2
    if len(y1)< len(y2):
        return (np.array(y1 + [0 for i in range(len(y2)- len(y1))]) + np.array(y2))/2
    if len(y1) > len(y2):
        return (np.array(y1) + np.array(y2 + [0 for i in range(len(y1)- len(y2))]))/2


def change_gain(
        path: str,
        multiplication_factor: float
):
    return multiplication_factor*np.array(librosa.load(path, sr= None)[0])

def discret_FFT(
        path: str,
        instance: np.array =None,
        windows: str = None
        ):
    y, sr = librosa.load(path, sr= None)
    if windows == 'Hann':
        windows = np.sin(np.pi*np.array(range(len(y)))/(len(y)-1))**2
    else:
        windows = 1
    fft_y = np.fft.fft(np.array(y)*windows)
    frequencies = np.fft.fftfreq(len(fft_y), d=1/sr)  # Tableau des fréquences associées à la FFT
    return fft_y, sr, frequencies



def low_pass_filter(
    fft_y: np.array,          # FFT du signal (complex array)
    sr: float,               # Taux d'échantillonnage (Hz) → **À ajouter en paramètre !**
    frequencies: np.array, # liste des frequences associé à chaque point
    filter_frequency: float, # Fréquence de coupure (Hz)
    filter: str = 'Hann'      # Type de filtre : 'Hann' ou 'butterworth'
):
    """
    Applique un filtre passe-bas à une FFT et retourne le signal temporel filtré.
    """
    N = len(fft_y)
    # --- 1. Créer le masque de filtre ---
    if filter == 'Hann':
        # Fenêtre de Hann centrée sur 0 (pour adoucir la transition)
        hann_window = np.hanning(N)
        hann_window = np.roll(hann_window, N//2)  # Centrer la fenêtre
        # hann_window[int(N*filter_frequency / sr) // 2: -1] = 0
        # hann_window = np.concat([hann_window, np.zeros(N-int(N*filter_frequency / sr))])
        # Masque : 1 pour |f| ≤ filter_frequency, multiplié par la fenêtre de Hann
        filter_mask = np.where(np.abs(frequencies) <= filter_frequency, hann_window, 0)

    elif filter == 'butterworth':
        # Filtre passe-bas idéal (Butterworth → transition raide)
        # On utilise un masque binaire pour simuler un filtre idéal dans le domaine fréquentiel
        filter_mask = (np.abs(frequencies) <= filter_frequency).astype(float)

    else:
        filter_mask = np.ones(N)  # Aucun filtre

    # --- 2. Appliquer le masque à la FFT ---
    filtered_fft = fft_y * filter_mask
    # return filtered_fft, frequencies
    # --- 3. Retourner le signal temporel (IFFT) ---
    return np.real(np.fft.ifft(filtered_fft))
        
test, sr, frequencies = discret_FFT(path)
test2 = low_pass_filter(test,sr, frequencies, 10000, filter = 'Hann')

# plt.plot(frequencies, test/np.max(test))
plt.plot(range(len(test2)), test2/np.max(test2), alpha = 0.5)
plt.plot(range(len(test)), librosa.load(path1, sr= None)[0], alpha = 0.5)
# plt.plot(range(len(test)), librosa.load(path, sr= None)[0], alpha = 0.5)
plt.show()

# test = cut_into_instance(path, 29)
# plt.figure()
# for lign in test:
#     plt.plot(range(0, len(lign)), lign)
# plt.show()