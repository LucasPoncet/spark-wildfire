"""
Module de traitement de signaux audio.

Ce module fournit une classe SignalProcessor pour manipuler et analyser des signaux audio,
incluant découpe, mixage, ajustement de gain, FFT et filtrage passe-bas.
Une fonction utilitaire permet d'appliquer un filtre passe-bas à une liste de fichiers.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt

class SignalProcessor:
    """Classe de traitement de signaux audio.

    Fournit des méthodes pour manipuler et analyser des signaux audio :
    découpe en instances, addition de signaux, ajustement de gain,
    transformée de Fourier discrète et filtrage passe-bas.
    """

    def cut_into_instance(self, path: str, nb_instance: int) -> np.ndarray:
        """Découpe un signal audio en plusieurs instances de taille similaire.

        Args:
            path: Chemin vers le fichier audio à découper.
            nb_instance: Nombre d'instances à créer.

        Returns:
            np.ndarray: Tableau 2D de forme (nb_instance, taille_instance) contenant les segments.
        """
        y, _ = librosa.load(path, sr=None)
        instance_size = int(len(y) / ((nb_instance + 1) // 2))
        return np.array([
            y[max(0, (i-1) * (instance_size // 2)):min((i+1) * (instance_size // 2), len(y))]
            for i in range(1, nb_instance + 1)
        ])

    def add_signal(self, path1: str, path2: str) -> np.ndarray:
        """Additionne deux signaux audio et retourne leur moyenne.

        Args:
            path1: Chemin vers le premier fichier audio.
            path2: Chemin vers le second fichier audio.

        Returns:
            np.ndarray: Signal résultant (moyenne des deux signaux, avec padding si nécessaire).
        """
        y1, _ = librosa.load(path1, sr=None)
        y2, _ = librosa.load(path2, sr=None)
        len1, len2 = len(y1), len(y2)

        if len1 == len2:
            return (np.array(y1) + np.array(y2)) / 2
        elif len1 < len2:
            padded_y1 = np.pad(y1, (0, len2 - len1), 'constant')
            return (padded_y1 + np.array(y2)) / 2
        else:
            padded_y2 = np.pad(y2, (0, len1 - len2), 'constant')
            return (np.array(y1) + padded_y2) / 2

    def change_gain(self, path: str, multiplication_factor: float) -> np.ndarray:
        """Ajuste le gain d'un signal audio par multiplication.

        Args:
            path: Chemin vers le fichier audio.
            multiplication_factor: Facteur multiplicatif pour le gain.

        Returns:
            np.ndarray: Signal avec gain ajusté.
        """
        y, _ = librosa.load(path, sr=None)
        return multiplication_factor * np.array(y)

    def discret_FFT(self, path: str, instance: np.ndarray = None, windows: str = None) -> tuple[np.ndarray, float, np.ndarray]:
        """Calcule la FFT discrète avec fenêtrage optionnel.

        Args:
            path: Chemin vers le fichier audio.
            instance: Instance de signal à utiliser (non utilisé actuellement, conservé pour compatibilité).
            windows: Type de fenêtre ('Hann' pour fenêtre de Hann, None pour pas de fenêtre).

        Returns:
            tuple: (fft_y, sr, frequencies) où :
                - fft_y: Transformée de Fourier (complexe)
                - sr: Taux d'échantillonnage (Hz)
                - frequencies: Fréquences associées à chaque point FFT
        """
        y, sr = librosa.load(path, sr=None)
        window = np.sin(np.pi * np.arange(len(y)) / (len(y) - 1)) ** 2 if windows == 'Hann' else 1
        fft_y = np.fft.fft(np.array(y) * window)
        frequencies = np.fft.fftfreq(len(fft_y), d=1/sr)
        return fft_y, sr, frequencies

    def low_pass_filter(
        self,
        fft_y: np.ndarray,
        sr: float,
        frequencies: np.ndarray,
        filter_frequency: float,
        filter: str = 'Hann'
    ) -> np.ndarray:
        """Applique un filtre passe-bas à une FFT et retourne le signal temporel filtré.

        Args:
            fft_y: FFT du signal (tableau complexe).
            sr: Taux d'échantillonnage (Hz).
            frequencies: Fréquences associées à la FFT.
            filter_frequency: Fréquence de coupure (Hz).
            filter: Type de filtre :
                - 'Hann': Transition douce avec fenêtre de Hann
                - 'butterworth': Transition raide (masque binaire)
                - autre: Aucun filtre

        Returns:
            np.ndarray: Signal temporel filtré (partie réelle de l'IFFT).
        """
        N = len(fft_y)

        if filter == 'Hann':
            hann_window = np.hanning(N)
            hann_window = np.roll(hann_window, N // 2)
            filter_mask = np.where(np.abs(frequencies) <= filter_frequency, hann_window, 0)
        elif filter == 'butterworth':
            filter_mask = (np.abs(frequencies) <= filter_frequency).astype(float)
        else:
            filter_mask = np.ones(N)

        filtered_fft = fft_y * filter_mask
        return np.real(np.fft.ifft(filtered_fft))

def apply_low_pass_filter_to_paths(
    paths: list[str],
    filter_frequency: float,
    filter_type: str = 'Hann'
) -> list[np.ndarray]:
    """Applique un filtre passe-bas à une liste de fichiers audio.

    Args:
        paths: Liste des chemins vers les fichiers audio à filtrer.
        filter_frequency: Fréquence de coupure du filtre (Hz).
        filter_type: Type de filtre à appliquer ('Hann' ou 'butterworth').

    Returns:
        list[np.ndarray]: Liste des signaux audio filtrés.
    """
    processor = SignalProcessor()
    return [
        processor.low_pass_filter(*processor.discret_FFT(path), filter_frequency, filter_type)
        for path in paths
    ]