import librosa
import numpy as np
import matplotlib.pyplot as plt

def discret_FFT(
    path: str,
    instance: np.ndarray = None,
    windows: str = None
) -> tuple[np.ndarray, float, np.ndarray]:
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
    window = np.sin(np.pi * np.arange(range(len(y))) / (len(y) - 1)) ** 2 if windows == 'Hann' else 1
    fft_y = np.fft.fft(np.array(y) * window)
    frequencies = np.fft.fftfreq(len(fft_y), d=1/sr)
    return fft_y, sr, frequencies

def apply_filter(
    fft_y: np.ndarray,
    sr: float,
    frequencies: np.ndarray,
    filter_frequency: float = None,
    band_frequencies: tuple[float, float] = None,
    filter_type: str = 'low',
    window_type: str = 'Hann',
    butterworth_order: int = 1
) -> np.ndarray:
    """Applique un filtre (passe-bas, passe-haut ou passe-bande) à une FFT et retourne le signal temporel filtré.

    Args:
        fft_y: FFT du signal (tableau complexe).
        sr: Taux d'échantillonnage (Hz).
        frequencies: Fréquences associées à la FFT.
        filter_frequency: Fréquence de coupure pour les filtres passe-bas ou passe-haut (Hz).
        band_frequencies: Tuple de fréquences (low, high) pour le filtre passe-bande (Hz).
        filter_type: Type de filtre : 'low', 'high', ou 'band'.
        window_type: Type de fenêtre pour le filtre : 'Hann', 'cutoff', ou 'butterworth'.
        butterworth_order: Ordre du filtre Butterworth (par défaut 1).

    Returns:
        np.ndarray: Signal temporel filtré (partie réelle de l'IFFT).
    """
    N = len(fft_y)

    if window_type == 'Hann':
        hann_window = np.hanning(N)
        hann_window = np.roll(hann_window, N // 2)
    elif window_type == 'cutoff':
        hann_window = np.ones(N)
    elif window_type == 'butterworth':
        # Calculate Butterworth weights for each frequency
        if filter_type == 'low':
            butterworth_weights = 1 / (1 + np.abs(frequencies / filter_frequency) ** (2 * butterworth_order))
        elif filter_type == 'high':
            butterworth_weights = 1 / (1 + (filter_frequency / np.abs(frequencies)) ** (2 * butterworth_order))
            butterworth_weights[np.abs(frequencies) == 0] = 0  # Avoid division by zero
        elif filter_type == 'band':
            if band_frequencies is None:
                raise ValueError("band_frequencies must be provided for band-pass filter.")
            low, high = band_frequencies
            # Combine low-pass and high-pass Butterworth filters
            low_pass_weights = 1 / (1 + np.abs(frequencies / high) ** (2 * butterworth_order))
            high_pass_weights = 1 / (1 + (low / np.abs(frequencies)) ** (2 * butterworth_order))
            high_pass_weights[np.abs(frequencies) == 0] = 0  # Avoid division by zero
            butterworth_weights = low_pass_weights * high_pass_weights
        else:
            raise ValueError("filter_type must be 'low', 'high', or 'band'")
        hann_window = butterworth_weights
    else:
        hann_window = np.ones(N)

    if filter_type == 'low':
        filter_mask = np.where(np.abs(frequencies) <= filter_frequency, hann_window, 0)
    elif filter_type == 'high':
        filter_mask = np.where(np.abs(frequencies) >= filter_frequency, hann_window, 0)
    elif filter_type == 'band':
        if band_frequencies is None:
            raise ValueError("band_frequencies must be provided for band-pass filter.")
        low, high = band_frequencies
        filter_mask = np.where(
            (np.abs(frequencies) >= low) & (np.abs(frequencies) <= high),
            hann_window,
            0
        )
    else:
        raise ValueError("filter_type must be 'low', 'high', or 'band'")

    filtered_fft = fft_y * filter_mask
    return np.real(np.fft.ifft(filtered_fft))

def apply_filter_to_paths(
    paths: list[str],
    filter_type: str = 'low',
    filter_frequency: float = None,
    band_frequencies: tuple[float, float] = None,
    window_type: str = 'Hann',
    temporal_filter_type: str = None,
    butterworth_order: int = 1
) -> list[np.ndarray]:
    """Applique un filtre (passe-bas, passe-haut ou passe-bande) à une liste de fichiers audio.

    Args:
        paths: Liste des chemins vers les fichiers audio à filtrer.
        filter_type: Type de filtre à appliquer ('low', 'high', ou 'band').
        filter_frequency: Fréquence de coupure pour les filtres passe-bas ou passe-haut (Hz).
        band_frequencies: Tuple de fréquences (low, high) pour le filtre passe-bande (Hz).
        window_type: Type de fenêtre pour le filtre : 'Hann', 'cutoff', ou 'butterworth'.
        temporal_filter_type: Type de fenêtre temporelle (optionnel).
        butterworth_order: Ordre du filtre Butterworth (par défaut 1).

    Returns:
        list[np.ndarray]: Liste des signaux audio filtrés.
    """
    if filter_type == 'band' and band_frequencies is None:
        raise ValueError("band_frequencies must be provided for band-pass filter.")

    return [
        apply_filter(
            *discret_FFT(path, windows=temporal_filter_type),
            filter_frequency=filter_frequency,
            band_frequencies=band_frequencies,
            filter_type=filter_type,
            window_type=window_type,
            butterworth_order=butterworth_order
        )
        for path in paths
    ]