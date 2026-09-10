import os
import numpy as np
import librosa
from src.audio.filter import apply_filter_to_paths  # Replace with the actual import path

# Specify the folder path
folder_path = 'data/raw_recordings/fsc22/ESC-50-master/ESC-50-master/audio/'

# Get all file names in the folder
file_names = os.listdir(folder_path)
audio_paths = [os.path.join(folder_path, file) for file in file_names if file.endswith('.wav')][:100]

# Load the original signals for comparison
original_signals = [librosa.load(path, sr=None)[0] for path in audio_paths]

# Test 1: Low-pass filter
print("Testing low-pass filter...")
filtered_signals_low = apply_filter_to_paths(
    paths=audio_paths,
    filter_type='low',
    filter_frequency=1000,
    window_type='Hann'
)
assert len(filtered_signals_low) == len(audio_paths), "Number of filtered signals does not match input."
for i, (original, filtered) in enumerate(zip(original_signals, filtered_signals_low)):
    assert not np.array_equal(original, filtered), f"Signal {i}: Low-pass filter did not modify the signal."
    assert not np.any(np.isnan(filtered)), f"Signal {i}: Low-pass filter produced NaN values."
    assert not np.any(np.isinf(filtered)), f"Signal {i}: Low-pass filter produced inf values."
print("Low-pass filter test passed!")

# Test 2: High-pass filter
print("Testing high-pass filter...")
filtered_signals_high = apply_filter_to_paths(
    paths=audio_paths,
    filter_type='high',
    filter_frequency=1000,
    window_type='Hann'
)
assert len(filtered_signals_high) == len(audio_paths), "Number of filtered signals does not match input."
for i, (original, filtered) in enumerate(zip(original_signals, filtered_signals_high)):
    assert not np.array_equal(original, filtered), f"Signal {i}: High-pass filter did not modify the signal."
    assert not np.any(np.isnan(filtered)), f"Signal {i}: High-pass filter produced NaN values."
    assert not np.any(np.isinf(filtered)), f"Signal {i}: High-pass filter produced inf values."
print("High-pass filter test passed!")

# Test 3: Band-pass filter
print("Testing band-pass filter...")
filtered_signals_band = apply_filter_to_paths(
    paths=audio_paths,
    filter_type='band',
    band_frequencies=(1000, 5000),
    window_type='Hann'
)
assert len(filtered_signals_band) == len(audio_paths), "Number of filtered signals does not match input."
for i, (original, filtered) in enumerate(zip(original_signals, filtered_signals_band)):
    assert not np.array_equal(original, filtered), f"Signal {i}: Band-pass filter did not modify the signal."
    assert not np.any(np.isnan(filtered)), f"Signal {i}: Band-pass filter produced NaN values."
    assert not np.any(np.isinf(filtered)), f"Signal {i}: Band-pass filter produced inf values."
print("Band-pass filter test passed!")

# Test 4: Butterworth low-pass filter
print("Testing Butterworth low-pass filter...")
filtered_signals_butterworth_low = apply_filter_to_paths(
    paths=audio_paths,
    filter_type='low',
    filter_frequency=1000,
    window_type='butterworth',
    butterworth_order=2
)
assert len(filtered_signals_butterworth_low) == len(audio_paths), "Number of filtered signals does not match input."
for i, (original, filtered) in enumerate(zip(original_signals, filtered_signals_butterworth_low)):
    assert not np.array_equal(original, filtered), f"Signal {i}: Butterworth low-pass filter did not modify the signal."
    assert not np.any(np.isnan(filtered)), f"Signal {i}: Butterworth low-pass filter produced NaN values."
    assert not np.any(np.isinf(filtered)), f"Signal {i}: Butterworth low-pass filter produced inf values."
print("Butterworth low-pass filter test passed!")

# Test 5: Butterworth high-pass filter
print("Testing Butterworth high-pass filter...")
filtered_signals_butterworth_high = apply_filter_to_paths(
    paths=audio_paths,
    filter_type='high',
    filter_frequency=1000,
    window_type='butterworth',
    butterworth_order=2
)
assert len(filtered_signals_butterworth_high) == len(audio_paths), "Number of filtered signals does not match input."
for i, (original, filtered) in enumerate(zip(original_signals, filtered_signals_butterworth_high)):
    assert not np.array_equal(original, filtered), f"Signal {i}: Butterworth high-pass filter did not modify the signal."
    assert not np.any(np.isnan(filtered)), f"Signal {i}: Butterworth high-pass filter produced NaN values."
    assert not np.any(np.isinf(filtered)), f"Signal {i}: Butterworth high-pass filter produced inf values."
print("Butterworth high-pass filter test passed!")

# Test 6: Butterworth band-pass filter
print("Testing Butterworth band-pass filter...")
filtered_signals_butterworth_band = apply_filter_to_paths(
    paths=audio_paths,
    filter_type='band',
    band_frequencies=(1000, 5000),
    window_type='butterworth',
    butterworth_order=2
)
assert len(filtered_signals_butterworth_band) == len(audio_paths), "Number of filtered signals does not match input."
for i, (original, filtered) in enumerate(zip(original_signals, filtered_signals_butterworth_band)):
    assert not np.array_equal(original, filtered), f"Signal {i}: Butterworth band-pass filter did not modify the signal."
    assert not np.any(np.isnan(filtered)), f"Signal {i}: Butterworth band-pass filter produced NaN values."
    assert not np.any(np.isinf(filtered)), f"Signal {i}: Butterworth band-pass filter produced inf values."
print("Butterworth band-pass filter test passed!")

print("All tests passed successfully!")