import numpy as np
import librosa
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt

# Charger l'audio
file_path = 'do4.wav'
audio, sr = librosa.load(file_path, sr=None)

# Sélectionner une empreinte de bruit (par exemple, les premières 0.5 secondes)
noise_sample = audio[:int(0.5 * sr)]

# Calculer le spectre du bruit
noise_stft = librosa.stft(noise_sample)
noise_magnitude, noise_phase = librosa.magphase(noise_stft)
noise_magnitude_mean = np.mean(noise_magnitude, axis=1)

# Appliquer la réduction du bruit
audio_stft = librosa.stft(audio)
audio_magnitude, audio_phase = librosa.magphase(audio_stft)

# Soustraire le spectre du bruit du spectre de l'audio
audio_magnitude_cleaned = np.maximum(audio_magnitude - noise_magnitude_mean[:, np.newaxis], 0)

# Reconstruire le signal audio
audio_stft_cleaned = audio_magnitude_cleaned * audio_phase
audio_cleaned = librosa.istft(audio_stft_cleaned)

# Sauvegarder l'audio nettoyé
sf.write('clean'+file_path, audio_cleaned, sr)

# Affichage des spectrogrammes avant et après
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(audio_stft), ref=np.max), sr=sr, y_axis='log', x_axis='time')
plt.title('Spectrogramme Original')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(audio_stft_cleaned), ref=np.max), sr=sr, y_axis='log', x_axis='time')
plt.title('Spectrogramme Nettoyé')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
