import sounddevice as sd
import soundfile as sf

def enregistrer_audio(filename, duration, samplerate=44100):
    print("Enregistrement en cours...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2, dtype='float64')
    sd.wait()  # Attendre la fin de l'enregistrement
    sf.write(filename, audio, samplerate)  # Enregistrer le fichier audio
    print(f"Enregistrement terminé. Audio enregistré dans '{filename}'.")

if __name__ == "__main__":
    filename = "do4.wav"
    duration = 5 # Durée de l'enregistrement en secondes
    enregistrer_audio(filename, duration)