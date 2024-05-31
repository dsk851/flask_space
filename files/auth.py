import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def load_audio(file_path):
    # Charger le fichier audio
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


def extract_mfcc(audio, sr, n_mfcc=13):
    # Extraire les coefficients MFCC de l'audio
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs, mfccs_mean


def calculate_distance(mfcc1_mean, mfcc2_mean):
    # Calculer la distance euclidienne entre deux vecteurs MFCC moyens
    distance = np.linalg.norm(mfcc1_mean - mfcc2_mean)
    return distance


def plot_waveform_and_mfcc(
    ref_audio, ref_sr, suspect_audio, suspect_sr, ref_mfccs, suspect_mfccs
):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Forme d'onde de la référence
    axs[0, 0].set_title("Waveform - Référence")
    librosa.display.waveshow(ref_audio, sr=ref_sr, ax=axs[0, 0])

    # MFCC de la référence
    axs[0, 1].set_title("MFCC - Référence")
    img = librosa.display.specshow(ref_mfccs, sr=ref_sr, x_axis="time", ax=axs[0, 1])
    fig.colorbar(img, ax=axs[0, 1])

    # Forme d'onde du suspect
    axs[1, 0].set_title("Waveform - Suspect")
    librosa.display.waveshow(suspect_audio, sr=suspect_sr, ax=axs[1, 0])

    # MFCC du suspect
    axs[1, 1].set_title("MFCC - Suspect")
    img = librosa.display.specshow(
        suspect_mfccs, sr=suspect_sr, x_axis="time", ax=axs[1, 1]
    )
    fig.colorbar(img, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()


def main(reference_file, suspect_file):
    # Charger et extraire les caractéristiques du fichier de référence
    ref_audio, ref_sr = load_audio(reference_file)
    ref_mfccs, ref_mfcc_mean = extract_mfcc(ref_audio, ref_sr)

    # Charger et extraire les caractéristiques du fichier suspect
    suspect_audio, suspect_sr = load_audio(suspect_file)
    suspect_mfccs, suspect_mfcc_mean = extract_mfcc(suspect_audio, suspect_sr)

    # Calculer la distance entre les MFCC des deux fichiers
    distance = calculate_distance(ref_mfcc_mean, suspect_mfcc_mean)

    print(f"Distance entre la voix de référence et la voix suspecte : {distance}")

    # Définir un seuil arbitraire pour juger de la similarité (ceci peut nécessiter des ajustements)
    threshold = 50  # Ce seuil doit être ajusté selon les données spécifiques
    if distance < threshold:
        print("La voix suspecte est similaire à la voix de référence.")
        print(f"Vecteurs mfcc n1 {ref_mfcc_mean} n2 {suspect_mfcc_mean}")
    else:
        print("La voix suspecte est différente de la voix de référence.")

    # Afficher les formes d'onde et les MFCC sur une même figure
    plot_waveform_and_mfcc(
        ref_audio, ref_sr, suspect_audio, suspect_sr, ref_mfccs, suspect_mfccs
    )


# Remplacez par les chemins de vos fichiers audio
reference_file = "do3.wav"
suspect_file = "do4.wav"

# main(reference_file, suspect_file)
