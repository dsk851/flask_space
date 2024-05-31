import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def load_audio(file_path):
    
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


def extract_mfcc(audio, sr, n_mfcc=13):
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs, mfccs_mean


def calculate_distance(mfcc1_mean, mfcc2_mean):
    
    distance = np.linalg.norm(mfcc1_mean - mfcc2_mean)
    return distance


def plot_waveform_and_mfcc(
    ref_audio, ref_sr, suspect_audio, suspect_sr, ref_mfccs, suspect_mfccs
):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    
    axs[0, 0].set_title("Waveform - Référence")
    librosa.display.waveshow(ref_audio, sr=ref_sr, ax=axs[0, 0])

    
    axs[0, 1].set_title("MFCC - Référence")
    img = librosa.display.specshow(ref_mfccs, sr=ref_sr, x_axis="time", ax=axs[0, 1])
    fig.colorbar(img, ax=axs[0, 1])

    
    axs[1, 0].set_title("Waveform - Suspect")
    librosa.display.waveshow(suspect_audio, sr=suspect_sr, ax=axs[1, 0])

    
    axs[1, 1].set_title("MFCC - Suspect")
    img = librosa.display.specshow(
        suspect_mfccs, sr=suspect_sr, x_axis="time", ax=axs[1, 1]
    )
    fig.colorbar(img, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()


def main(reference_file, suspect_file):
    
    ref_audio, ref_sr = load_audio(reference_file)
    ref_mfccs, ref_mfcc_mean = extract_mfcc(ref_audio, ref_sr)

    
    suspect_audio, suspect_sr = load_audio(suspect_file)
    suspect_mfccs, suspect_mfcc_mean = extract_mfcc(suspect_audio, suspect_sr)

    
    distance = calculate_distance(ref_mfcc_mean, suspect_mfcc_mean)

    print(f"Distance entre la voix de référence et la voix suspecte : {distance}")

    
    threshold = 50  
    if distance < threshold:
        print("La voix suspecte est similaire à la voix de référence.")
    else:
        print("La voix suspecte est différente de la voix de référence.")

    
    plot_waveform_and_mfcc(
        ref_audio, ref_sr, suspect_audio, suspect_sr, ref_mfccs, suspect_mfccs
    )


reference_file = "reference.wav"
suspect_file = "suspect.wav"

main(reference_file, suspect_file)
