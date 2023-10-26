import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def create_and_save_spectrogram(audio_file, image_save_path=None, data_save_path=None):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)  # sr=None preserves the original sampling rate
    
    # Create a mel-scaled spectrogram
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    
    # Convert the mel-spectrogram to log scale
    log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    # Save the visual spectrogram if requested
    if image_save_path:
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(log_mel_spect, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-scaled Spectrogram')
        plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Spectrogram image saved as {image_save_path}")

    # Save the log mel-spectrogram data to .npy format if requested
    if data_save_path:
        np.save(data_save_path, log_mel_spect)
        print(f"Spectrogram data saved as {data_save_path}")

# Prompt the user for the audio file path
audio_file = input("Please enter the path to your audio file: ")

# Extract the original file name without the extension
base_name = os.path.basename(audio_file).rsplit('.', 1)[0]

# Ask the user for their choice
print("What would you like to generate?")
print("1: Spectrogram image only")
print("2: Spectrogram data only")
print("3: Both image and data")
choice = input("Enter your choice (1/2/3): ")

# Decide the save paths based on user choice
image_save_path = None
data_save_path = None
if choice == "1":
    image_save_path = f'{base_name}_spectrogram.png'
elif choice == "2":
    data_save_path = f'{base_name}_spectrogram.npy'
elif choice == "3":
    image_save_path = f'{base_name}_spectrogram.png'
    data_save_path = f'{base_name}_spectrogram.npy'
else:
    print("Invalid choice!")
    exit()

# Create the spectrogram based on the user's choice
create_and_save_spectrogram(audio_file, image_save_path, data_save_path)
