import os
import librosa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define paths
dataset_root = "dataset"
data_folder = os.path.join(dataset_root, "speech_commands_v0.02")
spectrogram_folder = os.path.join(dataset_root, "spectrograms")

# Configuration for classes (all digits 0-9)
class config:
    CLASSES = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    

def ensure_16000_samples(sound_wave, target_length=16000):
    """Ensure audio has exactly 16000 samples by truncating or zero-padding."""
    num_samples = len(sound_wave)
    if num_samples > target_length:
        return sound_wave[:target_length]
    elif num_samples < target_length:
        return np.pad(sound_wave, (0, target_length - num_samples), mode='constant')
    return sound_wave

def sound_wave_to_mel_spectrogram(sound_wave, sample_rate, spec_h=128, spec_w=128, length=1):
    """Generate mel spectrogram from a sound wave."""
    NUM_MELS = spec_h
    HOP_LENGTH = int(sample_rate * length / (spec_w - 1))
    mel_spec = librosa.feature.melspectrogram(y=sound_wave, sr=sample_rate, hop_length=HOP_LENGTH, n_mels=NUM_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def save_spectrograms(data_dir, spectrogram_dir, sample_rate=16000):
    """Generate and save mel spectrogram images for WAV files in data_dir."""
    class_set = set(config.CLASSES)
    
    for label_folder in os.listdir(data_dir):
        if label_folder not in class_set:
            print(f"Skipping folder {label_folder} (not in CLASSES)")
            continue

        label_folder_path = os.path.join(data_dir, label_folder)
        spectrogram_folder_path = os.path.join(spectrogram_dir, label_folder)
        os.makedirs(spectrogram_folder_path, exist_ok=True)

        processed_files = 0
        for audio_file in os.listdir(label_folder_path):
            if audio_file.endswith('.wav'):
                audio_file_path = os.path.join(label_folder_path, audio_file)
                y, sr = librosa.load(audio_file_path, sr=sample_rate)

                # Ensure each audio has 16000 samples (should already be done by ensure_1sec_wav.py)
                # y = ensure_16000_samples(y, target_length=16000)

                # Convert to Mel Spectrogram
                mel_spec = sound_wave_to_mel_spectrogram(y, sr)
                mel_spec = np.array(mel_spec)
                
                # Resize to 128x128
                # mel_spec_resize = np.resize(mel_spec, (128, 128))

                # print(mel_spec)

                # Diagnostic: Check mel spectrogram values
                if np.any(np.isnan(mel_spec)) or np.any(np.isinf(mel_spec)):
                    print(f"Warning: Invalid values in mel spectrogram for {audio_file_path}")
                    continue
                
                # print(f"Mel spectrogram shape for {audio_file}: {mel_spec.shape}, "
                    #   f"min: {np.min(mel_spec):.2f}, max: {np.max(mel_spec):.2f}"

                # Save as grayscale image using plt.imsave
                output_path = os.path.join(spectrogram_folder_path, f'{audio_file[:-4]}.png')
                plt.imsave(output_path, mel_spec, cmap='gray', format='png')
                
                # Verify grayscale and file size
                with Image.open(output_path) as img:
                    if img.mode != 'L':
                        # print(f"Warning: {output_path} saved as {img.mode}, converting to grayscale")
                        img = img.convert('L')
                        img.save(output_path)
                
                # file_size = os.path.getsize(output_path)
                # print(f"Saved spectrogram: {output_path} ({file_size} bytes)")
                processed_files += 1

        print(f"Processed {processed_files} files in {label_folder_path}")

def main():
    try:
        if not os.path.exists(data_folder):
            raise RuntimeError(f"Dataset folder {data_folder} not found. Run prepare_speech_commands.py and ensure_1sec_wav.py first.")

        # Create spectrograms directory
        os.makedirs(spectrogram_folder, exist_ok=True)

        # Generate spectrograms
        save_spectrograms(data_folder, spectrogram_folder, sample_rate=16000)

        print("Completed generating spectrograms in:", spectrogram_folder)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()