import os
import torchaudio

# Define paths
dataset_root = "dataset"
data_folder = os.path.join(dataset_root, "speech_commands_v0.02")
resampled_folder = os.path.join(dataset_root, "4kh_speech_commands_v0.02")
target_sample_rate = 4000

def resample_wav(file_path, output_path, target_sr):
    """Resample WAV file to target sample rate."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    torchaudio.save(output_path, waveform, target_sr)
    # print(f"Resampled {file_path} to {output_path} ({target_sr} Hz)")

def main():
    try:
        if not os.path.exists(data_folder):
            raise RuntimeError(f"Dataset folder {data_folder} not found. Run prepare_speech_commands.py first.")

        # Create resampled dataset directory
        os.makedirs(resampled_folder, exist_ok=True)

        # Process each digit folder
        for digit in map(str, range(10)):
            digit_folder = os.path.join(data_folder, digit)
            output_digit_folder = os.path.join(resampled_folder, digit)
            os.makedirs(output_digit_folder, exist_ok=True)

            if not os.path.exists(digit_folder):
                print(f"Digit folder {digit_folder} not found, skipping.")
                continue

            print(f"Resampling files in: {digit_folder}")
            for file_name in os.listdir(digit_folder):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(digit_folder, file_name)
                    output_path = os.path.join(output_digit_folder, file_name)
                    resample_wav(file_path, output_path, target_sample_rate)

        print("Completed resampling in:", resampled_folder)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()