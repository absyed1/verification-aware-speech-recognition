import os
import torchaudio
import torch

# Define paths
dataset_root = "dataset"
data_folder = os.path.join(dataset_root, "speech_commands_v0.02")
target_sample_rate = 16000
target_samples = target_sample_rate  # 1 second at 16 kHz

def ensure_1sec_wav(file_path, output_path):
    """Ensure WAV file is exactly 1 second (16,000 samples at 16 kHz) by truncating or zero-padding."""
    waveform, sample_rate = torchaudio.load(file_path)
    actions = []

    # Resample to 16 kHz if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        actions.append(f"resampled from {sample_rate} Hz to {target_sample_rate} Hz")
    
    # Get current number of samples
    num_samples = waveform.shape[1]
    
    # Truncate or pad
    if num_samples > target_samples:
        waveform = waveform[:, :target_samples]
        actions.append(f"truncated from {num_samples} to {target_samples} samples")
    elif num_samples < target_samples:
        padding = torch.zeros(waveform.shape[0], target_samples - num_samples)
        waveform = torch.cat([waveform, padding], dim=1)
        actions.append(f"padded from {num_samples} to {target_samples} samples")
    else:
        actions.append("unchanged")
    
    # Save the modified WAV file
    torchaudio.save(output_path, waveform, target_sample_rate)
    
    # Print only if resampled, truncated, or padded
    if any(action != "unchanged" for action in actions):
        print(f"Processed {file_path} -> {output_path}: {', '.join(a for a in actions if a != 'unchanged')}")
    
    return actions

def main():
    try:
        if not os.path.exists(data_folder):
            raise RuntimeError(f"Dataset folder {data_folder} not found. Run prepare_speech_commands.py first.")

        # Process each digit folder
        for digit in map(str, range(10)):
            digit_folder = os.path.join(data_folder, digit)
            if not os.path.exists(digit_folder):
                print(f"Digit folder {digit_folder} not found, skipping.")
                continue

            print(f"Processing digit folder: {digit_folder}")
            processed_files = 0
            actions_count = {"resampled": 0, "truncated": 0, "padded": 0, "unchanged": 0}

            for file_name in os.listdir(digit_folder):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(digit_folder, file_name)
                    output_path = file_path  # Overwrite original file
                    actions = ensure_1sec_wav(file_path, output_path)
                    processed_files += 1
                    if "resampled" in actions:
                        actions_count["resampled"] += 1
                    if "truncated" in actions:
                        actions_count["truncated"] += 1
                    elif "padded" in actions:
                        actions_count["padded"] += 1
                    else:
                        actions_count["unchanged"] += 1

            print(f"Processed {processed_files} files in {digit_folder}: "
                  f"{actions_count['resampled']} resampled, {actions_count['truncated']} truncated, "
                  f"{actions_count['padded']} padded, {actions_count['unchanged']} unchanged")

        print("Completed ensuring 1-second WAV files in:", data_folder)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()