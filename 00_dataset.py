import os
import shutil
import tarfile
import argparse
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

# Parse command line arguments
parser = argparse.ArgumentParser(description="Prepare Google Speech Commands digits dataset")
parser.add_argument(
    "--keep-gz",
    action="store_false",
    default=True,
    help="Set to False to delete the downloaded .tar.gz file after processing (default is True, keeps the file)"
)
args = parser.parse_args()
keep_gz = args.keep_gz

# Mapping from word folders to digit folders
word_to_digit = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9"
}

# Root directory for dataset (relative to script in main folder)
dataset_root = "dataset"
data_folder = os.path.join(dataset_root, "speech_commands_v0.02")
os.makedirs(dataset_root, exist_ok=True)

def validate_and_move_audio_files(src_folder, dest_folder):
    """Validate WAV files using torchaudio and move them to the destination."""
    os.makedirs(dest_folder, exist_ok=True)
    valid_files = 0
    invalid_files = 0

    for file_name in os.listdir(src_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(src_folder, file_name)
            try:
                # Validate audio file with torchaudio
                waveform, sample_rate = torchaudio.load(file_path)
                if waveform.shape[0] > 0 and sample_rate > 0:  # Basic validation
                    shutil.move(file_path, os.path.join(dest_folder, file_name))
                    valid_files += 1
                else:
                    print(f"Invalid audio file: {file_path}")
                    invalid_files += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                invalid_files += 1
        else:
            print(f"Skipping non-WAV file: {file_name}")
            invalid_files += 1

    print(f"Moved {valid_files} valid audio files to {dest_folder}, skipped {invalid_files} invalid/non-WAV files")
    return valid_files > 0

def main():
    try:
        # Check if digit folders already exist in data_folder
        digit_folders_exist = all(os.path.exists(os.path.join(data_folder, d)) for d in word_to_digit.values())
        if digit_folders_exist:
            print("Digits dataset already prepared under:", data_folder)
            print("Final folders:", sorted(os.listdir(data_folder)))
            return

        # Check for existing .tar.gz file in dataset_root
        tar_files = [f for f in os.listdir(dataset_root) if f.endswith(".tar.gz")]
        if tar_files:
            print(f"Found existing archive: {tar_files[0]}")
            tar_path = os.path.join(dataset_root, tar_files[0])
        else:
            # Download the dataset using torchaudio
            class SubsetSC(SPEECHCOMMANDS):
                def __init__(self, root, download=True):
                    super().__init__(root, download=download)

            print("Downloading Speech Commands dataset...")
            dataset = SubsetSC(dataset_root, download=True)
            tar_files = [f for f in os.listdir(dataset_root) if f.endswith(".tar.gz")]
            if not tar_files:
                raise RuntimeError("Failed to download the .tar.gz file!")
            tar_path = os.path.join(dataset_root, tar_files[0])

        # Extract the archive
        print(f"Extracting {tar_path} to {dataset_root}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dataset_root)
        print("Extraction complete")

        # Check if speech_commands_v0.02 exists; if not, move extracted contents there
        if not os.path.exists(data_folder):
            extracted_folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
            temp_folder = None
            for folder in extracted_folders:
                if folder.startswith("speech_commands"):
                    temp_folder = os.path.join(dataset_root, folder)
                    break
            if temp_folder:
                os.rename(temp_folder, data_folder)
                print(f"Renamed {temp_folder} to {data_folder}")
            else:
                # If no enclosing folder, move contents to speech_commands_v0.02
                os.makedirs(data_folder, exist_ok=True)
                for item in os.listdir(dataset_root):
                    item_path = os.path.join(dataset_root, item)
                    if item != 'speech_commands_v0.02' and not item.endswith('.tar.gz'):
                        shutil.move(item_path, os.path.join(data_folder, item))
                print(f"Moved extracted contents to {data_folder}")

        # Move and rename digit folders, validate audio files
        for folder in os.listdir(data_folder):
            folder_path = os.path.join(data_folder, folder)
            if os.path.isdir(folder_path) and folder in word_to_digit:
                new_folder_path = os.path.join(data_folder, word_to_digit[folder])
                if validate_and_move_audio_files(folder_path, new_folder_path):
                    print(f"Processed and renamed {folder_path} to {new_folder_path}")
                    # Remove the original word folder after moving files
                    shutil.rmtree(folder_path)
                else:
                    print(f"No valid audio files in {folder_path}, removing {new_folder_path}")
                    if os.path.exists(new_folder_path):
                        shutil.rmtree(new_folder_path)

        # Remove non-digit folders and non-WAV files in data_folder
        for item in os.listdir(data_folder):
            item_path = os.path.join(data_folder, item)
            if os.path.isdir(item_path) and item not in word_to_digit.values():
                shutil.rmtree(item_path)
                print(f"Deleted non-digit folder: {item_path}")
            elif os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Deleted file: {item_path}")

        # Delete the .tar.gz file if keep_gz is False
        if not keep_gz and os.path.exists(tar_path):
            os.remove(tar_path)
            print(f"Deleted {tar_path}")

        print("Digits dataset prepared under:", data_folder)
        print("Final folders:", sorted(os.listdir(data_folder)))

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()