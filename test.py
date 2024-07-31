import os
import librosa
import soundfile as sf
import numpy as np


def process_wav_file(input_path, output_path, target_sr=44100, target_duration=5):
    y, sr = librosa.load(input_path, sr=None)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    target_length = target_duration * target_sr
    if len(y) > target_length:
        y = y[:target_length]
    else:
        repeats = target_length // len(y) + 1
        y = np.tile(y, repeats)[:target_length]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y, target_sr)
    print(output_path)

def process_folder(input_folder, output_folder, target_sr=44100, target_duration=5):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                process_wav_file(input_path, output_path, target_sr, target_duration)


if __name__ == "__main__":
    input_folder = "data/gtzan/audio"
    output_folder = "data/gtzan/changed"
    process_folder(input_folder, output_folder)
