from pathlib import Path

import torch
from pydub import AudioSegment
from tqdm import tqdm
from transformers import pipeline

input_path = Path().home() / "Downloads" / "Audios-20250104T174645Z-001" / "Audios"
output_path = input_path / "output_texts"


def main():
    """
    Simple script that iterates over a directory with audio files in mp4 format and transcribes them using
    open-ai's Whisper model
    """
    output_path.mkdir(exist_ok=True)
    # convert m4a files to mp3 format
    target_files_to_convert = [file_m4a for file_m4a in input_path.iterdir()
                               if file_m4a.is_file() and file_m4a.suffix == ".m4a"]

    for file_m4a in tqdm(target_files_to_convert):
        target_file = file_m4a.parent / f"{file_m4a.stem}.mp3"
        if target_file.is_file():
            continue
        song = AudioSegment.from_file(file_m4a, format="m4a")
        song.export(target_file, format="mp3")

    targets_mp3 = [file_mp3 for file_mp3 in input_path.iterdir()
                   if file_mp3.is_file() and file_mp3.suffix == ".mp3"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = pipeline("automatic-speech-recognition",
                    model="openai/whisper-base",
                    chunk_length_s=30,
                    stride_length_s=(4, 2),
                    device=device)

    for t_mp3 in tqdm(targets_mp3):
        # we could provide all paths at once, but for the moment we keep it this way to ease the handling of out paths
        output_file = output_path / f"{t_mp3.stem}.txt"
        prediction = pipe(t_mp3.__str__(), batch_size=8, return_timestamps=False)
        with open(output_file, "w") as f:
            f.write(prediction["text"])


if __name__ == "__main__":
    main()
