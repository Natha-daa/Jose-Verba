import torchaudio
import torch
import re
import os

def split_audio(file_path, segment_length=5.0):
    """
    Découpe un segment audio à partir de start_time = 54 secondes.
    segment_length en secondes (float)
    """
    # Charger l'audio
    waveform, sample_rate = torchaudio.load(file_path)  # waveform: Tensor [channels, samples]

    # Définir les indices de début et fin en échantillons
    start_time_sec = 54.0
    start_sample = int(start_time_sec * sample_rate)
    end_sample = int((start_time_sec + segment_length) * sample_rate)

    # S'assurer de ne pas dépasser la longueur de l'audio
    end_sample = min(end_sample, waveform.shape[1])

    segment = waveform[:, start_sample:end_sample]

    # Définir le chemin de sortie
    output_dir = "/home/stagiaire/verbalens/app/api/temp"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "thimothe.wav")

    # Sauvegarder le segment
    torchaudio.save(output_file, segment, sample_rate)
    print(f"Exported: {output_file}")

def clean_json_string(json_string):
    """
    Removes ```json and ``` wrapping from a JSON string.
    """
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()
