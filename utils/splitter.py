import torchaudio
import re
import os

def split_audio(file_path: str, start_time: float = 54.0, segment_length: float = 5.0, output_path: str = "temp/thimothe.wav"):
    """
    Extrait un segment audio à partir d'un fichier audio.
    
    :param file_path: chemin du fichier audio source
    :param start_time: temps de début du segment en secondes (par défaut 54s)
    :param segment_length: durée du segment en secondes (par défaut 5s)
    :param output_path: chemin de sortie du segment
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier audio introuvable : {file_path}")

    # Charger l'audio
    waveform, sample_rate = torchaudio.load(file_path)

    # Calcul des indices de début et de fin
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + segment_length) * sample_rate)

    # Extraire le segment
    segment = waveform[:, start_sample:end_sample]

    # Créer le dossier de sortie si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sauvegarder le segment
    torchaudio.save(output_path, segment, sample_rate)
    print(f"Exported: {output_path}")


def clean_json_string(json_string: str) -> str:
    """
    Supprime les balises ```json et ``` autour d'une chaîne JSON.
    
    :param json_string: chaîne JSON brute
    :return: chaîne JSON nettoyée
    """
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()
