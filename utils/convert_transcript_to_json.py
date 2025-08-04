import json
import re

def parse_transcript(file_path):
    transcript = []
    current_speaker = None
    current_start_time = None
    current_text = ""

    # Regex pour détecter les lignes de type : SPEAKER X 0:00:00
    pattern = re.compile(r"^(SPEAKER \d+)\s+(\d+:\d+:\d+)$")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue  # ignorer les lignes vides

        match = pattern.match(line)
        if match:
            # Sauvegarder le bloc précédent s'il existe
            if current_speaker and current_text:
                transcript.append({
                    "speaker": current_speaker,
                    "start_time": current_start_time,
                    "text": current_text.strip()
                })
                current_text = ""

            # Nouveau bloc
            current_speaker = match.group(1)
            current_start_time = match.group(2)
        else:
            # Ajouter la ligne de texte au bloc courant
            if current_text:
                current_text += " " + line
            else:
                current_text = line

    # Ajouter le dernier bloc
    if current_speaker and current_text:
        transcript.append({
            "speaker": current_speaker,
            "start_time": current_start_time,
            "text": current_text.strip()
        })

    return {"transcript": transcript}