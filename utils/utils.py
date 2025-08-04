import torch
import torchaudio
import sqlite3
import numpy as np
from scipy.spatial.distance import cdist
import json
import datetime


def extract_speaker_embeddings(speaker_file, device,classifier):
    waveform, sample_rate = torchaudio.load(speaker_file)
    waveform = waveform.to(device)
    embedding = classifier.encode_batch(waveform)
    norm_emb = embedding.squeeze(1).cpu().numpy()
    return norm_emb / np.linalg.norm(norm_emb)

def find_nearest_speaker(new_embedding, DB_FILE):
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, speaker_name, embedding FROM speakers")
        rows = cur.fetchall()

    min_dist = 1
    best_match = None

    for row in rows:
        speaker_id, speaker_name, emb_json = row

        speaker_emb = np.array(json.loads(emb_json))

        # Cosine distance avec cdist
        distances = cdist(new_embedding.tolist(), speaker_emb, metric="cosine")

        dist = distances.min()

        if dist < min_dist:
            min_dist = dist
            best_match = {
                "speaker": speaker_name,
                "similarity": 1.0 - dist
            }

    if best_match:
        return best_match
    else:
        return {"speaker": None, "similarity": 0.0}

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))[2:]
