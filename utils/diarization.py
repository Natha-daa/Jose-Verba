import subprocess
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
import numpy as np
import datetime
from  utils.convert_transcript_to_json import parse_transcript


def extract_speakers(model, embedding_model, path, num_speakers=2):
    """Do diarization with speaker names"""
    mono = 'temp/mono.wav'
    cmd = 'ffmpeg -i {} -y -ac 1 temp/mono.wav'.format(path)
    subprocess.check_output(cmd, shell=True)
    result = model.transcribe(mono)
    segments = result["segments"]
    
    with contextlib.closing(wave.open(mono,'r')) as f:
      frames = f.getnframes()
      rate = f.getframerate()
      duration = frames / float(rate)
        
    audio = Audio()
    def segment_embedding(segment):
        start = segment["start"]
        # Whisper overshoots the end timestamp in the last segment
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(mono, clip)
        return embedding_model(waveform[None])

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
      embeddings[i] = segment_embedding(segment)
    embeddings = np.nan_to_num(embeddings)
    
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
      segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
    return segments    

def write_segments(segments, outfile):
    """write out segments to file"""
    
    def time(secs):
      return datetime.timedelta(seconds=round(secs))
    
    f = open(outfile, "w")  
    for (i, segment) in enumerate(segments):
      if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
        f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
      f.write(segment["text"][1:] + ' ')
    f.close()
    result = parse_transcript(outfile)
    return result

