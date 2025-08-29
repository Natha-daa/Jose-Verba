from pydub import AudioSegment
import re

def split_audio(file_path, segment_length=5*1000): 
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    start_time = 54*1000
    end_time = start_time + segment_length
    segment = audio[start_time:end_time]

    output_file = f"/home/stagiaire/verbalens/app/api/temp/thimothe.wav"
    segment.export(output_file, format="wav")
    print(f"Exported: {output_file}")


def clean_json_string(json_string):
    """
    Removes ```json and ``` wrapping from a JSON string.
    """
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()
