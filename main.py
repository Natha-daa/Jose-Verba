# fichier : main.py
from fastapi import FastAPI, Query, HTTPException, Body, Form, status
from pydantic import BaseModel
import whisper
import requests
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Union
import uuid
from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from utils.diarization import extract_speakers, write_segments
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import json
import torchaudio
import math
import numpy as np
import re
import torch
from speechbrain.inference.speaker import EncoderClassifier
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Pipeline
from scipy.spatial.distance import cdist
from utils.utils import extract_speaker_embeddings, find_nearest_speaker, format_time
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from utils.prompt import generate_fact_check_prompt, generate_fact_extraction_prompt
from utils.splitter import clean_json_string
from langchain_tavily import TavilySearch
from sqlalchemy.orm import Session
import models
import schemas
from db import Base, engine, get_db
import uvicorn

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DB_FILE = "speakers_store/spearker_voice.db"  # Supprimé, remplacé par PostgreSQL

load_dotenv('.env.local')
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["SERPAPI_API_KEY"] = "17b4254cd1caa589a23ab4e13821a0784d17aa09450c3b94f1249ef3dd5313ad"
SERPER_API_KEY = "5c51947ee6b3e5fcf45e7ff19e4652f748168e70"
os.environ["SERPER_API_KEY"] = "5c51947ee6b3e5fcf45e7ff19e4652f748168e70"
os.environ["TAVILY_API_KEY"] = "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"
TAVILY_API_KEY = "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"
model_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key="AIzaSyAv5QViz6g3qDDiaX-jI3bzZ2HEf-7rXL0")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
).to(DEVICE)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

llm = whisper.load_model("turbo")  # ou "small", "medium", "large"

# Lazy loading des modèles pour optimiser la RAM au démarrage
def load_llm():
    return whisper.load_model("turbo")  # ou "small", "medium", "large"

def load_classifier():
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE}
    ).to(DEVICE)

def load_pipeline():
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )

def load_model_gemini():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key="AIzaSyAv5QViz6g3qDDiaX-jI3bzZ2HEf-7rXL0")

# Crée les tables si elles n'existent pas
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {"status": "API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}


# === ROUTE /media ===
@app.post("/media", response_model=schemas.MediaResponse)
async def create_media(
    name: str = Form(...),
    description: str = Form(None),
    numberSpeaker: int = Form(1),
    audio: UploadFile = None,
    video: UploadFile = None,
    db: Session = Depends(get_db)
):
    os.makedirs("/app/uploads", exist_ok=True)

    audio_link, video_link, file_size = None, None, 0

    # Sauvegarde audio
    if audio:
        audio_filename = f"{uuid.uuid4()}_{audio.filename}"
        audio_path = os.path.join("/app/uploads", audio_filename)
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        audio_link = f"/file/{audio_filename}"
        file_size = audio.size or 0

    # Sauvegarde vidéo
    if video:
        video_filename = f"{uuid.uuid4()}_{video.filename}"
        video_path = os.path.join("/app/uploads", video_filename)
        with open(video_path, "wb") as f:
            f.write(await video.read())
        video_link = f"/file/{video_filename}"
        file_size = video.size or 0

    # Création de l'objet Media et insertion dans PostgreSQL
    media_record = models.Media(
        name=name,
        description=description,
        numberSpeaker=numberSpeaker,
        audioLink=audio_link,
        videoLink=video_link,
        fileSize=str(file_size)
    )
    db.add(media_record)
    db.commit()
    db.refresh(media_record)

    return media_record

# === ROUTE pour récupérer un fichier uploadé ===
@app.get("/file/{file_name}")
def get_file(file_name: str):
    file_path = os.path.join("/app/uploads", file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path)

# === ROUTE GET /media pour lister tous les medias ===
@app.get("/media", response_model=list[schemas.MediaResponse])
def get_all_media(db: Session = Depends(get_db)):
    medias = db.query(models.Media).all()
    return medias    

class TranscriptionResponse(BaseModel):
    text: str

class DiarisationResponse(BaseModel):
    transcript: str

# ====== Modèles Pydantic =======

class SpeakerSegment(BaseModel):
    speaker: str
    start_time: str
    text: str

class Transcript(BaseModel):
    transcript: List[SpeakerSegment]

class TranscriptItem(BaseModel):
    speaker: str
    start_time: str
    text: str

class Transcript(BaseModel):
    transcript: List[TranscriptItem]

class FactCheckItem(BaseModel):
    statement: str
    verdict: str
    sources: List[str]
    confidence: float
    category: str
    start_time: str
    justification: str
    evidence: str
    speaker: str

class FactCheckRequest(BaseModel):
    transcript: List[TranscriptItem]

class IdeologyResult(BaseModel):
    speaker: str
    orientation: str
    justification: str
    confidence: float

class PsychProfile(BaseModel):
    speaker: str
    dominant_traits: List[str]
    tone: str
    intentions: List[str]
    confidence: float

class SummaryResult(BaseModel):
    global_summary: str
    by_speaker: Dict[str, str]
    
# AI agent tools
def serper_video_search(query: str) -> str:
    url = "https://google.serper.dev/videos"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    body = {"q": query}

    response = requests.post(url, headers=headers, data=json.dumps(body))
    data = response.json()

    if "videos" in data:
        results = []
        for vid in data["videos"]:
            title = vid.get("title", "")
            link = vid.get("link", "")
            snippet = vid.get("snippet", "")
            results.append(f"- Titre: {title}\n  Lien: {link}\n  Résumé: {snippet}")
        return "\n".join(results)
    else:
        return "Aucune vidéo pertinente trouvée."
    
# Initialize Tavily Search Tool
tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    include_answer=True
)

wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

def parse_single_psych_profile(output_text: str, speaker: str) -> PsychProfile:
    # Extraire Traits
    traits_match = re.search(r"Traits\s*:\s*(.*?)\n\s*Ton\s*:", output_text, re.DOTALL | re.IGNORECASE)
    dominant_traits = []
    if traits_match:
        traits_block = traits_match.group(1)
        dominant_traits = [line.strip(" *") for line in traits_block.strip().splitlines() if line.strip()]

    # Extraire Ton
    tone_match = re.search(r"Ton\s*:\s*\*\s*(.*?)\n\s*Intentions\s*:", output_text, re.DOTALL | re.IGNORECASE)
    tone = tone_match.group(1).strip() if tone_match else ""

    # Extraire Intentions
    intentions_match = re.search(r"Intentions\s*:\s*(.*?)\n\s*Confiance\s*:", output_text, re.DOTALL | re.IGNORECASE)
    intentions = []
    if intentions_match:
        intentions_block = intentions_match.group(1)
        intentions = [line.strip(" *") for line in intentions_block.strip().splitlines() if line.strip()]

    # Extraire Confiance
    confidence_match = re.search(r"Confiance\s*:\s*\*\s*(\d+)%", output_text, re.IGNORECASE)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.0

    return PsychProfile(
        speaker=speaker,
        dominant_traits=dominant_traits,
        tone=tone,
        intentions=intentions,
        confidence=confidence
    )

@app.post("/transcription/internal")
async def transcription(audio_id: str = Query(..., description="identifiant de l'audio hébergée")):
    try:
        audio_data = os.path.join("/app/uploads", audio_id)
        if not os.path.exists(audio_data):
            raise HTTPException(status_code=404, detail="File not found")
        else:
            print("audio found")
            # Transcription locale avec Whisper
            result = load_llm().transcribe(audio_data)
            return {"text": result["text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diarisation/internal")
async def diarisation(audio_id: str = Query(..., description="identifiant de l'audio hébergée"), num_speakers: int = Query(..., description="Nombre de locuteurs")):
    try:
        audio_data = os.path.join("/app/uploads", audio_id)
        if not os.path.exists(audio_data):
            raise HTTPException(status_code=404, detail="File not found")
        else:
            print("audio found")

            # Diarisation locale avec Whisper
            llm = load_llm()
            classifier = load_classifier()
            pipeline = load_pipeline()
            seg = extract_speakers(llm, classifier, audio_data, num_speakers)
            result = write_segments(seg, '/app/temp/transcript.txt')
            return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/uploads/")
async def create_upload_file(file: Union[UploadFile, None] = None):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No upload file sent")
        else:
            filename=str(uuid.uuid4())+"_"+file.filename
            file_path=os.path.join("/app/uploads", filename)
            print(file_path)
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            return {"message": "File saved successfully","filename":filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/file/{name_file}")
def get_file(name_file: str):
    try:
        if not os.path.exists(os.path.join("/app/uploads", name_file)):
            raise HTTPException(status_code=404, detail="File not found")
        else:
            return FileResponse(path=os.path.join("/app/uploads", name_file))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fact-checking")
async def fact_check_agent(transcript: Transcript):
    results = []
    for seg in transcript.transcript:
        prompt = generate_fact_extraction_prompt(seg.text, seg.speaker, seg.start_time)
        result = load_model_gemini().invoke(prompt)
        parse_result = clean_json_string(result.content)
        parse_result: List[Dict] = json.loads(parse_result)
        tavily_search_result = ""
        wikipedia_search_result = ""
        serper_video_search_result = ""
        for fact in parse_result:
            
            for query in fact["query_tavily"]:
                tems = tavily_search_tool.invoke({'query': query})
                tavily_search_result += json.dumps(tems)
            
            for query in fact["query_wikipedia"]:
                wikipedia_search_result += json.dumps(wikipedia_tool.run(query))
            
            for query in fact["query_serper_video"]:
                serper_video_search_result += serper_video_search(query)
            
        fact_check_prompt = generate_fact_check_prompt(fact["statement"], seg.speaker, seg.start_time, tavily_search_result, wikipedia_search_result, serper_video_search_result)
        print(fact_check_prompt)
        result = load_model_gemini().invoke(fact_check_prompt)
        print(result.content)
        parse_result = clean_json_string(result.content)
        parse_result = json.loads(parse_result)
        results.append(parse_result)

    return results

@app.post("/ideology", response_model=List[IdeologyResult])
async def ideology(transcript: Transcript):
    # Concatène le texte par intervenant
    speakers_text = {}
    for seg in transcript.transcript:
        speakers_text.setdefault(seg.speaker, "")
        speakers_text[seg.speaker] += seg.text + " "

    # Construit le bloc à analyser
    speakers_formatted = "\n\n".join(
        [f"{speaker}:\n{speakers_text[speaker]}" for speaker in speakers_text]
    )

    # Prompt journalistique expert pour l'idéologie
    prompt = ChatPromptTemplate.from_template("""
    Vous êtes un analyste politique expert.
    Votre mission est d'examiner attentivement les propos de chaque intervenant afin d'identifier son orientation idéologique ou politique dominante.

    **Consignes :**
    1️⃣ Pour chaque intervenant, indiquez son nom ou identifiant.
    2️⃣ Décrivez son **orientation idéologique** dominante (ex. : progressiste, conservateur, centriste, radical, libéral, socialiste…).
    3️⃣ Justifiez brièvement votre analyse à partir des propos tenus, sans extrapoler au-delà du contenu fourni.
    4️⃣ Attribuez un **niveau de confiance** en pourcentage (0-100) pour refléter la solidité de votre estimation.
    5️⃣ Rédigez en français formel, neutre, sans jugement de valeur.
    6️⃣ Chaque element de la réponse doit etre sur une ligne
    7️⃣ Sois strict, professionnel, précis et rigoureux et sans texte hors structure.
    

    **Format attendu :**
    Intervenant : [nom]
    Orientation : [orientation idéologique]
    Justification : [justification]
    Confiance : [pourcentage]

    **Exemples :**

    Exemple 1 :
    Intervenant : Mme Dupont
    Orientation : Progressiste
    Justification : Mme Dupont défend des mesures sociales inclusives, évoque la redistribution des richesses et insiste sur la solidarité nationale.
    Confiance : 85

    Exemple 2 :
    Intervenant : M. Martin
    Orientation : Conservateur
    Justification : M. Martin met l'accent sur la tradition, critique les réformes sociétales rapides et valorise la stabilité familiale.
    Confiance : 90


    **Texte à analyser :**
    {synthese}
    """)

    chain = prompt | load_model_gemini()

    result = chain.invoke({"synthese": speakers_formatted})

    # Traitement minimal du texte brut (prototype)
    lines = result.content.split("\n")
    results = []
    current = {}

    for line in lines:
        if line.strip().startswith("Intervenant"):
            if current:
                results.append(current)
            current = {}
            current['speaker'] = line.split(":")[1].strip()
        elif "Orientation" in line:
            current['orientation'] = line.split(":")[1].strip()
        elif "Justification" in line:
            current['justification'] = line.split(":")[1].strip()
        elif "Confiance" in line:
            conf = line.split(":")[1].strip().replace("%", "")
            current['confidence'] = float(conf)

    if current:
        results.append(current)

    return [IdeologyResult(**r) for r in results]

@app.post("/psych-profile", response_model=List[PsychProfile])
async def analyze_psych_profile(transcript: Transcript):
    # 🧠 2️⃣ Profilage psychologique
    speakers = {}
    for seg in transcript.transcript:
        speakers.setdefault(seg.speaker, []).append(seg.text)

    results = []
    for speaker, texts in speakers.items():
        full_text = " ".join(texts)

        prompt = f"""
        🔍 Vous êtes un psychologue et sociologue expert depuis plus de 50 ans, spécialisé en analyse conversationnelle.  
        Votre mission est d’établir un **profil psychologique précis** pour chaque intervenant, basé sur **leur discours** ET les résultats du **fact-checking**.
        ---

        **👉 EXEMPLES DE FORMAT À RESPECTER ABSOLUMENT :**

        EXEMPLE 1
        Traits :
            * Curieux (pose plusieurs questions ouvertes)
            * Précis (relance sur des détails chiffrés)
            * Observateur (note les nuances dans le discours)
        Ton :
            * Neutre (questions factuelles sans jugement)
        Intentions :
            * Comprendre (cherche des précisions)
            * Vérifier (confirme des affirmations)
        Confiance :
            * 80%
        ---

        EXEMPLE 2
        Traits :
            * Ambitieux (évoque des objectifs élevés)
            * Stratège (met en avant des plans à long terme)
        Ton :
            * Affirmatif (discours assuré)
        Intentions :
            * Convaincre (veut rallier à sa vision)
            * Inspirer (donne envie de suivre son plan)
        Confiance :
            * 95%

        ---

        EXEMPLE 3
        Traits :
            * Sceptique (pose des questions critiques)
            * Analytique (souligne les contradictions)
        Ton :
            * Direct (discours sans détour)
        Intentions :
            * Déstabiliser (met en difficulté)
            * Tester (vérifie la solidité du discours)
        Confiance :
            * 75%

        ---

        **👉 TEXTE À ANALYSER :**

        \"\"\"{full_text}\"\"\"

        ---

        **🎯 Tâche :**
        - Identifiez 2-5 traits psychologiques clés (chaque trait doit être justifié).
        - Identifiez le ton principal (justification).
        - Identifiez 3-5 intentions dominantes (justification).
        - Donnez un pourcentage de confiance global (justification implicite).

        ---

        **📌 Rappel importants:**  
        - Recherchers les informations sur la veracites des affirmations sur internet.
        - Respectez **scrupuleusement le format EXACT des EXEMPLES.**  
        - Ne changez jamais l'ordre : Traits ➜ Ton ➜ Intentions ➜ Confiance.
        - Une seule analyse par intervenant.
        - Ne mettez aucun texte hors structure.
        - Sois précis, concis et rigoureux.

        """
        result = load_model_gemini().invoke(prompt)

        profile = parse_single_psych_profile(result.content,speaker)
        results.append(profile)

    return results

@app.post("/summarize", response_model=SummaryResult)
def summarize(transcript: Transcript):
    all_text = "\n\n".join(
        [f"[{seg.speaker}] {seg.text}" for seg in transcript.transcript]
    )
    # 2️⃣ Prépare le prompt LangChain
    prompt = ChatPromptTemplate.from_template("""
    Vous êtes un rédacteur et analyste professionnel spécialisé dans le journalisme de synthèse.  
    Votre mission est de produire une synthèse claire, objective et hiérarchisée d’un dialogue, débat ou interview, tout en respectant la parole de chaque intervenant.  
    Votre style doit être neutre, élégant, et factuel, comme dans un compte-rendu destiné à la presse ou à une revue sérieuse.

    Consignes :
    Le resume consiste a dire ce qui est dit dans le dialogue en le reformulant de facon synthétique et concise.
    1️⃣ Rédigez d'abord un résumé global detaille du contenu, en présentant le contexte, le ton général et les principaux points abordés, sans jugement et dire ce qui est dit par rapport a ces points.
    2️⃣ Pour chaque intervenant, rédigez ensuite un **résumé individuel**, en indiquant son nom ou son identifiant et en reformulant ce qu'il dit.  
    3️⃣ Lorsque vous parlez de chaque intervenant, vouvoyez-le systématiquement, en rappelant ses propos avec precision et de facon detaille en utilisant le vous.  
    4️⃣ Votre style doit être fluide, structuré, rédigé en français formel, sans familiarité excessive.
    5️⃣ N'inventez pas de contenu : reformulez uniquement ce qui figure explicitement dans le dialogue fourni.
    6️⃣ Le resume doit etre humain et factuel et doit etre structuré.
    8️⃣ Tu dois egalement respecte le format de reponse exactement et repondre chaque ca sur une ligne (GLOBAL, SPEAKER 1,SPEAKER 2,...) comme ce dernier.


    Format :
    GLOBAL: ...
    SPEAKER:Nom : Résumé

    Exemple :
    GLOBAL: la dialogue parle en general de ...
    Thimote: Il soutinent le point selon lequel ...
    David: Selon David ...


    Dialogue à analyser :
    {dialogue}

    """)



    # 3️⃣ Génère
    print(prompt)
    chain = prompt | load_model_gemini()

    result = chain.invoke({"dialogue": all_text})
    print(result.content)

    # 4️⃣ Parse (simplifié)
    lines = result.content.split("\n")
    global_summary = None
    by_speaker = {}
    for line in lines:
        print(f"start {line} end")
        if line.startswith("GLOBAL"):
            global_summary = line.replace("GLOBAL:", "").strip()
        else:
            parts = line.strip().split(":")
            if len(parts) >= 2:
                speaker = parts[0].strip()
                summary = ":".join(parts[1:]).strip()
                by_speaker[speaker] = summary

    return SummaryResult(global_summary=global_summary, by_speaker=by_speaker)

@app.post("/add-speaker-voice", response_model=Dict[str, Union[str, int]])
async def add_speaker(speaker_name: str = Form(...), file: UploadFile = None, db: Session = Depends(get_db)):
    # Sauver temporairement le fichier audio
    tmp_path = f"/app/temp/voice/{speaker_name}.wav"
    
    if not os.path.exists(tmp_path):
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    
    # Extraire l'embedding
    embedding = extract_speaker_embeddings(tmp_path, DEVICE, load_classifier())
    embedding_json = json.dumps(embedding.tolist())

    # Ajout dans PostgreSQL
    speaker = models.Speaker(speaker_name=speaker_name, embedding=embedding_json)
    db.add(speaker)
    db.commit()
    db.refresh(speaker)

    return {"message": f"Speaker '{speaker_name}' ajouté avec succès.", "embedding_dim": len(embedding)}

@app.get("/all-speakers-voice")
async def list_speakers(db: Session = Depends(get_db)):
    speakers = db.query(models.Speaker).all()
    return [{"id": speaker.id, "speaker_name": speaker.speaker_name} for speaker in speakers]

@app.post("/identify-speaker", response_model=Dict[str, Union[str, float]])
async def identify_speaker_sqlite(file: UploadFile, db: Session = Depends(get_db)):
    tmp_path = f"/app/temp/{file.filename}.wav"
    if not os.path.exists(tmp_path):
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    
    with open(tmp_path, "wb") as f:
        f.write(await file.read())  
    
    new_emb = extract_speaker_embeddings(tmp_path, DEVICE, load_classifier())
    
    speakers = db.query(models.Speaker).all()
    known_speakers = [(speaker.speaker_name, np.array(json.loads(speaker.embedding))) for speaker in speakers]
    best_match = find_nearest_speaker(new_emb, known_speakers)

    if best_match:
        return {"speaker": best_match["speaker"], "similarity": best_match["similarity"]}
    else:
        return {"speaker": None, "similarity": 0.0}

@app.post("/asr")
async def asr(file: UploadFile, db: Session = Depends(get_db)):
    tmp_path = f"/app/temp/{file.filename}.wav"
    if not os.path.exists(tmp_path):
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    
    with open(tmp_path, "wb") as f:
        f.write(await file.read()) 
    
    transcript = load_llm().transcribe(tmp_path)
    segments = load_pipeline()(tmp_path)
    
    # Set a threshold for similarity scores to determine when a match is considered successful
    threshold = 0.7
    sample_rate=44100
    speaker_segments = []
    aligned_output = []

    # Iterate through each segment identified in the diarization process
    for segment, label, confidence in segments.itertracks(yield_label=True):
        start_time, end_time = segment.start, segment.end

        # Load the specific audio segment from the meeting recording
        waveform, sample_rate = torchaudio.load(tmp_path, num_frames=int((end_time-start_time)*sample_rate), frame_offset=int(start_time*sample_rate))
        waveform = waveform.to(DEVICE)

        # Extract the speaker embedding from the audio segment
        embedding = load_classifier().encode_batch(waveform).squeeze(1).cpu().numpy()/np.linalg.norm(load_classifier().encode_batch(waveform).squeeze(1).cpu().numpy())

        # Initialize variables to find the recognized speaker
        min_distance = float('inf')
        recognized_speaker_id = None

        speakers = db.query(models.Speaker).all()
        known_speaker_ids = [speaker.speaker_name for speaker in speakers]
        known_speakers = [np.array(json.loads(speaker.embedding)) for speaker in speakers]

        # Compare the segment's embedding to each known speaker's embedding using cosine distance
        for i, speaker_embedding in enumerate(known_speakers):
            distances = cdist(embedding.tolist(), speaker_embedding, metric="cosine")
            min_distance_candidate = distances.min()
            if min_distance_candidate < min_distance:
                min_distance = min_distance_candidate
                recognized_speaker_id = known_speaker_ids[i]

        # Output the identified speaker and the time range they were speaking, if a match is found
        if min_distance < threshold and end_time-start_time>1:
            speaker_segments.append({
                "speaker": recognized_speaker_id,
                "start_time": math.floor(float(start_time)),
                "end_time": math.floor(float(end_time)),
            })
        elif end_time-start_time > 1:
            speaker_segments.append({
                "speaker": "unknown",
                "start_time": start_time,
                "end_time": end_time,
            })
    
    for segment in transcript['segments']:
        start = segment['start']
        text = segment['text'].strip()
        start = math.floor(float(start))
        speak = None
        for seg in speaker_segments:
            if start >= seg["start_time"] and start < seg["end_time"]:
                speak = seg["speaker"]
                break
        if speak is None:
            speak = "unknown"
        timestamp = format_time(start)
        aligned_output.append({
            "speaker": speak,
            "timestamp": timestamp,
            "start": start,
            "end": segment['end'],
            "text": text
        })

    return aligned_output

# Démarrage de l'application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
