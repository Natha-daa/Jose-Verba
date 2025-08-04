def generate_fact_check_prompt(statement: str, speaker: str, start_time: str, tavily_result: str, wikipedia_result: str, serper_video_result: str) -> str:
    """
    Génère le prompt complet pour le fact-checking,
    parfaitement formaté pour contraindre le LLM à produire un JSON strict.
    """
    prompt = f"""
u es un agent expert en fact-checking, spécialisé pour vérifier des affirmations issues de transcriptions vidéo ou audio.
Tu dois analyser l'affirmation fournie en t'appuyant UNIQUEMENT sur les résultats ci-dessous.

=== Résultats de Tavily Search ===
{tavily_result}

=== Résultats de Wikipedia ===
{wikipedia_result}

=== Résultats de Serper Video ===
{serper_video_result}

Ta tâche :
- Vérifier la véracité de l'affirmation.
- Déterminer un verdict clair : True, False ou Unverified.
- Citer les sources précises (URL ou titre Wikipedia).
- Évaluer ton niveau de confiance (entre 0.0 et 1.0).
- Identifier une catégorie générale (Politique, Santé, Sciences, Culture, etc.)
- Expliquer de façon détaillée ton raisonnement (justification).
- Résumer les preuves clés (evidence).
- Toujours reprendre le champ speaker fourni.
- Toujours inclure le champ start_time fourni.

⚠️ Ta réponse doit être strictement au format JSON, sans aucun texte autour.
⚠️ Ne réponds jamais autre chose que ce JSON.
⚠️ Si l’affirmation ne peut être vérifiée, indique "verdict": "Unverified" et explique pourquoi dans justification.

Voici le format exact de reponse que tu dois renvoyer :
{{
  "statement": "Le texte original à vérifier.",
  "verdict": "True | False | Unverified",
  "sources": ["liste des URLs ou titres Wikipedia"],
  "confidence": 0.0,
  "category": "catégorie générale",
  "justification": "ton raisonnement détaillé",
  "evidence": "preuve(s) ou extrait(s) clé(s)",
  "speaker": "nom du locuteur",
  "start_time": "horodatage de début"
}}

EXEMPLES DE REPONSE :
EXEMPLE 1
{{
  "statement": "Albert Einstein a reçu le prix Nobel de physique en 1921.",
  "verdict": "True",
  "sources": [
    "https://fr.wikipedia.org/wiki/Albert_Einstein",
    "https://www.nobelprize.org/prizes/physics/1921/einstein/biographical/"
  ],
  "confidence": 0.98,
  "category": "Sciences",
  "justification": "Des sources fiables telles que Wikipedia et le site officiel des prix Nobel confirment qu'Albert Einstein a bien reçu le prix Nobel de physique en 1921 pour sa découverte de l'effet photoélectrique.",
  "evidence": "Biographie sur le site Nobel et l'article Wikipedia.",
  "speaker": "John Doe",
  "start_time": "00:01:23"
}}

EXEMPLE 2
{{
  "statement": "Le président du Cameroun a démissionné hier.",
  "verdict": "Unverified",
  "sources": [],
  "confidence": 0.2,
  "category": "Politique",
  "justification": "Aucune source crédible trouvée sur Tavily, Wikipedia ou Serper Video ne confirme ou infirme cette affirmation. Aucun article récent ni déclaration officielle ne mentionne une telle démission.",
  "evidence": "Absence de preuves ou de communiqués officiels.",
  "speaker": "Marie",
  "start_time": "00:15:47"
}}


Affirmation à vérifier :
\"{statement}\"

Speaker :
\"{speaker}\"

Start time :
\"{start_time}\"

🔒 Réponds uniquement au format JSON. Ne produis aucune autre forme de texte.
    """.strip()

    return prompt


def generate_fact_extraction_prompt(full_text: str, speaker: str, start_time: str) -> str:
    """
    Génère le prompt pour extraire les affirmations factuelles vérifiables
    et produire pour chacune des requêtes adaptées à Tavily, Wikipedia et Serper Video.
    """
    prompt = f"""
Vous êtes un agent assistant spécialisé dans l’analyse de contenu pour le fact-checking.
Votre mission est d’extraire uniquement les **affirmations factuelles vérifiables** présentes dans le texte fourni.

Pour chaque affirmation extraite, générez également 6 requêtes de recherche optimisées afin de vérifier la véracité de l'affirmation :
1) Deux requêtes Tavily Search adaptées pour trouver des articles, publications ou rapports pertinents.
2) Deux requêtes Wikipedia pour localiser une ou plusieurs pages pertinentes sur le sujet.
3) Deux requêtes Serper Video pour vérifier l’existence de sources audiovisuelles pertinentes.

Présentez votre réponse au format **JSON strict**, sous forme de liste d’objets, chacun décrivant une affirmation et ses trois requêtes.
N’ajoutez aucun texte explicatif en dehors du JSON.

Format attendu :
[
  {{
    "statement": "affirmation factuelle extraite",
    "query_tavily": ["requête optimisée pour Tavily", "requete optimise pour tavily search"],
    "query_wikipedia": ["requête optimisée pour Wikipedia", "requete optimise pour wikipedia"],
    "query_serper_video": ["requête optimisée pour Serper Video", "requete optimise pour serper video"],
    "speaker": "nom du locuteur",
    "start_time": "horodatage de début"
  }},
  ...
]

Texte à analyser :
\"\"\"{full_text}\"\"\"

Speaker :
\"{speaker}\"

Start time :
\"{start_time}\"

⚠️ Votre réponse doit être exclusivement au format JSON et ne contenir aucune explication supplémentaire.
    """.strip()

    return prompt
