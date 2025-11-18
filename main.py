import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 1. Lade Umgebungsvariablen (Lokal aus .env, bei GitHub aus Secrets)
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Kein API Key gefunden! Bitte GEMINI_API_KEY setzen.")

genai.configure(api_key=API_KEY)

# 2. Modell Konfiguration
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config={"response_mime_type": "application/json"}
)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 3. System Prompt
SYSTEM_INSTRUCTION = """
System Instruction / Role: You are an autonomous feedback-analysis agent for “Nordkurier App” and “Schwäbische App”.
Output Requirements: Return JSON with summary, metrics, topics, topReviews, bottomReviews.
"""

# 4. Dummy Daten (Später hier Scraper einfügen)
incoming_reviews = [
    {"store": "ios", "app": "nordkurier", "rating": 1, "text": "Login geht nicht mehr seit dem Update.", "timestamp": "2024-05-22"},
    {"store": "android", "app": "schwaebische", "rating": 5, "text": "Tolle Nachrichten App, weiter so!", "timestamp": "2024-05-22"}
]

def run_agent():
    print("--- Starte Feedback Agent ---")

    # MVP Analyse
    prompt = f"{SYSTEM_INSTRUCTION}\nAnalyze these reviews: {json.dumps(incoming_reviews)}"
    response = model.generate_content(prompt)
    data = json.loads(response.text)

    # Semantische Suche Vorbereitung (Beispiel)
    texts = [r['text'] for r in incoming_reviews if r['text']]
    if texts:
        embeddings = embedder.encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        print(f"Vektordatenbank erstellt mit {len(texts)} Einträgen.")

    # Output anzeigen (Später: Speichern oder per Mail senden)
    print(json.dumps(data, indent=2))
    print("--- Fertig ---")

if __name__ == "__main__":
    run_agent()