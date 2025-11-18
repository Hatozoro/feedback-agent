import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Lade den Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Fehler: Kein API Key in .env gefunden!")
else:
    genai.configure(api_key=api_key)

    print(f"Prüfe Modelle für Key: {api_key[:5]}...")

    try:
        # 2. Liste alle Modelle auf, die dein Key sehen kann
        print("\n--- VERFÜGBARE MODELLE ---")
        available_models = []
        for m in genai.list_models():
            # Wir suchen nur Modelle, die Text generieren können
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                available_models.append(m.name)

        if not available_models:
            print("Keine Modelle gefunden. Prüfe deine API-Key Berechtigungen im Google AI Studio.")

    except Exception as e:
        print(f"\nEin Fehler ist aufgetreten: {e}")