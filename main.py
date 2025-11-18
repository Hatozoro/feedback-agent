import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime

# 1. Setup
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Fallback für lokale Tests ohne Key (optional)
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config={"response_mime_type": "application/json"})
else:
    model = None

embedder = SentenceTransformer('all-MiniLM-L6-v2')

SYSTEM_INSTRUCTION = """
System Instruction: You are an autonomous feedback-analysis agent.
Output Requirements: Return JSON with summary, metrics (totalReviews, byStore, avgRating), topics, topReviews, bottomReviews.
"""

# Dummy Daten (Hier später Scraper einfügen)
incoming_reviews = [
    {"store": "ios", "app": "nordkurier", "rating": 1, "text": "Login geht nicht mehr.", "timestamp": "2024-05-22"},
    {"store": "android", "app": "schwaebische", "rating": 5, "text": "Super Update, läuft flüssig.", "timestamp": "2024-05-22"},
    {"store": "ios", "app": "schwaebische", "rating": 3, "text": "Ganz okay, aber Werbung nervt.", "timestamp": "2024-05-22"}
]

def generate_html(data):
    # Einfaches HTML/CSS Dashboard Template
    html = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>App Feedback Dashboard</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; background: #f4f4f9; color: #333; padding: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .metric-box {{ display: flex; gap: 20px; margin: 20px 0; }}
            .card {{ flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ddd; }}
            .card h3 {{ margin: 0; font-size: 14px; color: #666; }}
            .card p {{ margin: 5px 0 0; font-size: 24px; font-weight: bold; color: #007bff; }}
            .summary {{ background: #eef2f7; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin-bottom: 20px; line-height: 1.6; }}
            .reviews {{ margin-top: 30px; }}
            .review-item {{ background: #fff; border-bottom: 1px solid #eee; padding: 10px 0; }}
            .badge-good {{ background: #d4edda; color: #155724; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
            .badge-bad {{ background: #f8d7da; color: #721c24; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
            footer {{ margin-top: 40px; font-size: 12px; color: #999; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Tägliches App Feedback</h1>
            <p style="color: #666;">Generiert am {datetime.now().strftime('%d.%m.%Y um %H:%M Uhr')}</p>
            
            <div class="summary">
                <strong>KI Zusammenfassung:</strong><br>
                {data.get('summary', 'Keine Zusammenfassung verfügbar.')}
            </div>

            <div class="metric-box">
                <div class="card">
                    <h3>Total Reviews</h3>
                    <p>{data.get('metrics', {}).get('totalReviews', 0)}</p>
                </div>
                <div class="card">
                    <h3>Ø Rating</h3>
                    <p>{data.get('metrics', {}).get('avgRating', 0)} ⭐</p>
                </div>
                <div class="card">
                    <h3>iOS / Android</h3>
                    <p>{data.get('metrics', {}).get('byStore', {}).get('ios', 0)} / {data.get('metrics', {}).get('byStore', {}).get('android', 0)}</p>
                </div>
            </div>

            <div class="reviews">
                <h3>🔥 Top Themen</h3>
                <ul>
                    {''.join([f'<li>{t}</li>' for t in data.get('topics', [])])}
                </ul>
            </div>
            
            <footer>Automated by Gemini 2.0 & GitHub Actions</footer>
        </div>
    </body>
    </html>
    """
    return html

def run_agent():
    print("--- Starte Analyse ---")

    # 1. Analyse durchführen
    if model:
        prompt = f"{SYSTEM_INSTRUCTION}\nAnalyze these reviews: {json.dumps(incoming_reviews)}"
        response = model.generate_content(prompt)
        data = json.loads(response.text)
    else:
        # Fallback falls kein Key da ist (für lokale Tests ohne .env)
        data = {"summary": "Lokaler Test ohne KI", "metrics": {"totalReviews": 3, "avgRating": 3.0, "byStore": {"ios": 2, "android": 1}}, "topics": ["Test"]}

    # 2. HTML Dashboard generieren
    dashboard_html = generate_html(data)

    # 3. HTML speichern (wird später von GitHub Action veröffentlicht)
    # Wir speichern es im 'public' ordner, damit es sauber ist
    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    print("Dashboard generiert: public/index.html")

if __name__ == "__main__":
    run_agent()