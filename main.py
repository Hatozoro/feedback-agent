import os
import json
import time
import random
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Scraper Imports
from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews as play_reviews

# 1. Setup
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={"response_mime_type": "application/json"}
    )
else:
    model = None

# ---------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------
APP_CONFIG = [
    {
        "name": "Nordkurier",
        "ios_id": "1250964862",
        "android_id": "de.nordkurier.live",
        "country": "de"
    },
    {
        "name": "Schwäbische",
        "ios_id": "432491155",
        "android_id": "de.schwaebische.epaper",
        "country": "de"
    }
]

# ---------------------------------------------------------
# SCRAPING (Mit Pausen gegen Blockaden)
# ---------------------------------------------------------
def fetch_ios_reviews(app_name, app_id, country="de", count=20):
    print(f"   -> iOS: {app_name}...")
    try:
        # Zufällige Pause, um wie ein Mensch zu wirken
        time.sleep(random.uniform(2, 5))
        app = AppStore(country=country, app_name=app_name, app_id=app_id)
        app.review(how_many=count)

        results = []
        for r in app.reviews:
            results.append({
                "store": "ios",
                "app": app_name,
                "rating": r['rating'],
                "text": r['review'],
                "date": r['date'].strftime("%Y-%m-%d")
            })
        return results
    except Exception as e:
        print(f"      iOS Fehler: {e}")
        return []

def fetch_android_reviews(app_name, app_id, country="de", count=20):
    print(f"   -> Android: {app_name}...")
    try:
        result, _ = play_reviews(
            app_id, lang='de', country=country, sort=Sort.NEWEST, count=count
        )
        cleaned = []
        for r in result:
            cleaned.append({
                "store": "android",
                "app": app_name,
                "rating": r['score'],
                "text": r['content'],
                "date": r['at'].strftime("%Y-%m-%d")
            })
        return cleaned
    except Exception as e:
        print(f"      Android Fehler: {e}")
        return []

def get_all_reviews():
    all_data = []
    print("--- Starte Scraping ---")
    for app in APP_CONFIG:
        if app.get("ios_id"):
            all_data.extend(fetch_ios_reviews(app["name"], app["ios_id"], app["country"]))

        if app.get("android_id"):
            all_data.extend(fetch_android_reviews(app["name"], app["android_id"], app["country"]))
    return all_data

# ---------------------------------------------------------
# HTML GENERATOR (Berechnet Zahlen selbst!)
# ---------------------------------------------------------
def generate_html(ki_data, reviews):
    # 1. Zahlen selbst berechnen (Viel genauer als KI)
    total_count = len(reviews)
    ios_count = len([r for r in reviews if r['store'] == 'ios'])
    android_count = len([r for r in reviews if r['store'] == 'android'])

    if total_count > 0:
        avg_rating = sum([r['rating'] for r in reviews]) / total_count
        avg_rating = round(avg_rating, 2)
    else:
        avg_rating = 0

    html = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>App Feedback Dashboard</title>
        <style>
            body {{ font-family: sans-serif; background: #f4f6f8; color: #333; padding: 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
            h1 {{ border-bottom: 2px solid #eee; padding-bottom: 15px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 30px 0; }}
            .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #ddd; }}
            .card h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #666; text-transform: uppercase; }}
            .val {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
            .summary {{ background: #e3f2fd; padding: 20px; border-radius: 8px; border-left: 5px solid #2196f3; line-height: 1.6; }}
            .topics {{ margin: 20px 0; }}
            .topic {{ display: inline-block; background: #e8f5e9; color: #2e7d32; padding: 5px 12px; border-radius: 15px; margin: 5px; border: 1px solid #c8e6c9; }}
            .review-box {{ border: 1px solid #eee; padding: 15px; margin-bottom: 10px; border-radius: 6px; }}
            .good {{ border-left: 5px solid #4caf50; }}
            .bad {{ border-left: 5px solid #f44336; }}
            footer {{ margin-top: 50px; text-align: center; font-size: 12px; color: #aaa; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 App Feedback Report</h1>
            <p style="color: #888;">Stand: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
            
            <div class="summary">
                <strong>🤖 KI Analyse:</strong><br>
                {ki_data.get('summary', 'Keine Analyse möglich.')}
            </div>

            <div class="grid">
                <div class="card"><h3>Gesamt Reviews</h3><div class="val">{total_count}</div></div>
                <div class="card"><h3>Ø Sterne</h3><div class="val">{avg_rating} ⭐</div></div>
                <div class="card"><h3>iOS</h3><div class="val">{ios_count}</div></div>
                <div class="card"><h3>Android</h3><div class="val">{android_count}</div></div>
            </div>

            <h3>🔥 Aktuelle Themen</h3>
            <div class="topics">
                {''.join([f'<span class="topic">{t}</span>' for t in ki_data.get('topics', [])])}
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h3>👍 Top Feedback</h3>
                    {''.join([f'<div class="review-box good"><small>{r.get("app")} ({r.get("store")}) {r.get("rating")}★</small><br>"{r.get("text")[:120]}..."</div>' for r in ki_data.get('topReviews', [])])}
                </div>
                <div>
                    <h3>⚠️ Kritisches Feedback</h3>
                    {''.join([f'<div class="review-box bad"><small>{r.get("app")} ({r.get("store")}) {r.get("rating")}★</small><br>"{r.get("text")[:120]}..."</div>' for r in ki_data.get('bottomReviews', [])])}
                </div>
            </div>
            
            <footer>Automated by GitHub Actions</footer>
        </div>
    </body>
    </html>
    """
    return html

# ---------------------------------------------------------
# HAUPTPROGRAMM
# ---------------------------------------------------------
def run_agent():
    # 1. Daten holen
    reviews = get_all_reviews()

    if not reviews:
        print("⚠️ Keine Daten gefunden. Nutze Fallback.")
        reviews = [{"store": "System", "app": "Bot", "rating": 0, "text": "Fehler beim Abruf."}]

    # 2. KI Analyse
    print("--- Starte KI Analyse ---")
    ki_output = {"summary": "Konnte keine Analyse erstellen.", "topics": [], "topReviews": [], "bottomReviews": []}

    if model:
        # Wir senden nur relevante Daten an die KI
        prompt_data = [{k: v for k, v in r.items() if k in ['text', 'rating', 'app', 'store']} for r in reviews]
        prompt = f"""
        You are a QA Analyst. Analyze these app reviews.
        Output strictly JSON:
        {{
            "summary": "Short management summary in German",
            "topics": ["Topic1", "Topic2"],
            "topReviews": [list of 3 best reviews objects],
            "bottomReviews": [list of 3 worst reviews objects]
        }}
        
        Reviews:
        {json.dumps(prompt_data[:100], ensure_ascii=False)}
        """
        try:
            response = model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "") # Clean markdown
            ki_output = json.loads(text)
        except Exception as e:
            print(f"KI Fehler: {e}")

    # 3. HTML bauen (Zahlen kommen jetzt aus Python!)
    html = generate_html(ki_output, reviews)

    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Fertig!")

if __name__ == "__main__":
    run_agent()