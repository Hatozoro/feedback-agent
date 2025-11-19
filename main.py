import os
import json
import hashlib
import requests
from requests.exceptions import HTTPError
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Scraper Imports
from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews as play_reviews

# ---------------------------------------------------------
# 1. SETUP & KONFIGURATION
# ---------------------------------------------------------
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

# Konfiguration der Datenhaltung und Apps
DATA_FILE = "data/reviews_history.json"
APP_CONFIG = [
    {"name": "Nordkurier", "ios_id": "1250964862", "android_id": "de.nordkurier.live", "country": "de"},
    {"name": "Schwäbische", "ios_id": "432491155", "android_id": "de.schwaebische.epaper", "country": "de"}
]

# ---------------------------------------------------------
# 2. HILFSFUNKTIONEN FÜR DATENHALTUNG
# ---------------------------------------------------------
def generate_id(review):
    """Erstellt eine eindeutige ID für jedes Review basierend auf Text & Datum"""
    unique_str = f"{review['app']}{review['store']}{review['date']}{review['text'][:50]}"
    return hashlib.md5(unique_str.encode()).hexdigest()

def load_history():
    """Lädt die persistente Historie-Datei."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("WARNUNG: History-Datei korrupt, starte mit leerer Historie.")
            return []
    return []

def save_history(data):
    """Speichert die Historie-Datei, die von GitHub persistiert wird."""
    os.makedirs("data", exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------
# 3. SCRAPING
# ---------------------------------------------------------
# NEUE, STABILE SCRAPING FUNKTION FÜR iOS (über RSS-API)
# ---------------------------------------------------------
def fetch_ios_reviews(app_name, app_id, country="de", count=20):
    print(f"   -> iOS (RSS): {app_name}...")

    # Apple RSS API Endpoint (holt die neuesten Reviews)
    # Wir begrenzen auf 100, da die API keine dynamische Mengenbegrenzung erlaubt.
    # Count wird später im Code gefiltert.
    api_url = f"https://itunes.apple.com/de/rss/customerreviews/id={app_id}/sortBy=mostrecent/json"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status() # Prüft auf HTTP Fehler (4xx, 5xx)
        data = response.json()

        results = []
        # Die Apple RSS API hat eine verschachtelte Struktur: data['feed']['entry']
        for entry in data.get('feed', {}).get('entry', [])[:count]:
            # Ignoriere Einträge, die nur Metadaten sind (ohne 'content')
            if 'im:rating' not in entry or 'content' not in entry:
                continue

            # Daten aus der verschachtelten Struktur extrahieren
            rating = int(entry['im:rating']['label'])
            text = entry['content']['label']
            date_str = entry['updated']['label'][:10] # Datum extrahieren

            results.append({
                "store": "ios",
                "app": app_name,
                "rating": rating,
                "text": text,
                "date": date_str,
                "id": generate_id({'app': app_name, 'store': 'ios', 'date': date_str, 'text': text})
            })

        print(f"      ✅ iOS (RSS) fand {len(results)} Reviews.")
        return results
    except Exception as e:
        print(f"      ❌ iOS (RSS) Fehler: {e}")
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
                "store": "android", "app": app_name, "rating": r['score'],
                "text": r['content'], "date": r['at'].strftime("%Y-%m-%d")
            })
        return cleaned
    except Exception as e:
        print(f"      ❌ Android Fehler: {e}")
        return []

def get_fresh_reviews():
    """Holt neue Reviews und filtert Duplikate heraus"""
    all_scraped = []
    for app in APP_CONFIG:
        if app.get("ios_id"): all_scraped.extend(fetch_ios_reviews(app["name"], app["ios_id"], app["country"]))
        if app.get("android_id"): all_scraped.extend(fetch_android_reviews(app["name"], app["android_id"], app["country"]))

    # Duplikate checken
    history = load_history()
    existing_ids = {r['id'] for r in history if 'id' in r}

    new_reviews = []
    for r in all_scraped:
        r_id = generate_id(r)
        r['id'] = r_id
        if r_id not in existing_ids:
            new_reviews.append(r)

    print(f"--- Scraping fertig: {len(all_scraped)} geladen, davon {len(new_reviews)} NEU ---")

    # Neue Reviews kommen ganz oben in die Historie
    updated_history = new_reviews + history
    return updated_history[:1000], new_reviews

# ---------------------------------------------------------
# 4. TEAMS NOTIFICATION (MVP) - FIX gegen 400 Error
# ---------------------------------------------------------
def send_teams_notification(new_reviews, webhook_url):
    """Sendet eine Nachricht an den Teams/Power Automate Webhook."""
    if not new_reviews:
        print("Keine neuen Reviews, keine Teams Benachrichtigung notwendig.")
        return

    positive_count = sum(1 for r in new_reviews if r['rating'] >= 4)
    negative_count = sum(1 for r in new_reviews if r['rating'] <= 2)

    title = f"📢 NEUES FEEDBACK! ({len(new_reviews)} Reviews)"

    # Liste der neuen Texte
    review_list_text = ""
    for r in new_reviews[:5]: # Zeige maximal 5 Reviews an
        rating_star = "⭐" * r['rating']
        review_list_text += f"* **{r['app']} ({r['store']})**: {rating_star} *\"{r['text'][:60]}...\"*\n"

    # NEU: Erzeuge eine einfache Textnachricht für Power Automate/Teams
    full_text_message = f"""
📢 **{title}**
---
👍 Positiv: {positive_count} | 🚨 Kritisch: {negative_count}

**Neueste Nutzerstimmen (Auszug):**
{review_list_text}

[Zum vollständigen Dashboard](https://Hatozoro.github.io/feedback-agent/)
"""

    # Power Automate ist wählerisch und will oft nur einen einfachen Text-Body.
    teams_message = {
        "text": full_text_message
    }

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(webhook_url, json=teams_message, headers=headers, timeout=10)

        response.raise_for_status()
        print("✅ Teams Benachrichtigung erfolgreich gesendet.")
    except HTTPError as e:
        print(f"❌ HTTP Fehler beim Senden der Teams-Nachricht: {response.status_code} Client Error: {response.reason}. Prüfe den Power Automate Flow.")
    except Exception as e:
        print(f"❌ Allgemeiner Fehler beim Senden der Teams-Nachricht: {e}")

# ---------------------------------------------------------
# 5. HTML & ANALYSE
# ---------------------------------------------------------
def run_analysis_and_generate_html(full_history, new_only):
    """Erstellt die KI-Analyse und generiert das statische HTML-Dashboard."""

    analysis_set = full_history[:50]
    ki_output = {"summary": "Keine ausreichende Datenbasis für KI-Analyse.", "topics": [], "topReviews": [], "bottomReviews": []}

    if model and analysis_set:
        print(f"--- Starte KI-Analyse für {len(analysis_set)} Reviews ---")
        prompt_data = [{k: v for k, v in r.items() if k in ['text', 'rating', 'app', 'store']} for r in analysis_set]
        prompt = f"""
        Analyze these reviews (max 50). Summarize the key user complaints and product areas that need attention. Output strictly in JSON format.
        
        Output Structure:
        {{
            "summary": "Management summary in German focusing on recent trends and main issues.",
            "topics": ["Login", "Absturz", "Performance", "Neue Artikel"],
            "topReviews": [list of 3 positive review texts],
            "bottomReviews": [list of 3 negative review texts]
        }}
        
        Review Data: {json.dumps(prompt_data, ensure_ascii=False)}
        """
        try:
            response = model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            ki_output = json.loads(text)
        except Exception as e:
            print(f"❌ KI Fehler bei JSON-Verarbeitung: {e}")

    # HTML Generierung
    total_count = len(full_history)
    ios_count = len([r for r in full_history if r['store'] == 'ios'])
    android_count = len([r for r in full_history if r['store'] == 'android'])
    avg_rating = 0
    if total_count > 0:
        avg_rating = round(sum(r['rating'] for r in full_history) / total_count, 2)

    top_reviews = ki_output.get('topReviews', [])[:3]
    bottom_reviews = ki_output.get('bottomReviews', [])[:3]

    html = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>App Feedback Dashboard</title>
        <style>
            body {{ font-family: sans-serif; background: #f4f6f8; padding: 20px; color: #333; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
            h1 {{ border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
            .card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ddd; margin: 10px; flex: 1; }}
            .val {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
            .flex {{ display: flex; gap: 20px; flex-wrap: wrap; }}
            .summary {{ background: #e3f2fd; padding: 20px; border-radius: 8px; border-left: 5px solid #2196f3; margin: 20px 0; }}
            .topic {{ background: #e8f5e9; color: #2e7d32; padding: 5px 12px; border-radius: 15px; margin: 3px; display: inline-block; border: 1px solid #c8e6c9; font-size: 0.9em; }}
            .review-list {{ display: flex; gap: 20px; margin-top: 20px; }}
            .review-column {{ flex: 1; min-width: 40%; }}
            .review-item {{ border: 1px solid #eee; padding: 10px; border-radius: 5px; margin-bottom: 10px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }}
            .good {{ border-left: 5px solid #28a745; }}
            .bad {{ border-left: 5px solid #dc3545; }}
            .review-text {{ font-style: italic; font-size: 0.9em; margin-top: 5px; }}
            .metadata {{ font-size: 0.75em; color: #6c757d; margin-top: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 App Feedback Dashboard</h1>
            <p class="metadata">Datenbasis: **{total_count}** Reviews | Stand: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
            
            <div class="summary"><strong>KI Fazit & aktuelle Trends:</strong><br>{ki_output.get('summary')}</div>
            
            <div class="flex">
                <div class="card"><h3>Ø Gesamt (History)</h3><div class="val">{avg_rating} ⭐</div></div>
                <div class="card"><h3>Neue Reviews (Heute)</h3><div class="val">{len(new_only)}</div></div>
                <div class="card"><h3>iOS Reviews</h3><div class="val">{ios_count}</div></div>
                <div class="card"><h3>Android Reviews</h3><div class="val">{android_count}</div></div>
            </div>

            <h3>🔥 Themen Cluster (Semantische Analyse)</h3>
            <div>{''.join([f'<span class="topic">{t}</span>' for t in ki_output.get('topics', [])])}</div>
            
            <h2>⭐ Top- & Low-Reviews (KI-Auswahl)</h2>
            <div class="review-list">
                <div class="review-column">
                    <h4>Top 3 Reviews (Positiv)</h4>
                    {''.join([f'<div class="review-item good"><div class="metadata">KI Highlight</div><div class="review-text">"{r}"</div></div>' for r in top_reviews])}
                </div>
                <div class="review-column">
                    <h4>Low 3 Reviews (Kritisch)</h4>
                    {''.join([f'<div class="review-item bad"><div class="metadata">KI Fokus</div><div class="review-text">"{r}"</div></div>' for r in bottom_reviews])}
                </div>
            </div>
            
            <h2>📝 Neueste Rohdaten</h2>
            <div>
                {''.join([f'<div class="review-item"><div class="metadata"><b>{r["app"]} ({r["store"]}) - {r["rating"]}★</b> | {r["date"]}</div><div class="review-text">"{r["text"][:250]}..."</div></div>' for r in full_history[:10]])}
            </div>
        </div>
    </body>
    </html>
    """

    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html)

# ---------------------------------------------------------
# 6. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Daten holen & Duplikate filtern
    full_history, new_reviews = get_fresh_reviews()

    # 2. Speichern (Persistenz)
    save_history(full_history)

    # 3. Dashboard bauen (enthält KI-Analyse)
    run_analysis_and_generate_html(full_history, new_reviews)

    # 4. Chat-Benachrichtigung senden (MVP)
    teams_webhook = os.getenv("TEAMS_WEBHOOK_URL")
    if teams_webhook:
        send_teams_notification(new_reviews, teams_webhook)
    else:
        print("⚠️ TEAMS_WEBHOOK_URL Secret nicht gefunden. Keine Benachrichtigung gesendet.")

    print("✅ Durchlauf beendet. Ready for Commit.")