# App Feedback Agent - Finaler Live Test
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Import der Scraper
from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews as play_reviews

# 1. Setup & Konfiguration
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Modell-Konfiguration
if API_KEY:
    genai.configure(api_key=API_KEY)
    # Wir nutzen Gemini 2.0 Flash für Geschwindigkeit und Stabilität
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={"response_mime_type": "application/json"}
    )
else:
    model = None
    print("WARNUNG: Kein API Key gefunden. KI-Analyse wird übersprungen.")

# System-Instruktion für den Agenten
SYSTEM_INSTRUCTION = """
System Instruction: You are an autonomous feedback-analysis agent for 'Nordkurier' and 'Schwäbische'.
Input: A list of recent user reviews from Apple App Store and Google Play Store.
Output Requirements: Return a valid JSON object with the following structure:
{
  "summary": "A concise management summary of the current sentiment and technical state (in German).",
  "metrics": {
    "avgRating": number,
    "byStore": {"ios": number, "android": number}
  },
  "topics": ["Topic 1", "Topic 2", ...],
  "topReviews": [{"text": "...", "rating": 5, "app": "...", "store": "..."}],
  "bottomReviews": [{"text": "...", "rating": 1, "app": "...", "store": "..."}]
}
Keep the summary professional and actionable.
"""

# ---------------------------------------------------------
# KONFIGURATION DER APPS (DEINE RICHTIGEN IDs)
# ---------------------------------------------------------
APP_CONFIG = [
    {
        "name": "Nordkurier",
        "ios_id": "1250964862",           # https://apps.apple.com/de/app/nordkurier/id1250964862
        "android_id": "de.nordkurier.live", # https://play.google.com/store/apps/details?id=de.nordkurier.live
        "country": "de"
    },
    {
        "name": "Schwäbische",
        "ios_id": "432491155",            # https://apps.apple.com/de/app/schw%C3%A4bische/id432491155
        "android_id": "de.schwaebische.epaper", # https://play.google.com/store/apps/details?id=de.schwaebische.epaper
        "country": "de"
    }
]

# ---------------------------------------------------------
# SCRAPING FUNKTIONEN
# ---------------------------------------------------------
def fetch_ios_reviews(app_name, app_id, country="de", count=25):
    """Holt Bewertungen aus dem Apple App Store"""
    print(f"   -> Lade iOS Reviews für {app_name} (ID: {app_id})...")
    try:
        app = AppStore(country=country, app_name=app_name, app_id=app_id)
        app.review(how_many=count)

        results = []
        for r in app.reviews:
            results.append({
                "store": "ios",
                "app": app_name,
                "rating": r['rating'],
                "text": r['review'],
                "title": r.get('title', ''),
                "version": r.get('version', ''),
                "date": r['date'].strftime("%Y-%m-%d")
            })
        return results
    except Exception as e:
        print(f"   ❌ Fehler bei iOS ({app_name}): {e}")
        return []

def fetch_android_reviews(app_name, app_id, country="de", count=25):
    """Holt Bewertungen aus dem Google Play Store"""
    print(f"   -> Lade Android Reviews für {app_name} (ID: {app_id})...")
    try:
        result, _ = play_reviews(
            app_id,
            lang='de',
            country=country,
            sort=Sort.NEWEST,
            count=count
        )

        cleaned = []
        for r in result:
            cleaned.append({
                "store": "android",
                "app": app_name,
                "rating": r['score'],
                "text": r['content'],
                "version": r.get('reviewCreatedVersion', ''),
                "date": r['at'].strftime("%Y-%m-%d")
            })
        return cleaned
    except Exception as e:
        print(f"   ❌ Fehler bei Android ({app_name}): {e}")
        return []

def get_all_reviews():
    """Sammelt alle Reviews aller konfigurierten Apps"""
    all_data = []
    print("--- Starte Scraping ---")
    for app in APP_CONFIG:
        # iOS abrufen
        if app.get("ios_id"):
            all_data.extend(fetch_ios_reviews(app["name"], app["ios_id"], app["country"]))

        # Android abrufen
        if app.get("android_id"):
            all_data.extend(fetch_android_reviews(app["name"], app["android_id"], app["country"]))

    print(f"--- Scraping beendet: {len(all_data)} Reviews geladen ---\n")
    return all_data

# ---------------------------------------------------------
# HTML DASHBOARD GENERATOR
# ---------------------------------------------------------
def generate_html(data, review_count):
    """Erstellt die index.html Datei"""
    html = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>App Feedback Dashboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f0f2f5; color: #1c1e21; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 2px 15px rgba(0,0,0,0.05); }}
            h1 {{ margin-top: 0; border-bottom: 1px solid #e4e6eb; padding-bottom: 20px; font-size: 24px; }}
            .badge {{ background: #e4e6eb; color: #65676b; padding: 4px 8px; border-radius: 6px; font-size: 14px; font-weight: normal; margin-left: 10px; vertical-align: middle; }}
            
            /* Metriken */
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 30px 0; }}
            .metric-card {{ background: #f7f8fa; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #ddd; }}
            .metric-card h3 {{ margin: 0; font-size: 12px; color: #65676b; text-transform: uppercase; letter-spacing: 1px; }}
            .metric-card .value {{ font-size: 28px; font-weight: 700; color: #1877f2; margin-top: 10px; }}
            
            /* Zusammenfassung */
            .summary-box {{ background: #e7f3ff; padding: 25px; border-radius: 8px; border-left: 4px solid #1877f2; margin-bottom: 40px; line-height: 1.6; font-size: 16px; }}
            
            /* Topics */
            .tags {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 40px; }}
            .tag {{ background: #e4e6eb; color: #050505; padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: 500; }}
            
            /* Reviews Grid */
            .reviews-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
            @media (max-width: 768px) {{ .reviews-container {{ grid-template-columns: 1fr; }} }}
            
            .review-list h3 {{ color: #65676b; font-size: 16px; margin-bottom: 15px; text-transform: uppercase; }}
            .review-card {{ background: white; border: 1px solid #e4e6eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; font-size: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
            .review-card.good {{ border-left: 4px solid #31a24c; }}
            .review-card.bad {{ border-left: 4px solid #fa383e; }}
            .stars {{ color: #f5c33b; letter-spacing: 2px; font-size: 16px; margin-bottom: 8px; display: block; }}
            .meta {{ font-size: 12px; color: #65676b; margin-top: 10px; display: flex; justify-content: space-between; }}
            
            footer {{ margin-top: 60px; text-align: center; font-size: 12px; color: #b0b3b8; border-top: 1px solid #e4e6eb; padding-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>
                App Feedback Report
                <span class="badge">{datetime.now().strftime('%d.%m.%Y')}</span>
                <span class="badge" style="float:right; font-size: 12px;">Basis: {review_count} Reviews</span>
            </h1>
            
            <div class="summary-box">
                <strong>🤖 KI Management Summary:</strong><br><br>
                {data.get('summary', 'Keine Analyse verfügbar.')}
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Durchschnitt</h3>
                    <div class="value">{data.get('metrics', {}).get('avgRating', 0)} ⭐</div>
                </div>
                <div class="metric-card">
                    <h3>iOS Reviews</h3>
                    <div class="value">{data.get('metrics', {}).get('byStore', {}).get('ios', 0)}</div>
                </div>
                <div class="metric-card">
                    <h3>Android Reviews</h3>
                    <div class="value">{data.get('metrics', {}).get('byStore', {}).get('android', 0)}</div>
                </div>
            </div>

            <h3>🔥 Aktuelle Themen & Probleme</h3>
            <div class="tags">
                {''.join([f'<div class="tag">{t}</div>' for t in data.get('topics', [])])}
            </div>
            
            <div class="reviews-container">
                <div class="review-list">
                    <h3>👍 Was gut läuft</h3>
                    {''.join([f'''
                    <div class="review-card good">
                        <span class="stars">{'★' * int(r.get('rating', 0))}</span>
                        "{r.get('text', '')[:200]}..."
                        <div class="meta">
                            <b>{r.get('app', '')}</b>
                            <span>{r.get('store', '').upper()}</span>
                        </div>
                    </div>
                    ''' for r in data.get('topReviews', [])])}
                </div>
                
                <div class="review-list">
                    <h3>⚠️ Was kritisiert wird</h3>
                    {''.join([f'''
                    <div class="review-card bad">
                        <span class="stars">{'★' * int(r.get('rating', 0))}</span>
                        "{r.get('text', '')[:200]}..."
                        <div class="meta">
                            <b>{r.get('app', '')}</b>
                            <span>{r.get('store', '').upper()}</span>
                        </div>
                    </div>
                    ''' for r in data.get('bottomReviews', [])])}
                </div>
            </div>
            
            <footer>
                Automated Analysis Agent • Powered by Google Gemini 2.0 • GitHub Actions
            </footer>
        </div>
    </body>
    </html>
    """
    return html

# ---------------------------------------------------------
# HAUPTPROGRAMM (RUN AGENT)
# ---------------------------------------------------------
def run_agent():
    # 1. Daten holen
    real_reviews = get_all_reviews()

    if not real_reviews:
        print("⚠️ Keine Reviews gefunden. Prüfe Internetverbindung oder App-IDs.")
        # Dummy Fallback um Crash zu vermeiden
        real_reviews = [{"store": "error", "app": "System", "rating": 0, "text": "Keine Daten konnte geladen werden."}]

    # 2. KI Analyse
    print(f"--- Starte KI Analyse mit {len(real_reviews)} Reviews ---")
    if model:
        # Daten für Prompt vorbereiten (nur relevante Felder um Tokens zu sparen)
        prompt_data = [{k: v for k, v in r.items() if k in ['app', 'rating', 'text', 'store', 'date']} for r in real_reviews]

        prompt = f"{SYSTEM_INSTRUCTION}\n\nANALYZE THESE REVIEWS:\n{json.dumps(prompt_data, ensure_ascii=False)}"

        try:
            response = model.generate_content(prompt)
            analysis_data = json.loads(response.text)
            print("✅ Analyse erfolgreich.")
        except Exception as e:
            print(f"❌ KI Fehler: {e}")
            analysis_data = {"summary": f"Fehler: {e}", "metrics": {}, "topics": []}
    else:
        analysis_data = {"summary": "Lokaler Modus (Kein API Key gesetzt)", "metrics": {}, "topics": []}

    # 3. HTML generieren und speichern
    dashboard_html = generate_html(analysis_data, len(real_reviews))

    # Ordner 'public' erstellen, falls nicht vorhanden
    os.makedirs("public", exist_ok=True)

    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    print("\n✅ Dashboard generiert: public/index.html")

if __name__ == "__main__":
    run_agent()