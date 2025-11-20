import os
import json
import time
import hashlib
import requests
from requests.exceptions import HTTPError
from datetime import datetime, timedelta

# KI & Vektor Imports
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Scraper Imports
from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews as play_reviews

# ---------------------------------------------------------
# 1. SETUP & KONFIGURATION
# ---------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# KI und Embeddings initialisieren
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={"response_mime_type": "application/json"}
        )
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"WARNUNG: KI oder Embedding Modell konnte nicht geladen werden: {e}")
        model = None
        embedder = None
else:
    model = None
    embedder = None

# Dateipfade und App-Konfiguration
DATA_FILE = "data/reviews_history.json"
APP_CONFIG = [
    {"name": "Nordkurier", "ios_id": "1250964862", "android_id": "de.nordkurier.live", "country": "de"},
    {"name": "Schwäbische", "ios_id": "432491155", "android_id": "de.schwaebische.epaper", "country": "de"}
]

# ---------------------------------------------------------
# 2. HILFSFUNKTIONEN (Datenhaltung & Trends)
# ---------------------------------------------------------
def generate_id(review):
    """Erstellt einen eindeutigen Hash für das Review."""
    unique_string = f"{review.get('text', '')[:50]}{review.get('date', '')}{review.get('app', '')}{review.get('store', '')}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

def load_history():
    """Lädt die Historie und gibt sie als Dictionary zurück (WICHTIG für Performance)."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                # Konvertiert die Liste in ein Dictionary: { "ID": {ReviewDaten} }
                return {r['id']: r for r in raw_data if 'id' in r}
        except json.JSONDecodeError:
            print("WARNUNG: History-Datei korrupt, starte neu.")
    return {}

def save_history(history_dict):
    """Speichert das Dictionary wieder als sortierte Liste ab."""
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    # Sortieren: Neueste zuerst
    data_list = sorted(history_dict.values(), key=lambda x: x['date'], reverse=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

def calculate_trends(reviews):
    """Berechnet Durchschnittswerte für 7, 30 Tage und Gesamt."""
    today = datetime.now().date()
    dated_reviews = []

    for r in reviews:
        try:
            # Datum parsen
            review_date = datetime.strptime(r['date'], '%Y-%m-%d').date()
            if r.get('text'):
                dated_reviews.append((review_date, float(r['rating'])))
        except ValueError:
            continue

    if not dated_reviews:
        return {'overall': 0.0, 'last_7d': 0.0, 'last_30d': 0.0}

    def get_avg_for_days(days):
        cutoff_date = today - timedelta(days=days)
        filtered = [rating for date, rating in dated_reviews if date >= cutoff_date]
        if not filtered:
            return 0.0
        return round(sum(filtered) / len(filtered), 2)

    overall_avg = round(sum(r['rating'] for r in reviews) / len(reviews), 2)

    return {
        'overall': overall_avg,
        'last_7d': get_avg_for_days(7),
        'last_30d': get_avg_for_days(30)
    }

# ---------------------------------------------------------
# 3. SCRAPING FUNKTIONEN
# ---------------------------------------------------------
def fetch_ios_reviews(app_name, app_id, country="de", count=20):
    """Holt iOS Reviews via RSS Feed (Stabil)."""
    print(f"   -> iOS (RSS): {app_name}...")
    api_url = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostrecent/json"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = []

        entries = data.get('feed', {}).get('entry', [])
        # RSS Feed liefert oft bis zu 50, wir nehmen was da ist
        for entry in entries[:count]:
            if 'im:rating' not in entry or 'content' not in entry:
                continue

            rating = int(entry['im:rating']['label'])
            text = entry['content']['label']
            # Datumformat im RSS ist oft komplex, wir nehmen vereinfacht heute oder extrahieren grob
            # Im RSS Feed ist das Datum oft nicht sauber als YYYY-MM-DD, wir nutzen label[:10] als Näherung
            # oder 'updated' Feld.
            date_str = entry.get('updated', {}).get('label', datetime.now().strftime('%Y-%m-%d'))[:10]

            results.append({
                "store": "ios", "app": app_name, "rating": rating, "text": text, "date": date_str,
                "id": generate_id({'app': app_name, 'store': 'ios', 'date': date_str, 'text': text})
            })
        return results
    except Exception as e:
        print(f"      ❌ iOS (RSS) Fehler: {e}")
        return []

def fetch_android_reviews(app_name, app_id, country="de", count=20):
    """Holt Android Reviews via Scraper."""
    print(f"   -> Android: {app_name}...")
    try:
        result, _ = play_reviews(app_id, lang=country, country=country, sort=Sort.NEWEST, count=count)
        reviews = []
        for r in result:
            date_str = r['at'].strftime('%Y-%m-%d')
            reviews.append({
                "store": "android", "app": app_name, "rating": r['score'], "text": r['content'],
                "date": date_str,
                "id": generate_id({'app': app_name, 'store': 'android', 'date': date_str, 'text': r['content']})
            })
        return reviews
    except Exception as e:
        print(f"      ❌ Android Fehler: {e}")
        return []

def get_fresh_reviews(review_count=20):
    """Hauptfunktion zum Laden neuer Daten."""
    history_dict = load_history()
    new_reviews_list = []

    print(f"--- Starte Scrape für {len(APP_CONFIG)*2} Quellen ---")

    for app in APP_CONFIG:
        # iOS abrufen
        ios_reviews = fetch_ios_reviews(app['name'], app['ios_id'], app['country'], review_count)
        # Android abrufen
        android_reviews = fetch_android_reviews(app['name'], app['android_id'], app['country'], review_count)

        # Zusammenfügen
        all_scraped = ios_reviews + android_reviews

        for r in all_scraped:
            # Prüfen ob ID schon bekannt ist
            if r['id'] not in history_dict:
                history_dict[r['id']] = r
                new_reviews_list.append(r)

    # Rückgabe: Gesamte Historie (Liste) und Neue Reviews (Liste)
    full_history = sorted(history_dict.values(), key=lambda x: x['date'], reverse=True)
    print(f"\n--- STATUS: {len(full_history)} Gesamt in DB, davon {len(new_reviews_list)} NEU gefunden ---")
    return full_history, new_reviews_list

# ---------------------------------------------------------
# 4. SEMANTISCHE CLUSTER-ANALYSE (Stage 2)
# ---------------------------------------------------------
def get_semantic_topics(reviews, num_clusters=5):
    """Gruppiert Reviews nach Themen und benennt diese per KI."""
    if not embedder:
        return ["Embedding Modell nicht geladen."]

    # Nur Reviews mit ausreichend Text
    text_reviews = [r for r in reviews[:200] if r.get('text') and len(r['text']) > 15]

    if len(text_reviews) < num_clusters:
        return ["Zu wenige Reviews für Clustering."]

    texts = [r['text'] for r in text_reviews]

    # 1. Embeddings
    embeddings = embedder.encode(texts)

    # 2. KMeans Clustering
    # Begrenze Cluster-Anzahl, falls weniger Texte da sind als gewünschte Cluster
    actual_clusters = min(num_clusters, len(texts))
    clustering = KMeans(n_clusters=actual_clusters, random_state=0, n_init=10)
    clustering.fit(embeddings)

    topic_reviews = []

    # 3. Repräsentative Reviews finden
    for i in range(actual_clusters):
        cluster_indices = np.where(clustering.labels_ == i)[0]
        if not cluster_indices.size: continue

        centroid = clustering.cluster_centers_[i]
        similarity = cosine_similarity([centroid], embeddings[cluster_indices])
        closest_idx = cluster_indices[np.argmax(similarity)]

        topic_reviews.append(text_reviews[closest_idx])

    # 4. KI Benennung
    if not model or not topic_reviews:
        return ["Clustering erfolgreich, KI nicht verfügbar."]

    # Daten für Prompt vorbereiten
    prompt_reviews = [{"review": r['text'], "app": r['app'], "rating": r['rating']} for r in topic_reviews]

    prompt = f"""
    You are an expert market analyst. Assign a single, concise German label (max 3 words) to each review's underlying topic.
    Output strictly in a JSON list of strings, containing ONLY the {len(topic_reviews)} topic labels.
    
    Reviews to label: {json.dumps(prompt_reviews, ensure_ascii=False)}
    """

    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        topics = json.loads(text)
        # Sicherstellen, dass es eine Liste von Strings ist
        return [t for t in topics if isinstance(t, str) and t]
    except Exception as e:
        print(f"KI-Labeling Fehler: {e}")
        return ["KI Labeling Fehler"]

# ---------------------------------------------------------
# 5. HTML GENERIERUNG (Robust gegen Fehler)
# ---------------------------------------------------------
def run_analysis_and_generate_html(full_history, new_only):
    """Erstellt Dashboard HTML mit Trends, KI-Analyse und Suche."""

    # Trends berechnen
    trend_metrics = calculate_trends(full_history)
    total_count = len(full_history)

    # Themen Cluster
    semantic_topics = get_semantic_topics(full_history)

    # KI Analyse (Summary & Top/Low)
    analysis_set = full_history[:50]
    ki_output = {"summary": "Keine Analyse verfügbar.", "topReviews": [], "bottomReviews": []}

    if model and analysis_set:
        print(f"--- Starte KI-Analyse für {len(analysis_set)} Reviews ---")
        prompt_data = [{k: v for k, v in r.items() if k in ['text', 'rating', 'app', 'store']} for r in analysis_set]

        prompt = f"""
        Analysiere diese Reviews (max 50). 
        1. Fasse die Stimmung in einem kurzen Management Summary (Deutsch) zusammen.
        2. Wähle 3 Top-Reviews (Positiv) und 3 Bottom-Reviews (Negativ).
        
        Output MUSS valides JSON sein:
        {{
            "summary": "Dein Text hier...",
            "topReviews": [{{ "text": "...", "store": "...", "rating": 5 }}, ...],
            "bottomReviews": [{{ "text": "...", "store": "...", "rating": 1 }}, ...]
        }}
        
        Data: {json.dumps(prompt_data, ensure_ascii=False)}
        """
        try:
            response = model.generate_content(prompt)
            # JSON Cleaning
            text = response.text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            ki_output.update(parsed)
        except Exception as e:
            print(f"❌ KI Fehler: {e}")

    # --- ROBUSTE ZUSAMMENFASSUNG (Der Fix für den Absturz) ---
    raw_summary = ki_output.get('summary', 'Keine Analyse verfügbar.')

    # Falls die KI ein Objekt statt Text zurückgibt, konvertieren wir es
    if isinstance(raw_summary, dict):
        final_summary = raw_summary.get('text', raw_summary.get('content', str(raw_summary)))
    elif isinstance(raw_summary, list):
        final_summary = " ".join([str(x) for x in raw_summary])
    else:
        final_summary = str(raw_summary)

    # Endreinigung von JSON-Fragmenten
    final_summary = final_summary.strip().replace('{', '').replace('}', '').replace('"', '')
    # ----------------------------------------------------------

    # Top/Low Listen holen
    top_reviews = ki_output.get('topReviews', [])
    bottom_reviews = ki_output.get('bottomReviews', [])

    # HTML Template
    html = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>App Feedback Dashboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f4f6f8; padding: 20px; color: #333; }}
            .container {{ max-width: 960px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
            h1 {{ border-bottom: 2px solid #eee; padding-bottom: 15px; color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            
            .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #ddd; flex: 1; min-width: 150px; }}
            .val {{ font-size: 32px; font-weight: bold; color: #2c3e50; margin-top: 10px; }}
            .flex {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 30px; }}
            
            .summary {{ background: #e3f2fd; padding: 20px; border-radius: 8px; border-left: 6px solid #2196f3; line-height: 1.6; font-size: 1.1em; }}
            
            .topic {{ display: inline-block; background: #e8f5e9; color: #2e7d32; padding: 8px 15px; border-radius: 20px; margin: 5px; border: 1px solid #c8e6c9; font-weight: 500; }}
            
            .review-list {{ display: flex; gap: 20px; flex-wrap: wrap; }}
            .review-column {{ flex: 1; min-width: 300px; }}
            
            .review-item {{ border: 1px solid #eee; padding: 15px; border-radius: 8px; margin-bottom: 15px; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }}
            .good {{ border-left: 5px solid #28a745; }}
            .bad {{ border-left: 5px solid #dc3545; }}
            
            .metadata {{ font-size: 0.85em; color: #7f8c8d; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: bold; }}
            .review-text {{ font-style: italic; color: #555; }}
            
            .search-box input {{ width: 100%; padding: 15px; border: 2px solid #eee; border-radius: 8px; font-size: 16px; margin-bottom: 20px; box-sizing: border-box; }}
            
            #review-container {{ display: grid; gap: 15px; }}
            .raw-review {{ border-bottom: 1px solid #eee; padding-bottom: 15px; }}
        </style>
        
        <script>
            const ALL_REVIEWS = {json.dumps(full_history, ensure_ascii=False)};
        </script>
    </head>
    <body>
        <div class="container">
            <h1>📊 App Feedback Report</h1>
            <p class="metadata">Update: {datetime.now().strftime('%d.%m.%Y %H:%M')} | Basis: {total_count} Reviews</p>
            
            <div class="flex">
                <div class="card"><h3>7 Tage Ø</h3><div class="val">{trend_metrics['last_7d']} ⭐</div></div>
                <div class="card"><h3>30 Tage Ø</h3><div class="val">{trend_metrics['last_30d']} ⭐</div></div>
                <div class="card"><h3>Gesamt Ø</h3><div class="val">{trend_metrics['overall']} ⭐</div></div>
            </div>
            
            <div class="summary"><strong>🤖 KI Analyse:</strong><br>{final_summary}</div>
            
            <h3>🔥 Aktuelle Themen-Cluster</h3>
            <div>{''.join([f'<span class="topic">{t}</span> ' for t in semantic_topics])}</div>
            
            <div class="review-list">
                <div class="review-column">
                    <h3>👍 Top Stimmen</h3>
                    {''.join([f'''
                    <div class="review-item good">
                        <div class="metadata">{r.get('rating')}★ | {r.get('store', '').upper()}</div>
                        <div class="review-text">"{r.get('text')}"</div>
                    </div>
                    ''' for r in top_reviews[:3]])}
                </div>
                <div class="review-column">
                    <h3>⚠️ Kritische Stimmen</h3>
                    {''.join([f'''
                    <div class="review-item bad">
                        <div class="metadata">{r.get('rating')}★ | {r.get('store', '').upper()}</div>
                        <div class="review-text">"{r.get('text')}"</div>
                    </div>
                    ''' for r in bottom_reviews[:3]])}
                </div>
            </div>
            
            <h2 style="margin-top: 50px;">🔍 Review Explorer</h2>
            <div class="search-box">
                <input type="text" id="search" placeholder="Suche nach Stichworten (z.B. 'Login', 'Absturz')..." onkeyup="filterReviews()">
            </div>
            <div id="review-container"></div>
            
            <script>
                function render(reviews) {{
                    const c = document.getElementById('review-container');
                    c.innerHTML = '';
                    if(reviews.length === 0) {{ c.innerHTML = '<p style="color:#999">Keine Ergebnisse.</p>'; return; }}
                    
                    reviews.slice(0, 50).forEach(r => {{
                        c.innerHTML += `
                        <div class="raw-review">
                            <div class="metadata"><b>${{r.app}} (${{r.store}})</b> | ${{r.rating}}★ | ${{r.date}}</div>
                            <div>${{r.text}}</div>
                        </div>`;
                    }});
                }}
                
                function filterReviews() {{
                    const q = document.getElementById('search').value.toLowerCase();
                    const res = ALL_REVIEWS.filter(r => (r.text + r.app).toLowerCase().includes(q));
                    render(res);
                }}
                
                // Init
                render(ALL_REVIEWS);
            </script>
            
            <footer style="margin-top: 50px; text-align: center; color: #ccc; font-size: 0.8em;">
                Automated by Gemini 2.0 & GitHub Actions
            </footer>
        </div>
    </body>
    </html>
    """

    # WICHTIG: Speichern mit write(), nicht json.dump()
    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Dashboard HTML erfolgreich generiert.")

# ---------------------------------------------------------
# 6. TEAMS BENACHRICHTIGUNG (Einfacher Text)
# ---------------------------------------------------------
def send_teams_notification(new_reviews, webhook_url):
    if not new_reviews:
        return

    pos = sum(1 for r in new_reviews if r['rating'] >= 4)
    neg = sum(1 for r in new_reviews if r['rating'] <= 2)

    text_body = f"📢 **NEUES FEEDBACK!** ({len(new_reviews)})\n\n"
    text_body += f"👍 Positiv: {pos} | 🚨 Kritisch: {neg}\n\n"
    text_body += "**Auszug:**\n"

    for r in new_reviews[:3]:
        text_body += f"- {r['rating']}★: {r['text'][:60]}...\n"

    text_body += "\n[Zum Dashboard](https://Hatozoro.github.io/feedback-agent/)"

    try:
        requests.post(webhook_url, json={"text": text_body}, timeout=10)
        print("✅ Teams Nachricht gesendet.")
    except Exception as e:
        print(f"❌ Teams Fehler: {e}")

# ---------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Daten
    full, new = get_fresh_reviews()
    save_history({r['id']: r for r in full})

    # 2. HTML
    run_analysis_and_generate_html(full, new)

    # 3. Alert
    teams_url = os.getenv("TEAMS_WEBHOOK_URL")
    if teams_url:
        send_teams_notification(new, teams_url)

    print("✅ Fertig.")