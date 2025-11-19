import os
import json
import time
import hashlib
import requests
from requests.exceptions import HTTPError
from datetime import datetime

# KI & Vektor Imports
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Scraper Imports
from app_store_scraper import AppStore # Wird nur für Metadaten geladen
from google_play_scraper import Sort, reviews as play_reviews

# ---------------------------------------------------------
# 1. SETUP & KONFIGURATION
# ---------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

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

# Konfiguration der Datenhaltung und Apps
DATA_FILE = "data/reviews_history.json"
APP_CONFIG = [
    {"name": "Nordkurier", "ios_id": "1250964862", "android_id": "de.nordkurier.live", "country": "de"},
    {"name": "Schwäbische", "ios_id": "432491155", "android_id": "de.schwaebische.epaper", "country": "de"}
]

# ---------------------------------------------------------
# 2. DATENHALTUNG (FIX für TypeError: List vs. Dict)
# ---------------------------------------------------------
def generate_id(review):
    """Generiert eine stabile ID für ein Review (Hash von Text, Datum und App)."""
    unique_string = f"{review.get('text', '')[:50]}{review.get('date', '')}{review.get('app', '')}{review.get('store', '')}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

def load_history():
    """Lädt die persistente Historie-Datei als Dictionary (ID -> Review)."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                # Konvertiere Liste zu Dictionary (das verhindert den TypeError)
                return {r['id']: r for r in raw_data if 'id' in r}
        except json.JSONDecodeError:
            print("WARNUNG: History-Datei korrupt.")
    return {} # <--- FIX: Immer ein Dictionary zurückgeben!

def save_history(history_dict):
    """Speichert die Historie-Datei als sortierte Liste."""
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

    # Konvertiere Dictionary-Werte zurück zur Liste
    data_list = sorted(history_dict.values(), key=lambda x: x['date'], reverse=True)

    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

# ---------------------------------------------------------
# 3. SCRAPING FUNKTIONEN (iOS stabilisiert)
# ---------------------------------------------------------
def fetch_ios_reviews(app_name, app_id, country="de", count=20):
    """Holt iOS Reviews über die stabile Apple RSS API."""
    print(f"   -> iOS (RSS): {app_name}...")
    api_url = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostrecent/json"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for entry in data.get('feed', {}).get('entry', [])[:count]:
            if 'im:rating' not in entry or 'content' not in entry:
                continue

            rating = int(entry['im:rating']['label'])
            text = entry['content']['label']
            date_str = entry['updated']['label'][:10]

            results.append({
                "store": "ios", "app": app_name, "rating": rating,
                "text": text, "date": date_str,
                "id": generate_id({'app': app_name, 'store': 'ios', 'date': date_str, 'text': text})
            })
        return results
    except Exception as e:
        print(f"      ❌ iOS (RSS) Fehler: {e}")
        return []

def fetch_android_reviews(app_name, app_id, country="de", count=20):
    """Holt Android Reviews über google-play-scraper."""
    print(f"   -> Android: {app_name}...")
    try:
        result, _ = play_reviews(
            app_id, lang=country, country=country, sort=Sort.NEWEST, count=count
        )
        reviews = []
        for r in result:
            reviews.append({
                "store": "android", "app": app_name, "rating": r['score'],
                "text": r['content'], "date": r['at'].strftime('%Y-%m-%d'),
                "id": generate_id({'app': app_name, 'store': 'android', 'date': r['at'].strftime('%Y-%m-%d'), 'text': r['content']})
            })
        return reviews
    except Exception as e:
        print(f"      ❌ Android Fehler: {e}")
        return []

def get_fresh_reviews(review_count=20):
    """Sammelt neue Reviews von allen Stores und mergt sie mit der Historie."""

    print(f"--- Starte Scrape für 4 Quellen ---")

    history_dict = load_history()
    new_reviews_list = []

    for app in APP_CONFIG:
        # iOS
        ios_reviews = fetch_ios_reviews(app['name'], app['ios_id'], app['country'], review_count)
        # Android
        android_reviews = fetch_android_reviews(app['name'], app['android_id'], app['country'], review_count)

        all_scraped = ios_reviews + android_reviews

        for r in all_scraped:
            # FIX: Wenn ID schon existiert, überspringe das Review
            if r['id'] not in history_dict:
                history_dict[r['id']] = r
                new_reviews_list.append(r)

    # Sortiere die gesamte Historie nach Datum (neueste zuerst)
    full_history = sorted(history_dict.values(), key=lambda x: x['date'], reverse=True)

    print(f"\n--- ENDE: {len(full_history)} Gesamt, davon {len(new_reviews_list)} NEU ---")

    return full_history, new_reviews_list

# ---------------------------------------------------------
# 4. SEMANTISCHE CLUSTER-ANALYSE
# ---------------------------------------------------------
def get_semantic_topics(reviews, num_clusters=5):
    """Führt KMeans Clustering durch und lässt die KI die Cluster benennen."""
    if not embedder:
        return ["Embedding Modell nicht geladen."]

    text_reviews = [r for r in reviews[:200] if r.get('text') and len(r['text']) > 15]
    if len(text_reviews) < num_clusters:
        return ["Zu wenige Reviews für Clustering."]

    texts = [r['text'] for r in text_reviews]

    # 1. Vektorisierung
    embeddings = embedder.encode(texts)

    # 2. Clustering (K-Means)
    num_clusters = min(num_clusters, len(texts))
    clustering = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    clustering.fit(embeddings)

    # 3. Finde das repräsentativste Review pro Cluster
    topic_reviews = []

    for i in range(num_clusters):
        cluster_indices = np.where(clustering.labels_ == i)[0]
        if not cluster_indices.size:
            continue

        cluster_embeddings = embeddings[cluster_indices]
        centroid = clustering.cluster_centers_[i]

        similarity = cosine_similarity([centroid], cluster_embeddings)
        closest_index_in_cluster = cluster_indices[np.argmax(similarity)]

        topic_reviews.append(text_reviews[closest_index_in_cluster])

    # 4. KI-Benennung der Cluster
    if not model or not topic_reviews:
        return ["Clustering erfolgreich, KI nicht verfügbar."]

    prompt_reviews = [{"review": r['text'], "app": r['app'], "rating": r['rating']} for r in topic_reviews]

    prompt = f"""
    You are an expert market analyst. Analyze the following {len(topic_reviews)} reviews, each representing a distinct user feedback topic.
    Your task is to assign a single, concise German label (max 3 words) to each review's underlying topic.
    Output strictly in a JSON list of strings, containing ONLY the {len(topic_reviews)} topic labels.

    Example Output: ["Abstürze Allgemein", "Login Fehler", "Performance", "Neue Artikel"]

    Reviews to label: {json.dumps(prompt_reviews, ensure_ascii=False)}
    """

    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        topics = json.loads(text)
        return [t for t in topics if isinstance(t, str) and t]
    except Exception as e:
        print(f"KI-Labeling Fehler: {e}")
        return ["KI-Labeling Fehlgeschlagen"]

# ---------------------------------------------------------
# 5. TEAMS NOTIFICATION
# ---------------------------------------------------------
def send_teams_notification(new_reviews, webhook_url):
    """Sendet eine Nachricht an den Teams/Power Automate Webhook."""
    if not new_reviews:
        print("-> Keine neuen Reviews zum Senden.")
        return

    positive_count = sum(1 for r in new_reviews if r['rating'] >= 4)
    negative_count = sum(1 for r in new_reviews if r['rating'] <= 2)

    title = f"📢 NEUES FEEDBACK! ({len(new_reviews)} Reviews)"

    review_list_text = ""
    for r in new_reviews[:5]:
        rating_star = "⭐" * r['rating']
        review_list_text += f"* **{r['app']} ({r['store']})**: {rating_star} *\"{r['text'][:60]}...\"*\n"

    full_text_message = f"""
📢 **{title}**
---
👍 Positiv: {positive_count} | 🚨 Kritisch: {negative_count}

**Neueste Nutzerstimmen (Auszug):**
{review_list_text}

[Zum vollständigen Dashboard](https://Hatozoro.github.io/feedback-agent/)
"""
    teams_message = {
        "text": full_text_message
    }

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(webhook_url, json=teams_message, headers=headers, timeout=10)

        response.raise_for_status()
        print("✅ Teams Benachrichtigung erfolgreich gesendet.")
    except HTTPError as e:
        print(f"❌ HTTP Fehler beim Senden der Teams-Nachricht: {response.status_code} Client Error: {response.reason}. Prüfe den Webhook.")
    except Exception as e:
        print(f"❌ Allgemeiner Fehler beim Senden der Teams-Nachricht: {e}")

# ---------------------------------------------------------
# 6. HTML GENERIERUNG
# ---------------------------------------------------------
def run_analysis_and_generate_html(full_history, new_only):
    """Erstellt die KI-Analyse und generiert das statische HTML-Dashboard."""

    # 1. Clustering durchführen
    semantic_topics = get_semantic_topics(full_history)

    # 2. KI Analyse für Summary und Top/Low Reviews
    analysis_set = full_history[:50]
    ki_output = {"summary": "Keine ausreichende Datenbasis für KI-Analyse.", "topReviews": [], "bottomReviews": []}

    if model and analysis_set:
        print(f"--- Starte KI-Analyse für {len(analysis_set)} Reviews ---")
        prompt_data = [{k: v for k, v in r.items() if k in ['text', 'rating', 'app', 'store']} for r in analysis_set]

        prompt = f"""
        Analysiere diese Reviews (max 50). Fasse die wichtigsten Benutzerbeschwerden und Produktbereiche zusammen.
        Wähle **DREI** Top-Reviews (höchste positive Bewertung, bester Text) und **DREI** Bottom-Reviews (niedrigste Bewertung, kritischster Text) aus.
        
        Output muss STRIKT im JSON-Format erfolgen. topReviews und bottomReviews **MÜSSEN** jeweils 3 Elemente enthalten:
        {{
            "summary": "Management Summary in deutscher Sprache mit Fokus auf aktuelle Trends und Hauptprobleme. (Max 50 Wörter)",
            "topReviews": [
                {{ "text": "...", "store": "...", "rating": 5 }}, 
                {{ "text": "...", "store": "...", "rating": 5 }}, 
                {{ "text": "...", "store": "...", "rating": 5 }}
            ],
            "bottomReviews": [
                {{ "text": "...", "store": "...", "rating": 1 }}, 
                {{ "text": "...", "store": "...", "rating": 1 }}, 
                {{ "text": "...", "store": "...", "rating": 1 }}
            ]
        }}
        
        Review Data: {json.dumps(prompt_data, ensure_ascii=False)}
        """
        try:
            response = model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            ki_output.update(json.loads(text))
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
            .topic {{ display: inline-block; background: #e8f5e9; color: #2e7d32; padding: 5px 12px; border-radius: 15px; margin: 3px; display: inline-block; border: 1px solid #c8e6c9; font-size: 0.9em; }}
            .review-list {{ display: flex; gap: 20px; margin-top: 20px; }}
            .review-column {{ flex: 1; min-width: 40%; }}
            .review-item {{ border: 1px solid #eee; padding: 10px; border-radius: 5px; margin-bottom: 10px; background: white; box-shadow: 0 1px 3px rgba(0,0.02); }}
            .good {{ border-left: 5px solid #28a745; }}
            .bad {{ border-left: 5px solid #dc3545; }}
            .review-text {{ font-style: italic; font-size: 0.9em; margin-top: 5px; }}
            .metadata {{ font-size: 0.9em; color: #6c757d; margin-top: 5px; display: block; font-weight: bold; }}
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
            <div>{''.join([f'<span class="topic">{t}</span>' for t in semantic_topics])}</div>
            
            <h2>⭐ Top- & Low-Reviews (KI-Auswahl)</h2>
            <div class="review-list">
                <div class="review-column">
                    <h4>Top 3 Reviews (Positiv)</h4>
                    {''.join([f'''
                    <div class="review-item good">
                        <div class="metadata">
                            {r.get('rating', 'N/A')}★ | {r.get('store', 'N/A').upper()}
                        </div>
                        <div class="review-text">"{r.get('text', 'Review-Text fehlt')}"</div>
                    </div>
                    ''' for r in top_reviews])}
                </div>
                <div class="review-column">
                    <h4>Low 3 Reviews (Kritisch)</h4>
                    {''.join([f'''
                    <div class="review-item bad">
                        <div class="metadata">
                            {r.get('rating', 'N/A')}★ | {r.get('store', 'N/A').upper()}
                        </div>
                        <div class="review-text">"{r.get('text', 'Review-Text fehlt')}"</div>
                    </div>
                    ''' for r in bottom_reviews])}
                </div>
            </div>
            
            <h2>📝 Neueste Rohdaten</h2>
            <div>
                {''.join([f'''
                <div class="review-item">
                    <div class="metadata"><b>{r["app"]} ({r["store"]}) - {r["rating"]}★</b> | {r["date"]}</div>
                    <div class="review-text">"{r["text"][:250]}..."</div>
                </div>
                ''' for r in full_history[:10]])}
            </div>
        </div>
    </body>
    </html>
    """

    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html) # F.write ist der korrekte Befehl für HTML

    print("✅ Dashboard HTML erfolgreich generiert.")


# ---------------------------------------------------------
# 8. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Daten holen & Duplikate filtern
    full_history, new_reviews = get_fresh_reviews()

    # 2. Speichern (Persistenz)
    save_history({r['id']: r for r in full_history}) # Speichere als Dictionary, wie in load_history erwartet

    # 3. Dashboard bauen
    run_analysis_and_generate_html(full_history, new_reviews)

    # 4. Chat-Benachrichtigung senden
    teams_webhook = os.getenv("TEAMS_WEBHOOK_URL")
    if teams_webhook:
        send_teams_notification(new_reviews, teams_webhook)

    print("✅ Durchlauf beendet. Ready for Commit.")