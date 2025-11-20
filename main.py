import os
import json
import time
import hashlib
import requests
import re
from collections import Counter
from requests.exceptions import HTTPError
from datetime import datetime, timedelta

# ---------------------------------------------------------
# IMPORTS: KI & DATEN-ANALYSE
# ---------------------------------------------------------
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------------------------------
# IMPORTS: SCRAPER
# ---------------------------------------------------------
from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews as play_reviews

# ---------------------------------------------------------
# 1. SETUP & KONFIGURATION
# ---------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# KI-Modelle laden (mit Fehlerbehandlung)
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={"response_mime_type": "application/json"}
        )
        # Embedding Modell für Clustering
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"WARNUNG: KI Module konnten nicht geladen werden: {e}")
        model = None
        embedder = None
else:
    print("WARNUNG: Kein API Key gefunden.")
    model = None
    embedder = None

# Dateipfade
DATA_FILE = "data/reviews_history.json"

# App-Definitionen
APP_CONFIG = [
    {"name": "Nordkurier", "ios_id": "1250964862", "android_id": "de.nordkurier.live", "country": "de"},
    {"name": "Schwäbische", "ios_id": "432491155", "android_id": "de.schwaebische.epaper", "country": "de"}
]

# Stopwords für die Buzzword-Cloud (Wörter, die ignoriert werden)
STOP_WORDS = {
    "die", "der", "und", "in", "zu", "den", "das", "nicht", "von", "sie", "ist", "des", "sich", "mit", "dem", "dass",
    "er", "es", "wir", "ihr", "sie", "mich", "mir", "meine", "meiner", "mein",
    "sich", "uns", "euch", "ihnen", "ihrem", "ihres", "dieser", "diese", "dieses", "diesen",
    "und", "oder", "aber", "als", "wenn", "dass", "weil", "denn", "ob", "wie", "wo", "was",
    "in", "im", "an", "am", "auf", "aus", "bei", "beim", "mit", "nach", "von", "vom", "zu", "zum", "zur",
    "über", "unter", "vor", "hinter", "neben", "durch", "für", "gegen", "ohne", "um", "wegen", "seit",
    "ist", "sind", "war", "wäre", "wird", "werden", "wurde", "haben", "hat", "hatte", "habe", "gibt",
    "kann", "können", "konnte", "muss", "müssen", "musste", "soll", "sollen", "sollte", "will", "wollen",
    "geht", "ging", "lassen", "lässt", "machen", "macht", "getan", "sehen", "sieht", "schon", "nun",
    "nicht", "nichts", "nie", "wieder", "immer", "oft", "selten", "manchmal", "erst", "bereits",
    "noch", "jetzt", "heute", "damals", "hier", "da", "dort", "mal", "einmal", "viel", "sehr", "auch",
    "ganz", "gar", "mehr", "weniger", "nur", "doch", "etwas", "so", "dann", "wann", "warum", "wer",
    "einfach", "leider", "halt", "eben", "wohl", "zwar", "vielleicht", "bestimmt", "bitte", "danke",
    "app", "apps", "anwendung", "version", "update", "updates", "ios", "android", "handy", "tablet",
    "telefon", "iphone", "ipad", "samsung", "pixel", "gerät", "geräte", "nutzer", "kunde", "kunden",
    "schwäbische", "nordkurier", "zeitung", "artikel", "lesen", "leser", "hallo", "moin", "tag",
    "also", "alle", "alles", "viele", "zeit", "seit", "wochen", "monaten", "tagen", "jahre", "sterne", "stern"
}

# ---------------------------------------------------------
# 2. HILFSFUNKTIONEN (Datenbank, Trends, Buzzwords)
# ---------------------------------------------------------

def generate_id(review):
    """Erstellt eine eindeutige ID basierend auf dem Inhalt, um Duplikate zu vermeiden."""
    unique_string = f"{review.get('text', '')[:50]}{review.get('date', '')}{review.get('app', '')}{review.get('store', '')}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

def load_history():
    """Lädt die bestehende Datenbank."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                # Wir nutzen ein Dictionary für schnellen Zugriff per ID
                return {r['id']: r for r in raw_data if 'id' in r}
        except json.JSONDecodeError:
            print("Info: Datenbank war leer oder korrupt, starte neu.")
    return {}

def save_history(history_dict):
    """Speichert die Datenbank zurück auf die Festplatte."""
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    # Sortieren nach Datum (Neueste zuerst) für die JSON Datei
    data_list = sorted(history_dict.values(), key=lambda x: x['date'], reverse=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

def calculate_trends(reviews):
    """Berechnet die Durchschnittsbewertungen für KPIs."""
    today = datetime.now().date()
    dated_reviews = []

    for r in reviews:
        try:
            review_date = datetime.strptime(r['date'], '%Y-%m-%d').date()
            if r.get('text') and r.get('rating') is not None:
                dated_reviews.append((review_date, float(r['rating'])))
        except ValueError:
            continue

    if not dated_reviews:
        return {'overall': 0.0, 'last_7d': 0.0, 'last_30d': 0.0}

    def get_avg_for_days(days):
        cutoff_date = today - timedelta(days=days)
        filtered = [rating for date, rating in dated_reviews if date >= cutoff_date]
        if not filtered: return 0.0
        return round(sum(filtered) / len(filtered), 2)

    overall_avg = round(sum(r['rating'] for r in reviews if r.get('rating') is not None) / (len(reviews) or 1), 2)

    return {
        'overall': overall_avg,
        'last_7d': get_avg_for_days(7),
        'last_30d': get_avg_for_days(30)
    }

def prepare_chart_data(reviews, days=14):
    """Bereitet die Daten für den Graphen vor (Gestapelte Balken)."""
    today = datetime.now().date()
    stats = {}

    # Letzte X Tage initialisieren
    for i in range(days):
        # Wir speichern das Datum im Schlüssel für Sortierung
        date_obj = today - timedelta(days=i)
        date_key = date_obj.strftime('%Y-%m-%d')
        stats[date_key] = {'pos': 0, 'neg': 0, 'neu': 0}

    for r in reviews:
        d = r['date']
        rating = r.get('rating')
        if d in stats and rating is not None:
            if rating >= 4:
                stats[d]['pos'] += 1
            elif rating <= 2:
                stats[d]['neg'] += 1
            else:
                stats[d]['neu'] += 1

    labels_sorted = sorted(stats.keys())

    # Formatierung zu deutschem Datum (TT.MM.)
    formatted_labels = [datetime.strptime(l, '%Y-%m-%d').strftime('%d.%m.') for l in labels_sorted]

    return {
        'labels': formatted_labels,
        'pos': [stats[d]['pos'] for d in labels_sorted],
        'neg': [stats[d]['neg'] for d in labels_sorted],
        'neu': [stats[d]['neu'] for d in labels_sorted]
    }

def get_ai_buzzwords(reviews):
    """Nutzt KI, um echte Problem-Cluster zu zählen statt nur Wörter."""
    if not model: return []

    # Wir nehmen die letzten 100 Reviews für die Analyse
    text_sample = [r['text'] for r in reviews[:120] if len(r.get('text','')) > 10]

    prompt = f"""
    Analysiere die folgenden App-Reviews.
    Identifiziere die 12 häufigsten spezifischen Probleme oder Themen (z.B. "Login Absturz", "Kein Download", "Werbung").
    Fasse Synonyme zusammen (z.B. "stürzt ab", "Absturz", "crash" -> "App-Abstürze").
    Zähle, wie oft jedes Thema vorkommt.
    
    Output MUSS valides JSON sein, eine Liste von Objekten:
    [
        {{"term": "App-Abstürze", "count": 12}},
        {{"term": "Login-Probleme", "count": 8}}
    ]
    
    Reviews: {json.dumps(text_sample, ensure_ascii=False)}
    """

    try:
        resp = model.generate_content(prompt)
        text = resp.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        # Konvertiere in Tupel Format für HTML [(term, count), ...]
        return [(item['term'], item['count']) for item in data if isinstance(item, dict)]
    except Exception as e:
        print(f"Buzzword KI Fehler: {e}")
        return []

def is_genuine_positive(review):
    """SMART FILTER: Prüft, ob ein positives Review versteckte negative Wörter enthält."""
    bad_words = ["absturz", "stürzt", "fehler", "schlecht", "katastrophe", "mies", "flackern", "unbrauchbar", "nicht möglich", "enttäuscht"]

    if review.get('rating', 0) < 4:
        return True # Bei schlechten Ratings ist Negatives okay

    text = review.get('text', '').lower()
    if any(word in text for word in bad_words):
        return False
    return True

# ---------------------------------------------------------
# 3. SCRAPING FUNKTIONEN
# ---------------------------------------------------------
def fetch_ios_reviews(app_name, app_id, country="de", count=20):
    """Holt iOS Reviews über die stabile RSS Schnittstelle."""
    print(f"   -> iOS (RSS): {app_name}...")
    api_url = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostrecent/json"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = []

        entries = data.get('feed', {}).get('entry', [])
        valid_entries = 0
        for entry in entries:
            if valid_entries >= count: break
            if 'im:rating' not in entry or 'content' not in entry: continue

            rating = int(entry['im:rating']['label'])
            text = entry['content']['label']
            raw_date = entry.get('updated', {}).get('label', datetime.now().strftime('%Y-%m-%d'))
            date_str = raw_date[:10]

            results.append({
                "store": "ios", "app": app_name, "rating": rating, "text": text, "date": date_str,
                "id": generate_id({'app': app_name, 'store': 'ios', 'date': date_str, 'text': text})
            })
            valid_entries += 1
        return results
    except Exception as e:
        print(f"      ❌ iOS Fehler: {e}")
        return []

def fetch_android_reviews(app_name, app_id, country="de", count=20):
    """Holt Android Reviews über den Scraper."""
    print(f"   -> Android: {app_name}...")
    try:
        result, _ = play_reviews(
            app_id, lang=country, country=country, sort=Sort.NEWEST, count=count
        )
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
    """Sammelt Daten von allen Quellen."""
    history_dict = load_history()
    new_reviews_list = []

    print(f"--- Starte Scrape für {len(APP_CONFIG)*2} Quellen ---")

    for app in APP_CONFIG:
        ios_data = fetch_ios_reviews(app['name'], app['ios_id'], app['country'], review_count)
        for r in ios_data:
            if r['id'] not in history_dict:
                history_dict[r['id']] = r
                new_reviews_list.append(r)

        and_data = fetch_android_reviews(app['name'], app['android_id'], app['country'], review_count)
        for r in and_data:
            if r['id'] not in history_dict:
                history_dict[r['id']] = r
                new_reviews_list.append(r)

    full_history = sorted(history_dict.values(), key=lambda x: x['date'], reverse=True)
    print(f"\n--- STATUS: {len(full_history)} Gesamt gespeichert, {len(new_reviews_list)} NEU gefunden ---")

    return full_history, new_reviews_list

# ---------------------------------------------------------
# 4. INTELLIGENZ: CLUSTERING & LABELING
# ---------------------------------------------------------
def get_semantic_topics(reviews, num_clusters=5):
    """Erstellt Themencluster aus den Review-Texten."""
    if not embedder or not model:
        return ["KI Module nicht geladen"]

    text_reviews = [r for r in reviews[:300] if len(r.get('text','')) > 15]
    if len(text_reviews) < 5:
        return ["Zu wenige Daten für Cluster"]

    texts = [r['text'] for r in text_reviews]
    embeddings = embedder.encode(texts)

    kmeans = KMeans(n_clusters=min(num_clusters, len(texts)), random_state=0, n_init=10).fit(embeddings)

    topic_samples = []
    for i in range(kmeans.n_clusters):
        idx = np.where(kmeans.labels_ == i)[0]
        if idx.size == 0: continue
        center = kmeans.cluster_centers_[i]
        closest_idx = idx[np.argmax(cosine_similarity([center], embeddings[idx]))]
        topic_samples.append(text_reviews[closest_idx])

    prompt_data = [{"text": r['text']} for r in topic_samples]
    prompt = f"""
    Du bist ein Produkt-Analyst. Erstelle für jedes dieser {len(topic_samples)} Reviews ein kurzes, prägnantes Schlagwort (max 2 Wörter, z.B. "Login Fehler", "Abstürze").
    Reviews: {json.dumps(prompt_data, ensure_ascii=False)}
    Antworte NUR mit einer JSON Liste von Strings. Beispiel: ["Login", "Performance"]
    """

    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        labels = json.loads(text)
        return [str(l) for l in labels if isinstance(l, (str, int))]
    except Exception as e:
        print(f"Labeling Fehler: {e}")
        return ["Analyse-Fehler"]

# ---------------------------------------------------------
# 5. DASHBOARD GENERATOR
# ---------------------------------------------------------
def run_analysis_and_generate_html(full_history, new_only):
    trends = calculate_trends(full_history)
    chart_data = prepare_chart_data(full_history)
    topics = get_semantic_topics(full_history)
    buzzwords = get_ai_buzzwords(full_history)

    ki_output = {"summary": "Keine Analyse verfügbar.", "topReviews": [], "bottomReviews": []}

    rich_reviews = [r for r in full_history if len(r.get('text', '')) > 40]
    if len(rich_reviews) < 10: rich_reviews = full_history

    if model and rich_reviews:
        print("--- Starte KI Deep Dive Analyse ---")
        analysis_subset = rich_reviews[:50]

        prompt_data = [{'text': r['text'], 'rating': r['rating'], 'store': r['store'], 'app': r['app']} for r in analysis_subset]

        prompt = f"""
        Analysiere diese App-Reviews.
        1. Schreibe ein Management Summary (Deutsch, ca 3-4 Sätze).
        2. Wähle 3 Top-Reviews (Positiv, 4-5 Sterne) und 3 Bottom-Reviews (Negativ, 1-2 Sterne).
        
        WICHTIG: Gib im JSON für jedes Review auch 'app', 'store' und 'rating' exakt so zurück, wie sie in den Daten stehen.
        
        Output JSON Format: 
        {{ 
            "summary": "Text...", 
            "topReviews": [{{...}}], 
            "bottomReviews": [{{...}}] 
        }}
        
        Data: {json.dumps(prompt_data, ensure_ascii=False)}
        """
        try:
            resp = model.generate_content(prompt)
            text = resp.text.replace("```json", "").replace("```", "").strip()
            ki_output.update(json.loads(text))
        except Exception as e:
            print(f"KI Fehler: {e}")

    # --- FALLBACK & INTELLIGENTE AUSWAHL (Smart Filter) ---

    top_list = []
    bot_list = []

    # 1. Kandidaten sammeln: Nur "echte" Positive (Anti-Widerspruch-Filter)
    genuine_positive_candidates = [r for r in full_history if r['rating'] >= 4 and is_genuine_positive(r)]
    best_sorted = sorted(genuine_positive_candidates, key=lambda x: len(x['text']), reverse=True)

    for r in best_sorted:
        if len(top_list) >= 3: break
        if r['text'] not in [x.get('text') for x in top_list]:
            top_list.append(r)

    # 2. Kandidaten Negativ
    worst_sorted = sorted([r for r in full_history if r['rating'] <= 2], key=lambda x: len(x['text']), reverse=True)
    for r in worst_sorted:
        if len(bot_list) >= 3: break
        if r['text'] not in [x.get('text') for x in bot_list]:
            bot_list.append(r)

    # KI-Ergebnisse als Backup nutzen, falls Python nicht genug findet
    if len(top_list) < 1: top_list = ki_output.get('topReviews', [])
    if len(bot_list) < 1: bot_list = ki_output.get('bottomReviews', [])

    # Metadaten auffüllen
    for r in top_list + bot_list:
        if not r.get('app'):
            orig = next((x for x in full_history if x['text'][:20] == r.get('text', '')[:20]), None)
            if orig: r.update({'app': orig['app'], 'store': orig['store'], 'rating': orig['rating']})

    # Summary bereinigen
    summary_text = str(ki_output.get('summary', '')).strip().replace('{','').replace('}','').replace('"','')

    # Datum formatieren
    for r in full_history:
        if 'date' in r:
            try: r['fmt_date'] = datetime.strptime(r['date'], '%Y-%m-%d').strftime('%d.%m.%Y')
            except: r['fmt_date'] = r['date']

    # JS Daten
    js_reviews = json.dumps(full_history, ensure_ascii=False)
    js_labels = json.dumps(chart_data['labels'])
    js_pos = json.dumps(chart_data['pos'])
    js_neg = json.dumps(chart_data['neg'])
    js_neu = json.dumps(chart_data['neu'])

    html = f"""
    <!DOCTYPE html>
    <html lang="de" data-theme="light">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>App Feedback Pro</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {{
                --bg: #f8fafc; --text: #1e293b; --card-bg: #ffffff; --border: #e2e8f0;
                --primary: #2563eb; --summary-bg: #eff6ff; --shadow: rgba(0,0,0,0.05);
                --ios-color: #000000; --android-color: #3DDC84;
                --grid-color: #e2e8f0; --mark-bg: #fef08a; --mark-text: #854d0e;
                --buzz-base: 220, 38, 38;
            }}
            [data-theme="dark"] {{
                --bg: #0f172a; --text: #e2e8f0; --card-bg: #1e293b; --border: #334155;
                --primary: #3b82f6; --summary-bg: #1e293b; --shadow: rgba(0,0,0,0.3);
                --ios-color: #ffffff; --grid-color: #334155; --mark-bg: #854d0e; --mark-text: #fef08a;
                --buzz-base: 248, 113, 113;
            }}
            body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }}
            .theme-btn {{ background: none; border: 1px solid var(--border); color: var(--text); padding: 8px 12px; border-radius: 8px; cursor: pointer; }}
            
            .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .card {{ background: var(--card-bg); padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px var(--shadow); border: 1px solid var(--border); }}
            .kpi-val {{ font-size: 2.5rem; font-weight: 700; color: var(--primary); margin-top: 10px; }}
            .kpi-label {{ font-size: 0.9rem; opacity: 0.8; text-transform: uppercase; font-weight: bold; }}
            
            .chart-container {{ background: var(--card-bg); padding: 20px; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 30px; height: 350px; }}
            .summary-box {{ background: var(--summary-bg); padding: 25px; border-radius: 12px; border-left: 5px solid var(--primary); margin-bottom: 30px; line-height: 1.6; border: 1px solid var(--border); }}
            
            .tag {{ display: inline-block; background: var(--card-bg); border: 1px solid var(--border); padding: 6px 14px; border-radius: 20px; margin: 0 8px 8px 0; font-size: 0.9rem; color: var(--text); }}
            
            /* NEW BUZZWORD DESIGN */
            .buzz-container {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-start; }}
            .buzz-tag {{ 
                display: inline-flex; align-items: center; 
                padding: 6px 12px; border-radius: 20px; 
                background-color: rgba(var(--buzz-base), calc(0.05 + var(--intensity) * 0.2));
                border: 1px solid rgba(var(--buzz-base), calc(0.2 + var(--intensity) * 0.5));
                color: var(--text); font-weight: 500; transition: transform 0.2s;
            }}
            .buzz-tag:hover {{ transform: scale(1.05); }}
            .buzz-tag .count {{ background: rgba(0,0,0,0.1); padding: 2px 6px; border-radius: 10px; font-size: 0.75em; margin-left: 8px; }}
            
            .review-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }}
            .review-card {{ background: var(--card-bg); padding: 20px; border-radius: 8px; border: 1px solid var(--border); display: flex; flex-direction: column; gap: 10px; box-shadow: 0 1px 3px var(--shadow); }}
            .review-card.pos {{ border-top: 4px solid #22c55e; }}
            .review-card.neg {{ border-top: 4px solid #ef4444; }}
            
            .icon-android {{ color: var(--android-color); }} .icon-ios {{ color: var(--ios-color); }}
            .copy-btn {{ cursor: pointer; float: right; opacity: 0.5; }} .copy-btn:hover {{ opacity: 1; color: var(--primary); }}
            
            .search-input {{ flex: 1; padding: 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 1rem; background: var(--card-bg); color: var(--text); }}
            .filter-group {{ display: flex; gap: 5px; align-items: center; }}
            .filter-label {{ font-size: 0.85rem; color: #64748b; text-transform: uppercase; font-weight: bold; margin-right: 5px; }}
            .filter-btn {{ padding: 8px 16px; border: 1px solid var(--border); background: var(--card-bg); color: var(--text); border-radius: 8px; cursor: pointer; font-size: 0.9rem; }}
            .filter-btn.active {{ background: var(--primary); color: white; border-color: var(--primary); }}
            
            .review-text {{ margin-top: 8px; line-height: 1.5; position: relative; }}
            .review-text mark {{ background-color: var(--mark-bg); color: var(--mark-text); padding: 0 2px; border-radius: 2px; }}
            .review-text.clamped {{ display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }}
            .read-more {{ color: var(--primary); cursor: pointer; font-size: 0.9rem; display: none; margin-top: 5px; font-weight: 600; }}
            .meta {{ font-size: 0.85rem; opacity: 0.7; display: flex; justify-content: space-between; align-items: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div>
                    <h1 style="margin:0;">📊 App Feedback Pro</h1>
                    <span style="opacity:0.7;">Update: {datetime.now().strftime('%d.%m.%Y %H:%M')}</span>
                </div>
                <button class="theme-btn" onclick="toggleTheme()">🌗</button>
            </header>

            <div class="kpi-grid">
                <div class="card">
                    <div class="kpi-label">Gesamt Ø</div>
                    <div class="kpi-val">{trends['overall']} <span style="font-size:1.5rem">⭐</span></div>
                </div>
                <div class="card">
                    <div class="kpi-label">Trend (7 Tage)</div>
                    <div class="kpi-val">{trends['last_7d']} <span style="font-size:1.5rem">⭐</span></div>
                </div>
                <div class="card">
                    <div class="kpi-label">Erfasste Reviews</div>
                    <div class="kpi-val">{len(full_history)}</div>
                </div>
            </div>

            <div class="chart-container">
                <h4 style="margin:0 0 15px 0; opacity:0.7;">Bewertungsverlauf (14 Tage)</h4>
                <canvas id="trendChart"></canvas>
            </div>
            
            <div class="summary-box">
                <h3 style="margin-top:0;">🤖 KI-Analyse</h3>
                <p>{summary_text}</p>
            </div>

            <div class="row" style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:40px;">
                <div class="col" style="flex:1;">
                    <h3 style="margin-bottom: 15px;">🔥 Themen-Cluster</h3>
                    {''.join([f'<span class="tag"># {t}</span> ' for t in topics])}
                </div>
                <div class="col" style="flex:1;">
                    <h3 style="margin-bottom: 15px;">🚨 Häufigste Probleme (KI)</h3>
                    <div class="card buzz-container">
                        {buzz_html}
                    </div>
                </div>
            </div>

            <div class="review-grid">
                <div>
                    <h3>👍 Top Stimmen</h3>
                    {''.join([f'''
                    <div class="review-card pos">
                        <div class="meta">
                            <span>{'<i class="fab fa-apple icon-ios"></i>' if r.get('store')=='ios' else '<i class="fab fa-android icon-android"></i>'} <strong>{r.get('app')}</strong></span>
                            <span>{r.get('rating')}★</span>
                        </div>
                        <div class="review-content">
                            <div class="review-text clamped">{r.get('text')}</div>
                            <span class="read-more" onclick="toggleText(this)">Mehr anzeigen</span>
                        </div>
                    </div>''' for r in top_list[:3]])}
                </div>
                <div>
                    <h3>⚠️ Kritische Stimmen</h3>
                    {''.join([f'''
                    <div class="review-card neg">
                        <div class="meta">
                            <span>{'<i class="fab fa-apple icon-ios"></i>' if r.get('store')=='ios' else '<i class="fab fa-android icon-android"></i>'} <strong>{r.get('app')}</strong></span>
                            <span>{r.get('rating')}★</span>
                        </div>
                        <div class="review-content">
                            <div class="review-text clamped">{r.get('text')}</div>
                            <span class="read-more" onclick="toggleText(this)">Mehr anzeigen</span>
                        </div>
                    </div>''' for r in bot_list[:3]])}
                </div>
            </div>

            <h2 style="border-top: 1px solid var(--border); padding-top: 30px;">🔎 Explorer</h2>
            
            <div style="display:flex; gap:15px; margin-bottom:20px; flex-wrap:wrap; align-items: flex-end;">
                <div style="flex:1; min-width: 300px;">
                    <input type="text" class="search-input" id="search" placeholder="Suche nach Stichworten..." onkeyup="filterData()" style="width:100%; box-sizing:border-box;">
                </div>
                
                <div class="filter-group">
                    <span class="filter-label">App:</span>
                    <button class="filter-btn active" onclick="setFilter('all', this)">Alle</button>
                    <button class="filter-btn" onclick="setFilter('Nordkurier', this)">Nordkurier</button>
                    <button class="filter-btn" onclick="setFilter('Schwäbische', this)">Schwäbische</button>
                </div>

                <div class="filter-group">
                    <span class="filter-label">Sortierung:</span>
                    <button class="filter-btn" onclick="setSort('newest', this)">Neueste</button>
                    <button class="filter-btn" onclick="setSort('best', this)">Beste</button>
                    <button class="filter-btn" onclick="setSort('worst', this)">Schlechteste</button>
                </div>
            </div>

            <div id="list-container" style="display:grid; gap:15px;"></div>
        </div>

        <script>
            const REVIEWS = {js_reviews};
            let currentFilter = 'all';
            let currentSort = 'newest';

            // Theme Init
            const savedTheme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            
            function toggleTheme() {{
                const newTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                updateChartColors();
            }}

            // Chart Setup
            const ctx = document.getElementById('trendChart').getContext('2d');
            let chart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: {js_labels},
                    datasets: [
                        {{ label: 'Positiv (4-5★)', data: {js_pos}, backgroundColor: '#22c55e' }},
                        {{ label: 'Neutral (3★)', data: {js_neu}, backgroundColor: '#94a3b8' }},
                        {{ label: 'Negativ (1-2★)', data: {js_neg}, backgroundColor: '#ef4444' }}
                    ]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{ 
                        x: {{ stacked: true, grid: {{ display: false }} }}, 
                        y: {{ stacked: true, grid: {{ color: '#e2e8f0' }} }} 
                    }},
                    plugins: {{ legend: {{ position: 'bottom' }} }}
                }}
            }});
            
            function updateChartColors() {{
                const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
                chart.options.scales.y.grid.color = isDark ? '#334155' : '#e2e8f0';
                chart.update();
            }}
            updateChartColors();

            // Init Defaults
            document.querySelectorAll('.filter-group:last-child .filter-btn')[0].classList.add('active');

            // Read More Logic
            function initReadMore() {{
                document.querySelectorAll('.review-content').forEach(div => {{
                    const text = div.querySelector('.review-text');
                    const btn = div.querySelector('.read-more');
                    if (text.scrollHeight > text.clientHeight) {{
                        btn.style.display = 'inline-block';
                    }}
                }});
            }}
            initReadMore();

            function toggleText(btn) {{
                const text = btn.previousElementSibling;
                text.classList.toggle('clamped');
                btn.innerText = text.classList.contains('clamped') ? 'Mehr anzeigen' : 'Weniger anzeigen';
            }}

            // Filter & Sort
            function setFilter(app, btn) {{
                currentFilter = app;
                const group = btn.parentElement;
                group.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                filterData();
            }}
            
            function setSort(mode, btn) {{
                currentSort = mode;
                const group = btn.parentElement;
                group.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                filterData();
            }}

            function copyText(text) {{ navigator.clipboard.writeText(text); alert('Text kopiert!'); }}

            function filterData() {{
                const q = document.getElementById('search').value.toLowerCase().trim();
                const container = document.getElementById('list-container');
                container.innerHTML = '';

                let filtered = REVIEWS.filter(r => {{
                    return (currentFilter === 'all' || r.app === currentFilter) && (r.text + r.store).toLowerCase().includes(q);
                }});

                if (currentSort === 'newest') filtered.sort((a, b) => a.date < b.date ? 1 : -1);
                if (currentSort === 'best') filtered.sort((a, b) => b.rating - a.rating);
                if (currentSort === 'worst') filtered.sort((a, b) => a.rating - b.rating);

                if (filtered.length === 0) {{ container.innerHTML = '<div style="text-align:center;opacity:0.5;padding:20px;">Keine Ergebnisse</div>'; return; }}

                filtered.slice(0, 50).forEach(r => {{
                    const icon = r.store === 'ios' ? '<i class="fab fa-apple icon-ios"></i>' : '<i class="fab fa-android icon-android"></i>';
                    
                    // HIGHLIGHTING LOGIC (JS Syntax Fix)
                    let displayText = r.text;
                    if (q.length >= 2) {{
                        // Einfache String-Verkettung um Python-Fehler zu vermeiden
                        const regex = new RegExp('(' + q + ')', 'gi');
                        displayText = displayText.replace(regex, '<mark>$1</mark>');
                    }}
                    
                    const div = document.createElement('div');
                    div.className = 'review-card';
                    div.innerHTML = `
                        <div style="display:flex; justify-content:space-between; opacity:0.8; font-size:0.9rem; border-bottom:1px solid var(--border); padding-bottom:8px; margin-bottom:8px;">
                            <span style="display:flex; align-items:center; gap:6px;">
                                ${{icon}} <strong>${{r.app}}</strong> • ${{r.rating}}⭐
                            </span>
                            <span>${{r.fmt_date || r.date}}</span>
                        </div>
                        <div style="line-height:1.5;">
                            <span class="review-text">${{displayText}}</span>
                            <i class="fas fa-copy copy-btn" onclick="copyText('${{r.text.replace(/'/g, "\\'")}}')"></i>
                        </div>
                    `;
                    container.appendChild(div);
                }});
            }}
            
            filterData();
        </script>
    </body>
    </html>
    """

    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ Dashboard HTML erfolgreich generiert.")

# ---------------------------------------------------------
# 7. MAIN EXECUTION
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

if __name__ == "__main__":
    full, new = get_fresh_reviews()
    save_history({r['id']: r for r in full})
    run_analysis_and_generate_html(full, new)

    teams = os.getenv("TEAMS_WEBHOOK_URL")
    if teams: send_teams_notification(new, teams)

    print("✅ Durchlauf beendet.")