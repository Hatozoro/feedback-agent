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

DATA_FILE = "data/reviews_history.json"
APP_CONFIG = [
    {"name": "Nordkurier", "ios_id": "1250964862", "android_id": "de.nordkurier.live", "country": "de"},
    {"name": "Schwäbische", "ios_id": "432491155", "android_id": "de.schwaebische.epaper", "country": "de"}
]

# ---------------------------------------------------------
# 2. HILFSFUNKTIONEN
# ---------------------------------------------------------
def generate_id(review):
    unique_string = f"{review.get('text', '')[:50]}{review.get('date', '')}{review.get('app', '')}{review.get('store', '')}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

def load_history():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                return {r['id']: r for r in raw_data if 'id' in r}
        except json.JSONDecodeError:
            pass
    return {}

def save_history(history_dict):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    data_list = sorted(history_dict.values(), key=lambda x: x['date'], reverse=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

# NEU: Berechnung für den Chart (Tägliche Durchschnitte)
def prepare_chart_data(reviews, days=14):
    today = datetime.now().date()
    daily_stats = {}

    # Initialisiere letzte X Tage mit 0
    for i in range(days):
        date_key = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        daily_stats[date_key] = {'sum': 0, 'count': 0}

    for r in reviews:
        d = r['date']
        if d in daily_stats:
            daily_stats[d]['sum'] += float(r['rating'])
            daily_stats[d]['count'] += 1

    # Daten für Chart.js formatieren
    labels = sorted(daily_stats.keys())
    data_points = []
    for date in labels:
        stats = daily_stats[date]
        avg = round(stats['sum'] / stats['count'], 2) if stats['count'] > 0 else None
        data_points.append(avg)

    return {'labels': labels, 'data': data_points}

def calculate_kpis(reviews):
    """Berechnet statische KPIs."""
    if not reviews: return {'overall': 0, 'week': 0}

    today = datetime.now().date()
    week_ago = today - timedelta(days=7)

    ratings = [float(r['rating']) for r in reviews]
    overall = round(sum(ratings) / len(ratings), 2)

    week_ratings = []
    for r in reviews:
        try:
            if datetime.strptime(r['date'], '%Y-%m-%d').date() >= week_ago:
                week_ratings.append(float(r['rating']))
        except: continue

    week_avg = round(sum(week_ratings) / len(week_ratings), 2) if week_ratings else overall

    return {'overall': overall, 'week': week_avg}

# ---------------------------------------------------------
# 3. SCRAPING
# ---------------------------------------------------------
def fetch_ios_reviews(app_name, app_id, country="de", count=20):
    print(f"   -> iOS (RSS): {app_name}...")
    api_url = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostrecent/json"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = []
        for entry in data.get('feed', {}).get('entry', [])[:count]:
            if 'im:rating' not in entry or 'content' not in entry: continue
            results.append({
                "store": "ios", "app": app_name,
                "rating": int(entry['im:rating']['label']),
                "text": entry['content']['label'],
                "date": entry.get('updated', {}).get('label', datetime.now().strftime('%Y-%m-%d'))[:10],
                "id": generate_id({'app': app_name, 'store': 'ios', 'text': entry['content']['label']}) # Simplified ID gen call
            })
        return results
    except Exception as e:
        print(f"      ❌ iOS Fehler: {e}")
        return []

def fetch_android_reviews(app_name, app_id, country="de", count=20):
    print(f"   -> Android: {app_name}...")
    try:
        result, _ = play_reviews(app_id, lang=country, country=country, sort=Sort.NEWEST, count=count)
        reviews = []
        for r in result:
            d = r['at'].strftime('%Y-%m-%d')
            reviews.append({
                "store": "android", "app": app_name, "rating": r['score'], "text": r['content'], "date": d,
                "id": generate_id({'app': app_name, 'store': 'android', 'text': r['content']})
            })
        return reviews
    except Exception as e:
        print(f"      ❌ Android Fehler: {e}")
        return []

def get_fresh_reviews(review_count=20):
    history = load_history()
    new_list = []
    for app in APP_CONFIG:
        for r in fetch_ios_reviews(app['name'], app['ios_id'], app['country'], review_count) + \
                 fetch_android_reviews(app['name'], app['android_id'], app['country'], review_count):
            if r['id'] not in history:
                history[r['id']] = r
                new_list.append(r)

    full_list = sorted(history.values(), key=lambda x: x['date'], reverse=True)
    print(f"\n--- STATUS: {len(full_list)} Gesamt, {len(new_list)} NEU ---")
    return full_list, new_list

# ---------------------------------------------------------
# 4. KI & CLUSTERING
# ---------------------------------------------------------
def get_semantic_topics(reviews, num_clusters=5):
    if not embedder: return ["KI nicht bereit"]
    text_reviews = [r for r in reviews[:200] if len(r.get('text', '')) > 15]
    if len(text_reviews) < num_clusters: return ["Zu wenige Daten"]

    texts = [r['text'] for r in text_reviews]
    embeddings = embedder.encode(texts)

    kmeans = KMeans(n_clusters=min(num_clusters, len(texts)), random_state=0, n_init=10).fit(embeddings)

    topic_reviews = []
    for i in range(kmeans.n_clusters):
        cluster_idx = np.where(kmeans.labels_ == i)[0]
        if not cluster_idx.size: continue
        center = kmeans.cluster_centers_[i]
        closest = cluster_idx[np.argmax(cosine_similarity([center], embeddings[cluster_idx]))]
        topic_reviews.append(text_reviews[closest])

    if not model: return ["KI-Service fehlt"]

    prompt = f"""
    Erstelle kurze, prägnante Labels (max 2 Wörter) für diese Themen-Cluster.
    Output nur als JSON Liste von Strings.
    Reviews: {json.dumps([{'text': r['text']} for r in topic_reviews], ensure_ascii=False)}
    """
    try:
        resp = model.generate_content(prompt)
        return [t for t in json.loads(resp.text.replace("```json", "").replace("```", "").strip()) if isinstance(t, str)]
    except: return ["Analyse Fehler"]

# ---------------------------------------------------------
# 5. HTML GENERIERUNG (PRO VERSION)
# ---------------------------------------------------------
def run_analysis_and_generate_html(full_history, new_only):
    kpis = calculate_kpis(full_history)
    chart_data = prepare_chart_data(full_history)
    topics = get_semantic_topics(full_history)

    # KI Analyse
    ki_output = {"summary": "Keine Analyse.", "topReviews": [], "bottomReviews": []}
    analysis_set = full_history[:50]
    if model and analysis_set:
        print("--- Starte KI Deep Dive ---")
        prompt = f"""
        Analysiere diese App-Reviews.
        1. Schreibe ein Management Summary (Deutsch).
        2. Wähle 3 Top-Reviews (Positiv) und 3 Bottom-Reviews (Negativ).
        Output JSON: {{ "summary": "...", "topReviews": [{{...}}], "bottomReviews": [{{...}}] }}
        Data: {json.dumps([{'text': r['text'], 'rating': r['rating'], 'store': r['store']} for r in analysis_set], ensure_ascii=False)}
        """
        try:
            resp = model.generate_content(prompt)
            ki_output.update(json.loads(resp.text.replace("```json", "").replace("```", "").strip()))
        except: pass

    # Datenaufbereitung für JS
    js_reviews = json.dumps(full_history, ensure_ascii=False)
    js_chart_labels = json.dumps(chart_data['labels'])
    js_chart_values = json.dumps(chart_data['data'])

    # Robustes Summary
    summary_text = ki_output.get('summary', '')
    if isinstance(summary_text, (dict, list)): summary_text = str(summary_text)
    summary_text = summary_text.strip().replace('{', '').replace('}', '')

    html = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>App Feedback Pro</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {{ --primary: #2563eb; --bg: #f8fafc; --card: #ffffff; --text: #1e293b; }}
            body {{ font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            
            /* Header & KPIs */
            header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .card {{ background: var(--card); padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }}
            .kpi-val {{ font-size: 2.5rem; font-weight: 700; color: var(--primary); }}
            .kpi-label {{ color: #64748b; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; }}
            
            /* Chart Section */
            .chart-container {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 30px; height: 300px; }}
            
            /* Summary & Topics */
            .summary-box {{ background: #eff6ff; padding: 25px; border-radius: 12px; border-left: 5px solid var(--primary); margin-bottom: 30px; line-height: 1.6; }}
            .tag {{ display: inline-block; background: white; border: 1px solid #cbd5e1; padding: 6px 14px; border-radius: 20px; margin: 0 8px 8px 0; font-size: 0.9rem; color: #475569; font-weight: 500; }}
            
            /* Reviews Grid */
            .review-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }}
            @media(max-width: 768px) {{ .review-grid {{ grid-template-columns: 1fr; }} }}
            .review-card {{ background: white; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; display: flex; flex-direction: column; gap: 10px; }}
            .review-card.pos {{ border-top: 4px solid #22c55e; }}
            .review-card.neg {{ border-top: 4px solid #ef4444; }}
            .meta {{ display: flex; justify-content: space-between; font-size: 0.85rem; color: #64748b; }}
            
            /* Search & Filter */
            .controls {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
            .search-input {{ flex: 1; padding: 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 1rem; }}
            .filter-btn {{ padding: 10px 20px; border: 1px solid #cbd5e1; background: white; border-radius: 8px; cursor: pointer; font-weight: 500; transition: all 0.2s; }}
            .filter-btn:hover, .filter-btn.active {{ background: var(--primary); color: white; border-color: var(--primary); }}
            
            .icon-ios {{ color: #1C1C1E; }}
            .icon-android {{ color: #3DDC84; }}
            .copy-btn {{ cursor: pointer; color: #94a3b8; float: right; }}
            .copy-btn:hover {{ color: var(--primary); }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div>
                    <h1 style="margin:0;">📊 App Feedback Pro</h1>
                    <span style="color: #64748b; font-size: 0.9rem;">Letztes Update: {datetime.now().strftime('%d.%m.%Y %H:%M')}</span>
                </div>
            </header>

            <div class="kpi-grid">
                <div class="card">
                    <div class="kpi-label">Gesamt Ø</div>
                    <div class="kpi-val">{kpis['overall']} <span style="font-size:1.5rem">⭐</span></div>
                </div>
                <div class="card">
                    <div class="kpi-label">Trend (7 Tage)</div>
                    <div class="kpi-val">{kpis['week']} <span style="font-size:1.5rem">⭐</span></div>
                </div>
                <div class="card">
                    <div class="kpi-label">Total Reviews</div>
                    <div class="kpi-val">{len(full_history)}</div>
                </div>
            </div>

            <div class="chart-container">
                <canvas id="trendChart"></canvas>
            </div>

            <div class="summary-box">
                <h3 style="margin-top:0;">🤖 KI-Analyse</h3>
                {summary_text}
            </div>

            <div style="margin-bottom: 40px;">
                <h3 style="margin-bottom: 15px;">🔥 Trending Topics</h3>
                {''.join([f'<span class="tag"># {t}</span>' for t in topics])}
            </div>

            <div class="review-grid">
                <div>
                    <h3>👍 Top Stimmen</h3>
                    {''.join([f'''
                    <div class="review-card pos">
                        <div class="meta">
                            <span>{'<i class="fab fa-apple icon-ios"></i>' if r.get('store')=='ios' else '<i class="fab fa-android icon-android"></i>'} {r.get('rating')}★</span>
                            <span>{r.get('app')}</span>
                        </div>
                        <div style="font-style:italic">"{r.get('text')}"</div>
                    </div>''' for r in ki_output.get('topReviews', [])[:3]])}
                </div>
                <div>
                    <h3>⚠️ Kritische Stimmen</h3>
                    {''.join([f'''
                    <div class="review-card neg">
                        <div class="meta">
                            <span>{'<i class="fab fa-apple icon-ios"></i>' if r.get('store')=='ios' else '<i class="fab fa-android icon-android"></i>'} {r.get('rating')}★</span>
                            <span>{r.get('app')}</span>
                        </div>
                        <div style="font-style:italic">"{r.get('text')}"</div>
                    </div>''' for r in ki_output.get('bottomReviews', [])[:3]])}
                </div>
            </div>

            <h2 style="border-top: 1px solid #e2e8f0; padding-top: 30px;">🔎 Review Explorer</h2>
            
            <div class="controls">
                <input type="text" class="search-input" id="search" placeholder="Suche nach Stichworten..." onkeyup="filterData()">
                <button class="filter-btn active" onclick="setFilter('all', this)">Alle Apps</button>
                <button class="filter-btn" onclick="setFilter('Nordkurier', this)">Nordkurier</button>
                <button class="filter-btn" onclick="setFilter('Schwäbische', this)">Schwäbische</button>
            </div>

            <div id="list-container" style="display:grid; gap:15px;"></div>

        </div>

        <script>
            const REVIEWS = {js_reviews};
            let currentFilter = 'all';

            // Chart Setup
            const ctx = document.getElementById('trendChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {js_chart_labels},
                    datasets: [{{
                        label: 'Durchschnittsbewertung (Täglich)',
                        data: {js_chart_values},
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        tension: 0.3,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{ y: {{ min: 1, max: 5 }} }}
                }}
            }});

            function setFilter(app, btn) {{
                currentFilter = app;
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                filterData();
            }}

            function copyText(text) {{
                navigator.clipboard.writeText(text);
                alert('Review kopiert!');
            }}

            function filterData() {{
                const q = document.getElementById('search').value.toLowerCase();
                const container = document.getElementById('list-container');
                container.innerHTML = '';

                const filtered = REVIEWS.filter(r => {{
                    const matchesApp = currentFilter === 'all' || r.app === currentFilter;
                    const matchesSearch = (r.text + r.store).toLowerCase().includes(q);
                    return matchesApp && matchesSearch;
                }});

                if (filtered.length === 0) {{
                    container.innerHTML = '<div style="text-align:center; color:#94a3b8; padding:20px;">Keine Ergebnisse gefunden.</div>';
                    return;
                }}

                filtered.slice(0, 50).forEach(r => {{
                    const icon = r.store === 'ios' ? '<i class="fab fa-apple icon-ios"></i>' : '<i class="fab fa-android icon-android"></i>';
                    const stars = '⭐'.repeat(r.rating);
                    
                    const div = document.createElement('div');
                    div.className = 'review-card';
                    div.innerHTML = `
                        <div class="meta">
                            <span style="display:flex; align-items:center; gap:8px;">
                                ${{icon}} <strong>${{r.app}}</strong> • ${{stars}} • ${{r.date}}
                            </span>
                            <i class="fas fa-copy copy-btn" title="Text kopieren" onclick="copyText('${{r.text.replace(/'/g, "\\'")}}')"></i>
                        </div>
                        <div style="line-height:1.5; color:#334155;">${{r.text}}</div>
                    `;
                    container.appendChild(div);
                }});
            }}

            // Init
            filterData();
        </script>
    </body>
    </html>
    """

    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f: f.write(html)
    print("✅ Pro-Dashboard generiert.")

# ---------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    full_history, new_reviews = get_fresh_reviews()
    save_history({r['id']: r for r in full_history})
    run_analysis_and_generate_html(full_history, new_reviews)

    teams_url = os.getenv("TEAMS_WEBHOOK_URL")
    if teams_url and new_reviews:
        pos = sum(1 for r in new_reviews if r['rating']>=4)
        neg = sum(1 for r in new_reviews if r['rating']<=2)
        txt = f"🚀 **FEEDBACK UPDATE** ({len(new_reviews)})\n\n👍 {pos} | 🚨 {neg}\n\n"
        for r in new_reviews[:3]: txt += f"- {r['rating']}★ {r['app']}: {r['text'][:50]}...\n"
        txt += "\n[Zum Pro-Dashboard](https://Hatozoro.github.io/feedback-agent/)"
        try: requests.post(teams_url, json={"text": txt})
        except: pass

    print("✅ Fertig.")