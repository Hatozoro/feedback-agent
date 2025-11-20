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

def calculate_trends(reviews):
    today = datetime.now().date()
    dated_reviews = []
    for r in reviews:
        try:
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
        return round(sum(filtered) / len(filtered), 2) if filtered else 0.0

    overall_avg = round(sum(r['rating'] for r in reviews) / len(reviews), 2)
    return {
        'overall': overall_avg,
        'last_7d': get_avg_for_days(7),
        'last_30d': get_avg_for_days(30)
    }

# NEU: Bereitet Daten für 2 Achsen vor (Rating UND Anzahl)
def prepare_chart_data(reviews, days=14):
    today = datetime.now().date()
    daily_stats = {}
    # Letzte X Tage initialisieren
    for i in range(days):
        date_key = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        daily_stats[date_key] = {'sum': 0, 'count': 0}

    for r in reviews:
        d = r['date']
        if d in daily_stats:
            daily_stats[d]['sum'] += float(r['rating'])
            daily_stats[d]['count'] += 1

    labels = sorted(daily_stats.keys())
    ratings = []
    counts = []

    for date in labels:
        stats = daily_stats[date]
        avg = round(stats['sum'] / stats['count'], 2) if stats['count'] > 0 else None
        ratings.append(avg)
        counts.append(stats['count'])

    return {'labels': labels, 'ratings': ratings, 'counts': counts}

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
            rating = int(entry['im:rating']['label'])
            text = entry['content']['label']
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
    print(f"   -> Android: {app_name}...")
    try:
        result, _ = play_reviews(app_id, lang=country, country=country, sort=Sort.NEWEST, count=count)
        reviews = []
        for r in result:
            d = r['at'].strftime('%Y-%m-%d')
            reviews.append({
                "store": "android", "app": app_name, "rating": r['score'], "text": r['content'],
                "date": d,
                "id": generate_id({'app': app_name, 'store': 'android', 'date': d, 'text': r['content']})
            })
        return reviews
    except Exception as e:
        print(f"      ❌ Android Fehler: {e}")
        return []

def get_fresh_reviews(review_count=20):
    history_dict = load_history()
    new_reviews_list = []
    print(f"--- Starte Scrape für {len(APP_CONFIG)*2} Quellen ---")
    for app in APP_CONFIG:
        for r in fetch_ios_reviews(app['name'], app['ios_id'], app['country'], review_count) + \
                 fetch_android_reviews(app['name'], app['android_id'], app['country'], review_count):
            if r['id'] not in history_dict:
                history_dict[r['id']] = r
                new_reviews_list.append(r)
    full_history = sorted(history_dict.values(), key=lambda x: x['date'], reverse=True)
    print(f"\n--- STATUS: {len(full_history)} Gesamt, {len(new_reviews_list)} NEU ---")
    return full_history, new_reviews_list

# ---------------------------------------------------------
# 4. CLUSTERING
# ---------------------------------------------------------
def get_semantic_topics(reviews, num_clusters=5):
    if not embedder: return ["KI nicht bereit"]

    text_reviews = [r for r in reviews[:300] if len(r.get('text', '')) > 10]
    if len(text_reviews) < num_clusters: return ["Zu wenige Daten für Cluster"]

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
    Erstelle für jedes dieser {len(topic_reviews)} repräsentativen Reviews ein kurzes Schlagwort (max 2 Wörter, z.B. "Login Probleme", "Abstürze").
    Antworte NUR mit einer JSON Liste von Strings.
    Reviews: {json.dumps([r['text'][:100] for r in topic_reviews], ensure_ascii=False)}
    """
    try:
        resp = model.generate_content(prompt)
        text = resp.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        if isinstance(data, dict): data = list(data.values())
        return [str(t) for t in data if t]
    except Exception as e:
        print(f"Labeling Fehler: {e}")
        return ["Technische Probleme", "Login", "App-Qualität", "Inhalte", "Sonstiges"]

# ---------------------------------------------------------
# 5. HTML GENERIERUNG (VERSION 4.0)
# ---------------------------------------------------------
def run_analysis_and_generate_html(full_history, new_only):
    trend_metrics = calculate_trends(full_history)
    chart_data = prepare_chart_data(full_history)
    topics = get_semantic_topics(full_history)

    ki_output = {"summary": "Keine Analyse.", "topReviews": [], "bottomReviews": []}
    analysis_set = full_history[:50]
    if model and analysis_set:
        print("--- Starte KI Deep Dive ---")
        # FIX: Wir übergeben den App-Namen explizit im Prompt-Daten-Objekt
        prompt_data = [{'text': r['text'], 'rating': r['rating'], 'store': r['store'], 'app': r['app']} for r in analysis_set]

        prompt = f"""
        Analysiere diese App-Reviews.
        1. Schreibe ein Management Summary (Deutsch).
        2. Wähle 3 Top-Reviews (Positiv) und 3 Bottom-Reviews (Negativ).
        
        WICHTIG: Gib im JSON für jedes Review auch 'app', 'store' und 'rating' exakt so zurück, wie sie in den Daten stehen.
        
        Output JSON: {{ "summary": "...", "topReviews": [{{...}}], "bottomReviews": [{{...}}] }}
        Data: {json.dumps(prompt_data, ensure_ascii=False)}
        """
        try:
            resp = model.generate_content(prompt)
            text = resp.text.replace("```json", "").replace("```", "").strip()
            ki_output.update(json.loads(text))
        except: pass

    # Fallback, falls KI Felder leer lässt oder 'app' vergisst: Wir füllen Daten aus der Historie auf
    # Dies behebt das "None" Problem
    for review_list in [ki_output.get('topReviews', []), ki_output.get('bottomReviews', [])]:
        for r in review_list:
            if not r.get('app') or r.get('app') == 'None':
                # Versuch das Review im Original-Set zu finden
                match = next((x for x in analysis_set if x['text'][:20] == r.get('text', '')[:20]), None)
                if match:
                    r['app'] = match['app']
                    r['store'] = match['store']
                    r['rating'] = match['rating']

    # Datenaufbereitung für JS
    js_reviews = json.dumps(full_history, ensure_ascii=False)
    js_chart_labels = json.dumps(chart_data['labels'])
    js_chart_ratings = json.dumps(chart_data['ratings'])
    js_chart_counts = json.dumps(chart_data['counts'])

    summary_text = ki_output.get('summary', '')
    if isinstance(summary_text, (dict, list)): summary_text = str(summary_text)
    summary_text = summary_text.strip().replace('{', '').replace('}', '')

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
                --grid-color: #e2e8f0;
            }}
            [data-theme="dark"] {{
                --bg: #0f172a; --text: #e2e8f0; --card-bg: #1e293b; --border: #334155;
                --primary: #3b82f6; --summary-bg: #1e293b; --shadow: rgba(0,0,0,0.3);
                --ios-color: #ffffff; --grid-color: #334155;
            }}
            body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }}
            .theme-btn {{ background: none; border: 1px solid var(--border); color: var(--text); padding: 8px; border-radius: 8px; cursor: pointer; }}
            
            .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .card {{ background: var(--card-bg); padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px var(--shadow); border: 1px solid var(--border); }}
            .kpi-val {{ font-size: 2.5rem; font-weight: 700; color: var(--primary); }}
            .chart-container {{ background: var(--card-bg); padding: 20px; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 30px; height: 350px; }}
            .summary-box {{ background: var(--summary-bg); padding: 25px; border-radius: 12px; border-left: 5px solid var(--primary); margin-bottom: 30px; line-height: 1.6; border: 1px solid var(--border); }}
            .tag {{ display: inline-block; background: var(--card-bg); border: 1px solid var(--border); padding: 6px 14px; border-radius: 20px; margin: 0 8px 8px 0; font-size: 0.9rem; color: var(--text); }}
            
            .review-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }}
            .review-card {{ background: var(--card-bg); padding: 20px; border-radius: 8px; border: 1px solid var(--border); display: flex; flex-direction: column; gap: 10px; box-shadow: 0 1px 3px var(--shadow); }}
            .review-card.pos {{ border-top: 4px solid #22c55e; }}
            .review-card.neg {{ border-top: 4px solid #ef4444; }}
            
            /* Icons Color Fix */
            .icon-android {{ color: var(--android-color); }}
            .icon-ios {{ color: var(--ios-color); }}

            .search-input {{ flex: 1; padding: 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 1rem; background: var(--card-bg); color: var(--text); }}
            .filter-btn {{ padding: 8px 16px; border: 1px solid var(--border); background: var(--card-bg); color: var(--text); border-radius: 8px; cursor: pointer; font-size: 0.9rem; margin-right: 5px; }}
            .filter-btn.active {{ background: var(--primary); color: white; border-color: var(--primary); }}
            .copy-btn {{ cursor: pointer; float: right; opacity: 0.5; }}
            .copy-btn:hover {{ opacity: 1; color: var(--primary); }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div><h1 style="margin:0;">📊 App Feedback Pro</h1><span style="opacity:0.7;">Update: {datetime.now().strftime('%d.%m.%Y %H:%M')}</span></div>
                <button class="theme-btn" onclick="toggleTheme()">🌗</button>
            </header>

            <div class="kpi-grid">
                <div class="card"><div>Gesamt Ø</div><div class="kpi-val">{trend_metrics['overall']} <span style="font-size:1.5rem">⭐</span></div></div>
                <div class="card"><div>Trend (7 Tage)</div><div class="kpi-val">{trend_metrics['last_7d']} <span style="font-size:1.5rem">⭐</span></div></div>
                <div class="card"><div>Total Reviews</div><div class="kpi-val">{len(full_history)}</div></div>
            </div>

            <div class="chart-container"><canvas id="trendChart"></canvas></div>
            <div class="summary-box"><h3 style="margin-top:0;">🤖 KI-Analyse</h3>{summary_text}</div>
            <div style="margin-bottom: 40px;"><h3 style="margin-bottom: 15px;">🔥 Trending Topics</h3>{''.join([f'<span class="tag"># {t}</span>' for t in topics])}</div>

            <div class="review-grid">
                <div><h3>👍 Top Stimmen</h3>{''.join([f'''
                    <div class="review-card pos">
                        <div style="opacity:0.7; font-size:0.9rem;">
                            <span>{'<i class="fab fa-apple icon-ios"></i>' if r.get('store')=='ios' else '<i class="fab fa-android icon-android"></i>'} {r.get('rating')}★ {r.get('app')}</span>
                        </div>
                        <div style="font-style:italic">"{r.get('text')}"</div>
                    </div>''' for r in ki_output.get('topReviews', [])[:3]])}
                </div>
                <div><h3>⚠️ Kritische Stimmen</h3>{''.join([f'''
                    <div class="review-card neg">
                        <div style="opacity:0.7; font-size:0.9rem;">
                            <span>{'<i class="fab fa-apple icon-ios"></i>' if r.get('store')=='ios' else '<i class="fab fa-android icon-android"></i>'} {r.get('rating')}★ {r.get('app')}</span>
                        </div>
                        <div style="font-style:italic">"{r.get('text')}"</div>
                    </div>''' for r in ki_output.get('bottomReviews', [])[:3]])}
                </div>
            </div>

            <h2 style="border-top: 1px solid var(--border); padding-top: 30px;">🔎 Explorer</h2>
            
            <div style="display:flex; gap:10px; margin-bottom:10px; flex-wrap:wrap;">
                <input type="text" class="search-input" id="search" placeholder="Suchen..." onkeyup="filterData()">
            </div>
            <div style="margin-bottom:20px;">
                <button class="filter-btn active" onclick="setFilter('all', this)">Alle Apps</button>
                <button class="filter-btn" onclick="setFilter('Nordkurier', this)">Nordkurier</button>
                <button class="filter-btn" onclick="setFilter('Schwäbische', this)">Schwäbische</button>
                <span style="margin: 0 10px; color:var(--border);">|</span>
                <button class="filter-btn" onclick="setSort('newest', this)">Neueste</button>
                <button class="filter-btn" onclick="setSort('best', this)">Beste</button>
                <button class="filter-btn" onclick="setSort('worst', this)">Schlechteste</button>
            </div>

            <div id="list-container" style="display:grid; gap:15px;"></div>
        </div>

        <script>
            const REVIEWS = {js_reviews};
            let currentFilter = 'all';
            let currentSort = 'newest';

            // Theme
            const savedTheme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            
            function toggleTheme() {{
                const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-theme', next);
                localStorage.setItem('theme', next);
                updateChartColors();
            }}

            // Chart
            const ctx = document.getElementById('trendChart').getContext('2d');
            let chart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: {js_chart_labels},
                    datasets: [
                        {{
                            type: 'line',
                            label: 'Ø Bewertung',
                            data: {js_chart_ratings},
                            borderColor: '#2563eb',
                            borderWidth: 3,
                            tension: 0.4,
                            yAxisID: 'y'
                        }},
                        {{
                            type: 'bar',
                            label: 'Anzahl Reviews',
                            data: {js_chart_counts},
                            backgroundColor: 'rgba(37, 99, 235, 0.2)',
                            borderRadius: 4,
                            yAxisID: 'y1'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ min: 1, max: 5, position: 'left', grid: {{ color: '#e2e8f0' }} }},
                        y1: {{ position: 'right', grid: {{ drawOnChartArea: false }} }},
                        x: {{ grid: {{ display: false }} }}
                    }}
                }}
            }});
            
            function updateChartColors() {{
                const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
                const gridColor = isDark ? '#334155' : '#e2e8f0';
                chart.options.scales.y.grid.color = gridColor;
                chart.update();
            }}
            updateChartColors(); // Init

            function setFilter(app, btn) {{
                currentFilter = app;
                // Nur Filter-Buttons resetten (die ersten 3)
                const btns = document.querySelectorAll('.filter-btn');
                btns[0].classList.remove('active'); btns[1].classList.remove('active'); btns[2].classList.remove('active');
                btn.classList.add('active');
                filterData();
            }}

            function setSort(mode, btn) {{
                currentSort = mode;
                // Sortier Buttons resetten (ab Index 3)
                const btns = document.querySelectorAll('.filter-btn');
                btns[3].classList.remove('active'); btns[4].classList.remove('active'); btns[5].classList.remove('active');
                btn.classList.add('active');
                filterData();
            }}

            function copyText(text) {{ navigator.clipboard.writeText(text); }}

            function filterData() {{
                const q = document.getElementById('search').value.toLowerCase();
                const container = document.getElementById('list-container');
                container.innerHTML = '';

                let filtered = REVIEWS.filter(r => {{
                    return (currentFilter === 'all' || r.app === currentFilter) && (r.text + r.store).toLowerCase().includes(q);
                }});

                // Sortierung
                if (currentSort === 'newest') filtered.sort((a, b) => a.date < b.date ? 1 : -1);
                if (currentSort === 'best') filtered.sort((a, b) => b.rating - a.rating);
                if (currentSort === 'worst') filtered.sort((a, b) => a.rating - b.rating);

                if (filtered.length === 0) {{ container.innerHTML = '<div style="text-align:center;opacity:0.5">Keine Ergebnisse</div>'; return; }}

                filtered.slice(0, 50).forEach(r => {{
                    const icon = r.store === 'ios' ? '<i class="fab fa-apple icon-ios"></i>' : '<i class="fab fa-android icon-android"></i>';
                    const div = document.createElement('div');
                    div.className = 'review-card';
                    div.innerHTML = `
                        <div style="display:flex; justify-content:space-between; opacity:0.8; font-size:0.9rem;">
                            <span>${{icon}} <strong>${{r.app}}</strong> • ${{r.rating}}⭐ • ${{r.date}}</span>
                            <i class="fas fa-copy copy-btn" onclick="copyText('${{r.text.replace(/'/g, "\\'")}}')"></i>
                        </div>
                        <div style="margin-top:10px;">${{r.text}}</div>
                    `;
                    container.appendChild(div);
                }});
            }}
            
            // Init default sort btn
            document.querySelectorAll('.filter-btn')[3].classList.add('active');
            filterData();
        </script>
    </body>
    </html>
    """

    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f: f.write(html)
    print("✅ Pro-Dashboard v4.0 generiert.")

# ---------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    full, new = get_fresh_reviews()
    save_history({r['id']: r for r in full})
    run_analysis_and_generate_html(full, new)

    teams = os.getenv("TEAMS_WEBHOOK_URL")
    if teams and new:
        pos = sum(1 for r in new if r['rating']>=4)
        neg = sum(1 for r in new if r['rating']<=2)
        txt = f"🚀 **FEEDBACK UPDATE** ({len(new)})\n\n👍 {pos} | 🚨 {neg}\n[Dashboard](https://Hatozoro.github.io/feedback-agent/)"
        try: requests.post(teams, json={"text": txt})
        except: pass

    print("✅ Fertig.")