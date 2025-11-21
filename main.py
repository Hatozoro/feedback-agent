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
# IMPORTS
# ---------------------------------------------------------
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews as play_reviews

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config={"response_mime_type": "application/json"})
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    except: model = None; embedder = None
else: model = None; embedder = None

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
        except: pass
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
            if r.get('text') and r.get('rating') is not None:
                dated_reviews.append((review_date, float(r['rating'])))
        except: continue

    if not dated_reviews: return {'overall': 0.0, 'last_7d': 0.0, 'last_30d': 0.0}
    def get_avg(days):
        cutoff = today - timedelta(days=days)
        f = [r for d, r in dated_reviews if d >= cutoff]
        return round(sum(f)/len(f), 2) if f else 0.0
    return {'overall': round(sum(r for d, r in dated_reviews)/len(dated_reviews), 2), 'last_7d': get_avg(7), 'last_30d': get_avg(30)}

def prepare_chart_data(reviews, days=14):
    today = datetime.now().date()
    stats = {}
    for i in range(days):
        k = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        stats[k] = {'pos': 0, 'neg': 0, 'neu': 0}
    for r in reviews:
        d = r['date']
        rat = r.get('rating')
        if d in stats and rat is not None:
            if rat >= 4: stats[d]['pos'] += 1
            elif rat <= 2: stats[d]['neg'] += 1
            else: stats[d]['neu'] += 1
    labels = sorted(stats.keys())
    fmt_labels = [datetime.strptime(l, '%Y-%m-%d').strftime('%d.%m.') for l in labels]
    return {'labels': fmt_labels, 'pos': [stats[d]['pos'] for d in labels], 'neg': [stats[d]['neg'] for d in labels], 'neu': [stats[d]['neu'] for d in labels]}

def get_ai_buzzwords(reviews):
    if not model: return []
    text_sample = [r['text'] for r in reviews[:100] if len(r.get('text','')) > 10]
    prompt = f"""
    Analysiere die Reviews. Identifiziere die 10 häufigsten spezifischen Probleme.
    Output JSON Liste: [ {{"term": "Thema", "count": 12}}, ... ]
    Reviews: {json.dumps(text_sample, ensure_ascii=False)}
    """
    try:
        resp = model.generate_content(prompt)
        data = json.loads(resp.text.replace("```json", "").replace("```", "").strip())
        return [(i['term'], i['count']) for i in data if isinstance(i, dict)]
    except: return []

def is_genuine_positive(review):
    bad_words = ["absturz", "stürzt", "fehler", "schlecht", "katastrophe", "mies", "flackern", "unbrauchbar", "geht nicht"]
    if review.get('rating', 0) < 4: return True
    text = review.get('text', '').lower()
    if any(word in text for word in bad_words): return False
    return True

# ---------------------------------------------------------
# 3. SCRAPING
# ---------------------------------------------------------
def fetch_ios_reviews(app_name, app_id, country="de", count=20):
    print(f"   -> iOS (RSS): {app_name}...")
    try:
        url = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostrecent/json"
        data = requests.get(url, timeout=10).json()
        res = []
        for e in data.get('feed', {}).get('entry', [])[:count]:
            if 'content' not in e: continue
            res.append({
                "store": "ios", "app": app_name, "rating": int(e['im:rating']['label']),
                "text": e['content']['label'], "date": e.get('updated', {}).get('label', '')[:10],
                "id": generate_id({'app': app_name, 'store': 'ios', 'text': e['content']['label']})
            })
        return res
    except: return []

def fetch_android_reviews(app_name, app_id, country="de", count=20):
    print(f"   -> Android: {app_name}...")
    try:
        res = play_reviews(app_id, lang=country, country=country, sort=Sort.NEWEST, count=count)[0]
        out = []
        for r in res:
            d = r['at'].strftime('%Y-%m-%d')
            out.append({
                "store": "android", "app": app_name, "rating": r['score'], "text": r['content'], "date": d,
                "id": generate_id({'app': app_name, 'store': 'android', 'date': d, 'text': r['content']})
            })
        return out
    except: return []

def get_fresh_reviews(count=20):
    hist = load_history()
    new = []
    print(f"--- Scrape Start ---")
    for app in APP_CONFIG:
        for r in fetch_ios_reviews(app['name'], app['ios_id'], app['country'], count) + \
                 fetch_android_reviews(app['name'], app['android_id'], app['country'], count):
            if r['id'] not in hist:
                hist[r['id']] = r
                new.append(r)
    return sorted(hist.values(), key=lambda x: x['date'], reverse=True), new

# ---------------------------------------------------------
# 4. INTELLIGENZ
# ---------------------------------------------------------
def get_semantic_topics(reviews):
    if not embedder or not model: return ["KI nicht bereit"]
    txts = [r['text'] for r in reviews[:300] if len(r.get('text','')) > 15]
    if len(txts) < 5: return ["Zu wenige Daten"]

    vecs = embedder.encode(txts)
    kmeans = KMeans(n_clusters=min(5, len(txts)), n_init=10).fit(vecs)

    samples = []
    for i in range(kmeans.n_clusters):
        idx = np.where(kmeans.labels_ == i)[0]
        if idx.size == 0: continue
        center = kmeans.cluster_centers_[i]
        best = idx[np.argmax(cosine_similarity([center], vecs[idx]))]
        samples.append(txts[best])

    try:
        p = f'Erstelle Labels (1-2 Wörter) für diese Themen: {json.dumps(samples, ensure_ascii=False)}. Output JSON Liste.'
        return json.loads(model.generate_content(p).text.replace("```json","").replace("```","").strip())
    except: return ["Allgemein"]

# ---------------------------------------------------------
# 5. DASHBOARD GENERATOR
# ---------------------------------------------------------
def run_analysis_and_generate_html(full_history, new_only):
    trends = calculate_trends(full_history)
    chart = prepare_chart_data(full_history)
    topics = get_semantic_topics(full_history)
    buzzwords = get_ai_buzzwords(full_history)

    ki_data = {"summary": "Keine Analyse.", "topReviews": [], "bottomReviews": []}
    rich = [r for r in full_history if len(r.get('text', '')) > 40]
    if len(rich) < 10: rich = full_history

    if model and rich:
        print("--- KI Analyse ---")
        p = f"""
        Analysiere diese Reviews (max 50).
        1. Management Summary (Deutsch).
        2. 3 Top-Reviews (Positiv) und 3 Bottom-Reviews (Negativ).
        Output JSON: {{ "summary": "...", "topReviews": [{{...}}], "bottomReviews": [{{...}}] }}
        Data: {json.dumps([{'text': r['text'], 'rating': r['rating'], 'store': r['store'], 'app': r['app']} for r in rich[:50]], ensure_ascii=False)}
        """
        try:
            resp = model.generate_content(p)
            ki_data.update(json.loads(resp.text.replace("```json","").replace("```","").strip()))
        except: pass

    # --- ANTI-DOPPELGÄNGER LOGIK (STRIKT) ---
    seen_texts = set()
    top_list = []
    bot_list = []

    # 1. Python Filter
    candidates_pos = sorted([r for r in full_history if r['rating']>=4 and is_genuine_positive(r)], key=lambda x: len(x['text']), reverse=True)
    for r in candidates_pos:
        if len(top_list) >= 3: break
        if r['text'] not in seen_texts:
            top_list.append(r)
            seen_texts.add(r['text'])

    candidates_neg = sorted([r for r in full_history if r['rating']<=2], key=lambda x: len(x['text']), reverse=True)
    for r in candidates_neg:
        if len(bot_list) >= 3: break
        if r['text'] not in seen_texts:
            bot_list.append(r)
            seen_texts.add(r['text'])

    # 2. KI Fallback (Nur wenn Python nichts gefunden hat)
    if len(top_list) < 1:
        for r in ki_data.get('topReviews', []):
            if len(top_list) >= 3: break
            if r.get('text') not in seen_texts:
                top_list.append(r)
                seen_texts.add(r.get('text'))

    if len(bot_list) < 1:
        for r in ki_data.get('bottomReviews', []):
            if len(bot_list) >= 3: break
            if r.get('text') not in seen_texts:
                bot_list.append(r)
                seen_texts.add(r.get('text'))

    # Metadaten auffüllen
    for r in top_list + bot_list:
        if not r.get('app'):
            m = next((x for x in full_history if x['text'][:20] == r.get('text','').strip()[:20]), None)
            if m: r.update({'app': m['app'], 'store': m['store'], 'rating': m['rating']})

    summary = str(ki_data.get('summary', '')).strip().replace('{','').replace('}','').replace('"','')

    for r in full_history:
        if 'date' in r:
            try: r['fmt_date'] = datetime.strptime(r['date'], '%Y-%m-%d').strftime('%d.%m.%Y')
            except: r['fmt_date'] = r['date']

    max_c = buzzwords[0][1] if buzzwords else 1
    buzz_html = '<div class="buzz-container">'
    for w, c in buzzwords:
        intensity = min(1.0, max(0.1, c / max_c))
        buzz_html += f'<span class="buzz-tag" style="--intensity:{intensity};">{w} <span class="count">{c}</span></span>'
    buzz_html += '</div>'

    js_reviews = json.dumps(full_history, ensure_ascii=False)
    js_labels = json.dumps(chart['labels'])
    js_pos = json.dumps(chart['pos'])
    js_neg = json.dumps(chart['neg'])
    js_neu = json.dumps(chart['neu'])

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
                --bg: #f8fafc; --text: #1e293b; --card: #fff; --border: #e2e8f0; --primary: #2563eb; 
                --summary: #eff6ff; --ios: #000; --android: #3DDC84; --mark-bg: #fef08a; --mark-text: #854d0e;
                --buzz-base: 220, 38, 38; 
            }}
            [data-theme="dark"] {{ 
                --bg: #0f172a; --text: #f8fafc; --card: #1e293b; --border: #334155; --primary: #60a5fa; 
                --summary: #1e293b; --ios: #fff; --mark-bg: #854d0e; --mark-text: #fef08a;
                --buzz-base: 248, 113, 113;
            }}
            body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }}
            .theme-btn {{ background: none; border: 1px solid var(--border); color: var(--text); padding: 8px 12px; border-radius: 8px; cursor: pointer; }}
            
            .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .card {{ background: var(--card); padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid var(--border); }}
            .kpi-val {{ font-size: 2.5rem; font-weight: 700; color: var(--primary); margin-top: 10px; }}
            .kpi-label {{ font-size: 0.9rem; opacity: 0.8; text-transform: uppercase; font-weight: bold; }}
            
            .chart-container {{ background: var(--card); padding: 20px; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 30px; height: 350px; }}
            .summary-box {{ background: var(--summary); padding: 25px; border-radius: 12px; border-left: 5px solid var(--primary); margin-bottom: 30px; line-height: 1.6; border: 1px solid var(--border); }}
            
            .tag {{ display: inline-block; background: var(--card); border: 1px solid var(--border); padding: 6px 14px; border-radius: 20px; margin: 0 8px 8px 0; font-size: 0.9rem; color: var(--text); }}
            
            /* BUZZWORD DESIGN */
            .buzz-container {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-start; }}
            .buzz-tag {{ 
                display: inline-flex; align-items: center; 
                padding: 6px 12px; border-radius: 20px; 
                background-color: rgba(var(--buzz-base), calc(0.05 + var(--intensity) * 0.2));
                border: 1px solid rgba(var(--buzz-base), calc(0.2 + var(--intensity) * 0.5));
                color: var(--text); font-weight: 500;
            }}
            .buzz-tag .count {{ background: rgba(0,0,0,0.1); padding: 2px 6px; border-radius: 10px; font-size: 0.75em; margin-left: 8px; }}

            .review-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }}
            .review-card {{ background: var(--card); padding: 20px; border-radius: 8px; border: 1px solid var(--border); display: flex; flex-direction: column; gap: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
            .review-card.pos {{ border-top: 4px solid #22c55e; }}
            .review-card.neg {{ border-top: 4px solid #ef4444; }}
            
            .icon-android {{ color: var(--android); }} .icon-ios {{ color: var(--ios); }}
            .copy-btn {{ cursor: pointer; float: right; opacity: 0.5; }} .copy-btn:hover {{ opacity: 1; color: var(--primary); }}
            
            .search-input {{ flex: 1; padding: 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 1rem; background: var(--card); color: var(--text); }}
            
            /* NEW FILTER GROUP DESIGN */
            .filter-row {{ display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap; align-items: center; }}
            .filter-group {{ display: flex; gap: 5px; align-items: center; padding: 4px; background: var(--card); border: 1px solid var(--border); border-radius: 8px; }}
            .filter-label {{ font-size: 0.75rem; opacity: 0.7; text-transform: uppercase; font-weight: bold; margin: 0 8px; }}
            .filter-btn {{ padding: 8px 16px; border: none; background: transparent; color: var(--text); border-radius: 6px; cursor: pointer; font-size: 0.9rem; transition: background 0.2s; }}
            .filter-btn:hover {{ background: rgba(0,0,0,0.05); }}
            .filter-btn.active {{ background: var(--primary); color: white; }}
            
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
                    <div class="kpi-val">{trends['overall']} ⭐</div>
                </div>
                <div class="card">
                    <div class="kpi-label">Trend (7 Tage)</div>
                    <div class="kpi-val">{trends['last_7d']} ⭐</div>
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
                <p>{summary}</p>
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
            
            <div style="margin-bottom:20px;">
                <input type="text" class="search-input" id="search" placeholder="Suche nach Stichworten..." onkeyup="filterData()" style="width:100%; box-sizing:border-box;">
                
                <div class="filter-row">
                    <div class="filter-group">
                        <span class="filter-label">App</span>
                        <button class="filter-btn active" onclick="setFilter('app', 'all', this)">Alle</button>
                        <button class="filter-btn" onclick="setFilter('app', 'Nordkurier', this)">NK</button>
                        <button class="filter-btn" onclick="setFilter('app', 'Schwäbische', this)">SZ</button>
                    </div>
                    
                    <div class="filter-group">
                        <span class="filter-label">Store</span>
                        <button class="filter-btn active" onclick="setFilter('store', 'all', this)">Alle</button>
                        <button class="filter-btn" onclick="setFilter('store', 'ios', this)"><i class="fab fa-apple"></i></button>
                        <button class="filter-btn" onclick="setFilter('store', 'android', this)"><i class="fab fa-android"></i></button>
                    </div>

                    <div class="filter-group">
                        <span class="filter-label">Sort</span>
                        <button class="filter-btn active" onclick="setSort('newest', this)">Neu</button>
                        <button class="filter-btn" onclick="setSort('best', this)">Beste</button>
                        <button class="filter-btn" onclick="setSort('worst', this)">Schlechteste</button>
                    </div>
                </div>
            </div>

            <div id="list-container" style="display:grid; gap:15px;"></div>
        </div>

        <script>
            const REVIEWS = {js_reviews};
            let filterApp = 'all';
            let filterStore = 'all';
            let currentSort = 'newest';

            const theme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', theme);
            
            function toggleTheme() {{
                const newTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                updateChartColors();
            }}

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

            function setFilter(type, value, btn) {{
                if (type === 'app') filterApp = value;
                if (type === 'store') filterStore = value;
                
                // Reset active class in group
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
                    const appMatch = (filterApp === 'all' || r.app === filterApp);
                    const storeMatch = (filterStore === 'all' || r.store === filterStore);
                    const searchMatch = (r.text + r.store).toLowerCase().includes(q);
                    return appMatch && storeMatch && searchMatch;
                }});

                if (currentSort === 'newest') filtered.sort((a, b) => a.date < b.date ? 1 : -1);
                if (currentSort === 'best') filtered.sort((a, b) => b.rating - a.rating);
                if (currentSort === 'worst') filtered.sort((a, b) => a.rating - b.rating);

                if (filtered.length === 0) {{ container.innerHTML = '<div style="text-align:center;opacity:0.5;padding:20px;">Keine Ergebnisse</div>'; return; }}

                filtered.slice(0, 50).forEach(r => {{
                    const icon = r.store === 'ios' ? '<i class="fab fa-apple icon-ios"></i>' : '<i class="fab fa-android icon-android"></i>';
                    
                    // HIGHLIGHTING LOGIC
                    let displayText = r.text;
                    if (q.length >= 2) {{
                        const terms = q.split(' ').filter(t => t.length > 1);
                        const allFound = terms.every(term => r.text.toLowerCase().includes(term));
                        
                        if (allFound) {{
                            terms.forEach(term => {{
                                const regex = new RegExp('(' + term + ')', 'gi');
                                displayText = displayText.replace(regex, '<mark>$1</mark>');
                            }});
                        }} else {{
                            return;
                        }}
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