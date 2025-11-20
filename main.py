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
# 1. SETUP
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
    except: model = None; embedder = None
else: model = None; embedder = None

DATA_FILE = "data/reviews_history.json"
APP_CONFIG = [
    {"name": "Nordkurier", "ios_id": "1250964862", "android_id": "de.nordkurier.live", "country": "de"},
    {"name": "Schwäbische", "ios_id": "432491155", "android_id": "de.schwaebische.epaper", "country": "de"}
]

# ---------------------------------------------------------
# 2. HELPER
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

    def get_avg_for_days(days):
        cutoff = today - timedelta(days=days)
        filtered = [r for d, r in dated_reviews if d >= cutoff]
        return round(sum(filtered) / len(filtered), 2) if filtered else 0.0

    overall_avg = round(sum(r['rating'] for r in reviews if r.get('rating')) / (len(reviews) or 1), 2)
    return {'overall': overall_avg, 'last_7d': get_avg_for_days(7), 'last_30d': get_avg_for_days(30)}

# NEU: Stacked Bar Chart Data (Positiv/Neutral/Negativ pro Tag)
def prepare_chart_data(reviews, days=14):
    today = datetime.now().date()
    stats = {}
    # Init letzte X Tage
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
    # Formatiere Label zu DD.MM
    fmt_labels = [datetime.strptime(l, '%Y-%m-%d').strftime('%d.%m') for l in labels]

    return {
        'labels': fmt_labels,
        'pos': [stats[d]['pos'] for d in labels],
        'neg': [stats[d]['neg'] for d in labels],
        'neu': [stats[d]['neu'] for d in labels]
    }

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
    except Exception as e:
        print(f"iOS Error: {e}")
        return []

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
    except Exception as e:
        print(f"Android Error: {e}")
        return []

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

    full = sorted(hist.values(), key=lambda x: x['date'], reverse=True)
    print(f"--- Status: {len(full)} Total, {len(new)} Neu ---")
    return full, new

# ---------------------------------------------------------
# 4. KI & CLUSTERING
# ---------------------------------------------------------
def get_semantic_topics(reviews):
    if not embedder or not model: return ["KI nicht bereit"]

    # Filter: Nur lange Reviews für Cluster
    txts = [r['text'] for r in reviews[:300] if len(r.get('text','')) > 15]
    if len(txts) < 5: return ["Zu wenige Daten"]

    embeddings = embedder.encode(txts)
    kmeans = KMeans(n_clusters=min(5, len(txts)), n_init=10).fit(embeddings)

    samples = []
    for i in range(kmeans.n_clusters):
        idx = np.where(kmeans.labels_ == i)[0]
        if idx.size == 0: continue
        center = kmeans.cluster_centers_[i]
        best = idx[np.argmax(cosine_similarity([center], embeddings[idx]))]
        samples.append(txts[best])

    try:
        p = f'Erstelle prägnante Labels (1-2 Wörter) für diese Themen: {json.dumps(samples, ensure_ascii=False)}. Output nur JSON Liste.'
        resp = model.generate_content(p)
        return json.loads(resp.text.replace("```json","").replace("```","").strip())
    except: return ["Themen Analyse Fehler"]

# ---------------------------------------------------------
# 5. HTML GENERATOR (MIT SICHERHEITSNETZ)
# ---------------------------------------------------------
def run_analysis_and_generate_html(full_history, new_only):
    trends = calculate_trends(full_history)
    chart = prepare_chart_data(full_history)
    topics = get_semantic_topics(full_history)

    # KI Analyse
    ki_data = {"summary": "Keine Analyse verfügbar.", "topReviews": [], "bottomReviews": []}

    # Filter: Bevorzuge aussagekräftige Reviews für die KI
    rich_reviews = [r for r in full_history if len(r.get('text', '')) > 40]
    if len(rich_reviews) < 10: rich_reviews = full_history

    if model and rich_reviews:
        print("--- KI Analyse läuft ---")
        p = f"""
        Analysiere diese App-Reviews.
        1. Schreibe ein Management Summary (Deutsch).
        2. Wähle 3 Top-Reviews (Positiv, 4-5 Sterne) und 3 Bottom-Reviews (Negativ, 1-2 Sterne).
        WICHTIG: Wähle Reviews mit AUSSAGEKRAFT (Textlänge > 10 Wörter), vermeide "Gut", "Toll" etc.
        
        Output JSON: {{ "summary": "...", "topReviews": [{{...}}], "bottomReviews": [{{...}}] }}
        Data: {json.dumps([{'text': r['text'], 'rating': r['rating'], 'store': r['store'], 'app': r['app']} for r in rich_reviews[:50]], ensure_ascii=False)}
        """
        try:
            resp = model.generate_content(p)
            text = resp.text.replace("```json","").replace("```","").strip()
            ki_data.update(json.loads(text))
        except Exception as e:
            print(f"KI Fehler: {e}")

    # --- DAS SICHERHEITSNETZ (FALLBACK) ---
    # Falls die KI keine Reviews geliefert hat (oder zu wenige), füllen wir sie manuell auf.
    # Das habe ich im vorherigen Code wegoptimiert, jetzt ist es wieder da.

    top_list = ki_data.get('topReviews', [])
    bot_list = ki_data.get('bottomReviews', [])

    if len(top_list) < 3:
        # Fülle mit besten echten Reviews auf
        sorted_best = sorted([r for r in full_history if r['rating'] >= 4], key=lambda x: len(x['text']), reverse=True)
        for r in sorted_best:
            if len(top_list) >= 3: break
            if r['text'] not in [t['text'] for t in top_list]:
                top_list.append({'text': r['text'], 'rating': r['rating'], 'store': r['store'], 'app': r['app']})

    if len(bot_list) < 3:
        # Fülle mit schlechtesten echten Reviews auf
        sorted_worst = sorted([r for r in full_history if r['rating'] <= 2], key=lambda x: len(x['text']), reverse=True)
        for r in sorted_worst:
            if len(bot_list) >= 3: break
            if r['text'] not in [t['text'] for t in bot_list]:
                bot_list.append({'text': r['text'], 'rating': r['rating'], 'store': r['store'], 'app': r['app']})

    # Update KI Data mit Fallback Werten
    ki_data['topReviews'] = top_list
    ki_data['bottomReviews'] = bot_list

    # Metadaten-Korrektur (falls KI App-Namen vergessen hat)
    for l in [ki_data['topReviews'], ki_data['bottomReviews']]:
        for r in l:
            if not r.get('app'):
                m = next((x for x in full_history if x['text'][:15] == r.get('text','').strip()[:15]), None)
                if m: r.update({'app': m['app'], 'store': m['store'], 'rating': m['rating']})

    summary = str(ki_data.get('summary', '')).strip().replace('{','').replace('}','')

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
            :root {{ --bg: #f8fafc; --text: #0f172a; --card: #fff; --border: #e2e8f0; --primary: #3b82f6; }}
            [data-theme="dark"] {{ --bg: #0f172a; --text: #f8fafc; --card: #1e293b; --border: #334155; --primary: #60a5fa; }}
            body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .card {{ background: var(--card); padding: 20px; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 20px; }}
            .row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
            .col {{ flex: 1; min-width: 300px; }}
            
            .kpi {{ text-align: center; padding: 15px; }}
            .kpi-val {{ font-size: 2rem; font-weight: 800; color: var(--primary); }}
            
            .tag {{ background: var(--bg); padding: 5px 12px; border-radius: 15px; border: 1px solid var(--border); display: inline-block; margin: 0 5px 5px 0; font-size: 0.9rem; }}
            
            /* Review Styling & Read More */
            .review-box {{ background: var(--card); border: 1px solid var(--border); padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
            .review-box.pos {{ border-left: 4px solid #22c55e; }}
            .review-box.neg {{ border-left: 4px solid #ef4444; }}
            
            .review-text {{ margin-top: 8px; line-height: 1.5; position: relative; }}
            .review-text.clamped {{ display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }}
            .read-more {{ color: var(--primary); cursor: pointer; font-size: 0.9rem; display: none; margin-top: 5px; }}
            
            .meta {{ font-size: 0.85rem; opacity: 0.7; display: flex; justify-content: space-between; }}
            
            /* Inputs */
            input, button {{ padding: 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--card); color: var(--text); }}
            button.active {{ background: var(--primary); color: white; border-color: var(--primary); }}
        </style>
    </head>
    <body>
        <div class="container">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <h1 style="margin:0">📊 Feedback Cockpit</h1>
                <button onclick="toggleTheme()">🌗</button>
            </div>

            <div class="row">
                <div class="card col kpi"><div>Gesamt Ø</div><div class="kpi-val">{trends['overall']} ⭐</div></div>
                <div class="card col kpi"><div>Trend (7 Tage)</div><div class="kpi-val">{trends['last_7d']} ⭐</div></div>
                <div class="card col kpi"><div>Reviews (Total)</div><div class="kpi-val">{len(full_history)}</div></div>
            </div>

            <div class="card" style="height: 300px;">
                <canvas id="chart"></canvas>
            </div>

            <div class="card">
                <h3>🤖 KI-Analyse</h3>
                <p>{summary}</p>
                <div style="margin-top:15px;">
                    <strong>Themen:</strong><br>
                    {''.join([f'<span class="tag"># {t}</span>' for t in topics])}
                </div>
            </div>

            <div class="row">
                <div class="col">
                    <h3>👍 Top Insights (4-5★)</h3>
                    {''.join([f'''
                    <div class="review-box pos">
                        <div class="meta"><span>{r.get('rating')}★ {r.get('app')}</span> <span>{r.get('store')}</span></div>
                        <div class="review-content">
                            <div class="review-text clamped">{r.get('text')}</div>
                            <span class="read-more" onclick="toggleText(this)">Mehr anzeigen</span>
                        </div>
                    </div>''' for r in ki_data.get('topReviews', [])])}
                </div>
                <div class="col">
                    <h3>⚠️ Kritische Stimmen (1-2★)</h3>
                    {''.join([f'''
                    <div class="review-box neg">
                        <div class="meta"><span>{r.get('rating')}★ {r.get('app')}</span> <span>{r.get('store')}</span></div>
                        <div class="review-content">
                            <div class="review-text clamped">{r.get('text')}</div>
                            <span class="read-more" onclick="toggleText(this)">Mehr anzeigen</span>
                        </div>
                    </div>''' for r in ki_data.get('bottomReviews', [])])}
                </div>
            </div>

            <h2 style="margin-top:40px">🔎 Explorer</h2>
            <div style="margin-bottom:20px; display:flex; gap:10px; flex-wrap:wrap;">
                <input type="text" id="search" placeholder="Suchen..." style="flex:1;" onkeyup="filter()">
                <button class="filter-btn active" onclick="setFilter('all', this)">Alle</button>
                <button class="filter-btn" onclick="setFilter('Nordkurier', this)">NK</button>
                <button class="filter-btn" onclick="setFilter('Schwäbische', this)">SZ</button>
            </div>
            <div id="list"></div>
        </div>

        <script>
            const DATA = {json.dumps(full_history, ensure_ascii=False)};
            let filterApp = 'all';

            // Theme
            const theme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', theme);
            function toggleTheme() {{
                const newTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
            }}

            // Chart
            new Chart(document.getElementById('chart'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(chart['labels'])},
                    datasets: [
                        {{ label: 'Positiv (4-5★)', data: {json.dumps(chart['pos'])}, backgroundColor: '#22c55e' }},
                        {{ label: 'Neutral (3★)', data: {json.dumps(chart['neu'])}, backgroundColor: '#94a3b8' }},
                        {{ label: 'Negativ (1-2★)', data: {json.dumps(chart['neg'])}, backgroundColor: '#ef4444' }}
                    ]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{ x: {{ stacked: true, grid: {{ display: false }} }}, y: {{ stacked: true }} }}
                }}
            }});

            // Read More Logic
            document.querySelectorAll('.review-content').forEach(div => {{
                const text = div.querySelector('.review-text');
                const btn = div.querySelector('.read-more');
                if (text.scrollHeight > text.clientHeight) btn.style.display = 'inline-block';
            }});
            function toggleText(btn) {{
                const text = btn.previousElementSibling;
                text.classList.toggle('clamped');
                btn.innerText = text.classList.contains('clamped') ? 'Mehr anzeigen' : 'Weniger anzeigen';
            }}

            // Filter Logic
            function setFilter(app, btn) {{
                filterApp = app;
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                filter();
            }}
            function filter() {{
                const q = document.getElementById('search').value.toLowerCase();
                const list = document.getElementById('list');
                list.innerHTML = '';
                
                const res = DATA.filter(r => (filterApp === 'all' || r.app === filterApp) && (r.text + r.store).toLowerCase().includes(q));
                
                if(res.length === 0) list.innerHTML = '<div style="text-align:center; opacity:0.5">Keine Ergebnisse</div>';
                
                res.slice(0, 50).forEach(r => {{
                    const div = document.createElement('div');
                    div.className = 'review-box';
                    div.innerHTML = `
                        <div class="meta"><span>${{r.rating}}★ ${{r.app}} (${{r.store}})</span> <span>${{r.date}}</span></div>
                        <div style="margin-top:5px;">${{r.text}}</div>
                    `;
                    list.appendChild(div);
                }});
            }}
            filter();
        </script>
    </body>
    </html>
    """
    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f: f.write(html)
    print("✅ Dashboard v5.1 generiert.")

# ---------------------------------------------------------
# 7. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    full, new = get_fresh_reviews()
    save_history({r['id']: r for r in full})
    run_analysis_and_generate_html(full, new)

    teams = os.getenv("TEAMS_WEBHOOK_URL")
    if teams and new:
        try:
            pos = sum(1 for r in new if r['rating']>=4)
            neg = sum(1 for r in new if r['rating']<=2)
            txt = f"🚀 **UPDATE** ({len(new)})\n\n👍 {pos} | 🚨 {neg}\n[Dashboard](https://Hatozoro.github.io/feedback-agent/)"
            requests.post(teams, json={"text": txt})
        except: pass

    print("✅ Fertig.")