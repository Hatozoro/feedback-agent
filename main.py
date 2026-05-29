import os
import json
import time
import hashlib
import requests
import re
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. IMPORTS & SETUP
# ---------------------------------------------------------
from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews as play_reviews

DATA_FILE = "data/reviews_history.json"

APP_CONFIG = [
    {"name": "Nordkurier", "ios_id": "1250964862", "android_id": "de.nordkurier.live", "country": "de"},
    {"name": "Schwäbische", "ios_id": "432491155", "android_id": "de.schwaebische.epaper", "country": "de"}
]

STOP_WORDS = {
    "die", "der", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", "einem", "einen",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "mir", "meine", "meiner", "mein",
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

POS_WORDS = {"gut", "super", "klasse", "toll", "perfekt", "danke", "schnell", "übersichtlich", "lob", "zufrieden", "schön", "top", "besser"}
NEG_WORDS = {"schlecht", "fehler", "absturz", "stürzt", "nervt", "werbung", "katastrophe", "langsam", "unbrauchbar", "müll", "schlimm", "teuer", "abo", "friert", "geht nicht"}

# ---------------------------------------------------------
# 2. DATA MANAGEMENT
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

# ---------------------------------------------------------
# 3. ANALYTICS & NLP (REGELBASIERT)
# ---------------------------------------------------------
def calculate_trends(reviews):
    today = datetime.now().date()
    dated_reviews = []
    breakdown = {'Nordkurier': {'ios': [], 'android': []}, 'Schwäbische': {'ios': [], 'android': []}}
    ios_total = 0
    android_total = 0

    for r in reviews:
        if r.get('store') == 'ios': ios_total += 1
        elif r.get('store') == 'android': android_total += 1

        try:
            r_date = datetime.strptime(r['date'], '%Y-%m-%d').date()
            rating = float(r['rating'])
            if r.get('text'):
                dated_reviews.append((r_date, rating))
            if r['app'] in breakdown and r['store'] in breakdown[r['app']]:
                breakdown[r['app']][r['store']].append(rating)
        except: continue

    if not dated_reviews:
        return {'overall': 0.0, 'breakdown': {}, 'ios_total': 0, 'android_total': 0}

    final_breakdown = {}
    for app, stores in breakdown.items():
        final_breakdown[app] = {}
        for store, ratings in stores.items():
            final_breakdown[app][store] = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

    overall_avg = round(sum(r['rating'] for r in reviews if r.get('rating') is not None) / (len(reviews) or 1), 2)

    return {
        'overall': overall_avg,
        'breakdown': final_breakdown,
        'ios_total': ios_total,
        'android_total': android_total
    }

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

def analyze_review_quality(text, rating):
    """Berechnet einen Quality/Sentiment Score für das Ranking."""
    text_lower = text.lower()
    words = text_lower.split()
    
    pos_hits = sum(1 for w in words if any(p in w for p in POS_WORDS))
    neg_hits = sum(1 for w in words if any(n in w for n in NEG_WORDS))
    
    # Basis-Sentiment aus Rating (-2 bis +2)
    base_score = (rating - 3) * 2
    sentiment_score = base_score + pos_hits - neg_hits
    
    # Textqualität: Längere Texte (bis ca. 40 Wörter) sind besser, danach Cap
    quality_bonus = min(len(words) / 20, 2.0)
    
    # Kombinierter Score (Positive profitieren von Länge, Negative werden "negativer" durch Länge)
    total_score = sentiment_score + (quality_bonus if sentiment_score >= 0 else -quality_bonus)
    
    return sentiment_score, total_score

def get_tfidf_keywords(reviews, top_n=15):
    """TF-IDF ähnliche Logik für deutlich bessere Themen-Cluster."""
    doc_freqs = defaultdict(int)
    term_freqs = defaultdict(int)
    total_docs = len(reviews)
    
    if total_docs == 0: return []

    for r in reviews:
        text = re.sub(r'[^\w\säöüß]', ' ', r.get('text', '').lower())
        words = [w for w in text.split() if len(w) > 3 and w not in STOP_WORDS]
        unique_words = set(words)
        
        for w in unique_words: doc_freqs[w] += 1
        for w in words: term_freqs[w] += 1

    tfidf_scores = {}
    for w, tf in term_freqs.items():
        if doc_freqs[w] > 1:  # Muss in mind. 2 Reviews vorkommen
            idf = math.log(total_docs / doc_freqs[w])
            tfidf_scores[w] = tf * idf

    sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]

def generate_auto_summary(trends, buzzwords, total_count):
    """Generiert eine dynamische Zusammenfassung basierend auf den Daten."""
    if total_count == 0: return "Aktuell sind keine Nutzerbewertungen zur Analyse vorhanden."
    
    avg = trends['overall']
    trend_desc = "sehr positiv" if avg >= 4.0 else ("überwiegend positiv" if avg >= 3.5 else ("durchwachsen" if avg >= 2.5 else "kritisch"))
    
    top_issues = ", ".join([w[0].capitalize() for w in buzzwords[:4]]) if buzzwords else "allgemeinen Themen"
    
    return f"Die aktuelle Analyse umfasst {total_count} Reviews. Die durchschnittliche Bewertung liegt bei {avg} Sternen, was auf ein {trend_desc}es Stimmungsbild hindeutet. Besondere Aufmerksamkeit in den Feedbacks erhalten aktuell folgende Themen: {top_issues}."

# ---------------------------------------------------------
# 4. SCRAPING
# ---------------------------------------------------------
def fetch_ios_reviews(app_name, app_id, country="de", count=50):
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

def fetch_android_reviews(app_name, app_id, country="de", count=50):
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

def get_fresh_reviews(count=50):
    hist = load_history()
    new = []
    print(f"--- Scrape Start ---")
    for app in APP_CONFIG:
        fetched = fetch_ios_reviews(app['name'], app['ios_id'], app['country'], count) + \
                  fetch_android_reviews(app['name'], app['android_id'], app['country'], count)
        for r in fetched:
            if r['id'] not in hist:
                hist[r['id']] = r
                new.append(r)
    return sorted(hist.values(), key=lambda x: x['date'], reverse=True), new

# ---------------------------------------------------------
# 5. DASHBOARD GENERATOR
# ---------------------------------------------------------
def run_analysis_and_generate_html(full_history):
    trends = calculate_trends(full_history)
    chart = prepare_chart_data(full_history)

    # NLP Features berechnen
    buzzwords = get_tfidf_keywords(full_history, top_n=15)
    topics = [w[0].capitalize() for w in buzzwords[:5]]
    summary = generate_auto_summary(trends, buzzwords, len(full_history))

    # Scoring & Ranking
    for r in full_history:
        r['sentiment'], r['score'] = analyze_review_quality(r['text'], r['rating'])
        if 'date' in r:
            try: r['fmt_date'] = datetime.strptime(r['date'], '%Y-%m-%d').strftime('%d.%m.%Y')
            except: r['fmt_date'] = r['date']

    # Top/Bottom Listen filtern (ohne Duplikate durch identische Texte)
    seen_texts = set()
    top_list = []
    bot_list = []
    
    sorted_reviews = sorted(full_history, key=lambda x: x['score'], reverse=True)
    
    for r in sorted_reviews:
        if r['score'] > 2 and len(top_list) < 3 and r['text'] not in seen_texts:
            top_list.append(r)
            seen_texts.add(r['text'])
            
    for r in reversed(sorted_reviews):
        if r['score'] < -2 and len(bot_list) < 3 and r['text'] not in seen_texts:
            bot_list.append(r)
            seen_texts.add(r['text'])

    # Fallback falls Filter zu streng
    if not top_list: top_list = sorted([r for r in full_history if r['rating'] >= 4], key=lambda x: x['score'], reverse=True)[:3]
    if not bot_list: bot_list = sorted([r for r in full_history if r['rating'] <= 2], key=lambda x: x['score'])[:3]

    max_c = buzzwords[0][1] if buzzwords else 1
    buzz_html = '<div class="buzz-container">'
    for w, c in buzzwords:
        intensity = min(1.0, max(0.1, c / max_c))
        buzz_html += f'<span class="buzz-tag" style="--intensity:{intensity};" onclick="setSearch(\'{w}\')">{w} <span class="count">{int(c)}</span></span>'
    buzz_html += '</div>'

    breakdown_html = ""
    for app, stores in trends.get('breakdown', {}).items():
        ios_score = stores.get('ios', 0)
        android_score = stores.get('android', 0)
        breakdown_html += f'<div style="margin-bottom:8px; padding-bottom:8px; border-bottom:1px solid var(--border);"><strong>{app}</strong><br>'
        breakdown_html += f'<span style="font-size:0.9rem"><i class="fab fa-apple"></i> {ios_score}⭐ &nbsp;|&nbsp; <i class="fab fa-android"></i> {android_score}⭐</span></div>'

    # Unnötige Daten für Frontend entfernen
    clean_history = [{k: v for k, v in r.items() if k not in ['sentiment', 'score']} for r in full_history]
    js_reviews = json.dumps(clean_history, ensure_ascii=False)

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
            
            .buzz-container {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-start; }}
            .buzz-tag {{ 
                display: inline-flex; align-items: center; cursor: pointer;
                padding: 6px 12px; border-radius: 20px; 
                background-color: rgba(var(--buzz-base), calc(0.05 + var(--intensity) * 0.2));
                border: 1px solid rgba(var(--buzz-base), calc(0.2 + var(--intensity) * 0.5));
                color: var(--text); font-weight: 500; transition: transform 0.2s;
            }}
            .buzz-tag:hover {{ transform: scale(1.05); border-color: var(--primary); }}
            .buzz-tag .count {{ background: rgba(0,0,0,0.1); padding: 2px 6px; border-radius: 10px; font-size: 0.75em; margin-left: 8px; display:none;}} /* Ausgeblendet da Float TF-IDF Score, kann wieder rein wenn gewünscht */

            .review-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }}
            .review-card {{ background: var(--card); padding: 20px; border-radius: 8px; border: 1px solid var(--border); display: flex; flex-direction: column; gap: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
            .review-card.pos {{ border-top: 4px solid #22c55e; }}
            .review-card.neg {{ border-top: 4px solid #ef4444; }}
            
            .icon-android {{ color: var(--android); }} .icon-ios {{ color: var(--ios); }}
            .copy-btn {{ cursor: pointer; float: right; opacity: 0.5; }} .copy-btn:hover {{ opacity: 1; color: var(--primary); }}
            
            .search-input {{ flex: 1; padding: 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 1rem; background: var(--card); color: var(--text); }}
            
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
                    <h1 style="margin:0;">📊 App Feedback Pro (Regelbasiert)</h1>
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
                    <div class="kpi-label">DETAILS PRO APP</div>
                    <div style="margin-top:15px;">{breakdown_html}</div>
                </div>
                <div class="card">
                    <div class="kpi-label">Reviews pro Store</div>
                    <div style="margin-top:10px; font-size:1.1rem;">
                        <i class="fab fa-apple"></i> {trends['ios_total']} &nbsp;|&nbsp; 
                        <i class="fab fa-android"></i> {trends['android_total']}
                    </div>
                </div>
            </div>

            <div class="chart-container">
                <h4 style="margin:0 0 15px 0; opacity:0.7;">Bewertungsverlauf (14 Tage)</h4>
                <canvas id="trendChart"></canvas>
            </div>
            
            <div class="summary-box">
                <h3 style="margin-top:0;">🤖 Daten-Zusammenfassung</h3>
                <p>{summary}</p>
            </div>

            <div class="row" style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:40px;">
                <div class="col" style="flex:1;">
                    <h3 style="margin-bottom: 15px;">🔥 Themen-Cluster (TF-IDF)</h3>
                    <div class="card" style="min-height:100px;">
                        {''.join([f'<span class="tag"># {t}</span> ' for t in topics])}
                    </div>
                </div>
                <div class="col" style="flex:1;">
                    <h3 style="margin-bottom: 15px;">🚨 Relevante Keywords</h3>
                    <div class="card buzz-container">
                        {buzz_html}
                    </div>
                </div>
            </div>

            <div class="review-grid">
                <div>
                    <h3>👍 Top Stimmen (Sentiment-Score)</h3>
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
                    </div>''' for r in top_list])}
                </div>
                <div>
                    <h3>⚠️ Kritische Stimmen (Sentiment-Score)</h3>
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
                    </div>''' for r in bot_list])}
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
                    labels: {json.dumps(chart['labels'])},
                    datasets: [
                        {{ label: 'Positiv (4-5★)', data: {json.dumps(chart['pos'])}, backgroundColor: '#22c55e' }},
                        {{ label: 'Neutral (3★)', data: {json.dumps(chart['neu'])}, backgroundColor: '#94a3b8' }},
                        {{ label: 'Negativ (1-2★)', data: {json.dumps(chart['neg'])}, backgroundColor: '#ef4444' }}
                    ]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{ 
                        x: {{ stacked: true, grid: {{ display: false }}, ticks: {{ color: '#64748b' }} }}, 
                        y: {{ stacked: true, grid: {{ color: '#e2e8f0' }}, ticks: {{ color: '#64748b', padding: 10 }} }} 
                    }},
                    plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#64748b' }} }} }}
                }}
            }});
            
            function updateChartColors() {{
                const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
                const textColor = isDark ? '#cbd5e1' : '#64748b';
                const gridColor = isDark ? '#334155' : '#e2e8f0';
                
                chart.options.scales.x.ticks.color = textColor;
                chart.options.scales.y.ticks.color = textColor;
                chart.options.scales.y.grid.color = gridColor;
                chart.options.plugins.legend.labels.color = textColor;
                chart.update();
            }}
            updateChartColors();

            document.querySelectorAll('.filter-group:last-child .filter-btn')[0].classList.add('active');

            function initReadMore() {{
                document.querySelectorAll('.review-content').forEach(div => {{
                    const text = div.querySelector('.review-text');
                    const btn = div.querySelector('.read-more');
                    if (text.scrollHeight > text.clientHeight) {{
                        btn.style.display = 'inline-block';
                    }}
                }});
            }}

            // NEUE ACCORDION-LOGIK HIER: Schließt alle anderen bevor der neue geöffnet wird
            function toggleText(btn) {{
                const text = btn.previousElementSibling;
                const isClamped = text.classList.contains('clamped');
                
                // Alle schliessen
                document.querySelectorAll('.review-text').forEach(t => t.classList.add('clamped'));
                document.querySelectorAll('.read-more').forEach(b => b.innerText = 'Mehr anzeigen');
                
                // Gewählten öffnen, wenn er vorher zu war
                if (isClamped) {{
                    text.classList.remove('clamped');
                    btn.innerText = 'Weniger anzeigen';
                }}
            }}

            function setFilter(type, value, btn) {{
                if (type === 'app') filterApp = value;
                if (type === 'store') filterStore = value;
                
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

            function setSearch(term) {{
                const input = document.getElementById('search');
                input.value = term;
                filterData();
                input.scrollIntoView({{behavior: 'smooth'}});
            }}

            function copyText(text) {{ navigator.clipboard.writeText(text); alert('Text kopiert!'); }}

            function filterData() {{
                const q = document.getElementById('search').value.toLowerCase().trim();
                const container = document.getElementById('list-container');
                container.innerHTML = '';

                let filtered = REVIEWS.filter(r => {{
                    const appMatch = (filterApp === 'all' || r.app === filterApp);
                    const storeMatch = (filterStore === 'all' || r.store === filterStore);
                    const searchMatch = (r.text + r.store + r.app).toLowerCase().includes(q);
                    return appMatch && storeMatch && searchMatch;
                }});

                if (currentSort === 'newest') filtered.sort((a, b) => a.date < b.date ? 1 : -1);
                if (currentSort === 'best') filtered.sort((a, b) => b.rating - a.rating);
                if (currentSort === 'worst') filtered.sort((a, b) => a.rating - b.rating);

                if (filtered.length === 0) {{ container.innerHTML = '<div style="text-align:center;opacity:0.5;padding:20px;">Keine Ergebnisse</div>'; return; }}

                filtered.slice(0, 50).forEach(r => {{
                    const icon = r.store === 'ios' ? '<i class="fab fa-apple icon-ios"></i>' : '<i class="fab fa-android icon-android"></i>';
                    
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
                                ${{icon}} <strong>${{r.app}}</strong> (${{r.store.toUpperCase()}}) • ${{r.rating}}⭐
                            </span>
                            <span>${{r.fmt_date || r.date}}</span>
                        </div>
                        <div class="review-content" style="line-height:1.5;">
                            <div class="review-text clamped">${{displayText}}</div>
                            <span class="read-more" onclick="toggleText(this)">Mehr anzeigen</span>
                            <i class="fas fa-copy copy-btn" onclick="copyText('${{r.text.replace(/'/g, "\\'")}}')"></i>
                        </div>
                    `;
                    container.appendChild(div);
                }});
                
                // Neu initialisieren für die gefilterten Items
                initReadMore();
            }}
            
            filterData();
            // Löst Init Read More nach initialem Aufbau einmalig aus
            setTimeout(initReadMore, 100);
        </script>
    </body>
    </html>
    """

    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ Dashboard (ohne KI) generiert.")

# ---------------------------------------------------------
# 6. TEAMS MESSAGECARD
# ---------------------------------------------------------
def send_teams_notification(new_reviews, webhook_url):
    if not new_reviews: return

    top_reviews = new_reviews[:10]
    pos = sum(1 for r in new_reviews if r['rating']>=4)
    neu = sum(1 for r in new_reviews if r['rating']==3)
    neg = sum(1 for r in new_reviews if r['rating']<=2)

    card = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "0076D7",
        "summary": f"Neues App Feedback ({len(new_reviews)})",
        "sections": [
            {
                "activityTitle": f"🚀 **Neues Feedback ({len(new_reviews)})**",
                "facts": [
                    {"name": "Positiv:", "value": str(pos)},
                    {"name": "Neutral:", "value": str(neu)},
                    {"name": "Negativ:", "value": str(neg)}
                ],
                "markdown": True
            }
        ],
        "potentialAction": [
            {
                "@type": "OpenUri",
                "name": "Zum Dashboard",
                "targets": [{"os": "default", "uri": "https://Hatozoro.github.io/feedback-agent/"}]
            }
        ]
    }

    for r in top_reviews:
        icon = "🍏" if r['store'] == 'ios' else "🤖"
        star = "⭐" * r['rating']
        section = {
            "title": f"{icon} {r['app']} ({star})",
            "text": r['text'],
            "markdown": True
        }
        card['sections'].append(section)

    if len(new_reviews) > 10:
        card['sections'].append({
            "text": f"... und {len(new_reviews) - 10} weitere auf dem Dashboard.",
            "markdown": True
        })

    try:
        requests.post(webhook_url, json=card)
        print("✅ Teams Notification gesendet.")
    except Exception as e:
        print(f"❌ Teams Fehler: {e}")

if __name__ == "__main__":
    full, new = get_fresh_reviews(count=50)
    save_history({r['id']: r for r in full})
    run_analysis_and_generate_html(full)

    teams = os.getenv("TEAMS_WEBHOOK_URL")
    if teams: send_teams_notification(new, teams)

    print("✅ Durchlauf beendet.")