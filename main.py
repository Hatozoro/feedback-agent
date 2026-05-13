"""
App Feedback Pro – Dashboard Generator
Scrapes iOS & Android reviews, runs hybrid AI analysis, generates a HTML dashboard.
"""

# ---------------------------------------------------------
# 1. IMPORTS & SETUP
# ---------------------------------------------------------
import os
import json
import logging
import hashlib
import re
import requests
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from google import genai
from google.genai import types
from dotenv import load_dotenv
from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews as play_reviews

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feedback_agent")

# ---------------------------------------------------------
# 2. CONSTANTS & CONFIG
# ---------------------------------------------------------
DATA_FILE  = "data/reviews_history.json"
CACHE_FILE = "data/analysis_cache.json"

APP_CONFIG: list[dict] = [
    {"name": "Nordkurier", "ios_id": "1250964862", "android_id": "de.nordkurier.live",      "country": "de"},
    {"name": "Schwäbische","ios_id": "432491155",  "android_id": "de.schwaebische.epaper", "country": "de"},
]

STOP_WORDS: set[str] = {
    "die","der","das","den","dem","des","ein","eine","einer","eines","einem","einen",
    "ich","du","er","sie","es","wir","ihr","mich","mir","meine","meiner","mein",
    "sich","uns","euch","ihnen","ihrem","ihres","dieser","diese","dieses","diesen",
    "und","oder","aber","als","wenn","dass","weil","denn","ob","wie","wo","was",
    "in","im","an","am","auf","aus","bei","beim","mit","nach","von","vom","zu","zum","zur",
    "über","unter","vor","hinter","neben","durch","für","gegen","ohne","um","wegen","seit",
    "ist","sind","war","wäre","wird","werden","wurde","haben","hat","hatte","habe","gibt",
    "kann","können","konnte","muss","müssen","musste","soll","sollen","sollte","will","wollen",
    "geht","ging","lassen","lässt","machen","macht","getan","sehen","sieht","schon","nun",
    "nicht","nichts","nie","wieder","immer","oft","selten","manchmal","erst","bereits",
    "noch","jetzt","heute","damals","hier","da","dort","mal","einmal","viel","sehr","auch",
    "ganz","gar","mehr","weniger","nur","doch","etwas","so","dann","wann","warum","wer",
    "einfach","leider","halt","eben","wohl","zwar","vielleicht","bestimmt","bitte","danke",
    "app","apps","anwendung","version","update","updates","ios","android","handy","tablet",
    "telefon","iphone","ipad","samsung","pixel","gerät","geräte","nutzer","kunde","kunden",
    "schwäbische","nordkurier","zeitung","artikel","lesen","leser","hallo","moin","tag",
    "also","alle","alles","viele","zeit","seit","wochen","monaten","tagen","jahre","sterne","stern",
    "läuft","funktioniert","problem","probleme","fehler","stürzt","absturz","kein","keine",
    "bin","bis","bist","dadurch","daher","dabei","darum","dein","deine","dessen","dich","dir",
    "einige","einigen","einiger","einiges","jede","jedem","jeden","jeder","jedes","jene",
    "kannst","könnt","solche","solchem","solchen","solcher","solches","sondern","sonst",
    "werde","welche","welchem","welchen","welcher","welches","wirst","wollte","würde","würden",
    "hab","hatten","indem","ins","meinem","meinen","seiner","seines","selbst","unser",
    "unsere","unserem","unseren","unserer","unseres","während","warst","weg","weiter",
}

# ---------------------------------------------------------
# 3. AI INITIALISATION
# ---------------------------------------------------------
def _init_ai():  # Type hints vereinfacht für Kompatibilität
     api_key = os.getenv("GEMINI_API_KEY")
     if not api_key:
         log.info("Kein GEMINI_API_KEY gefunden – KI deaktiviert.")
         return None, None
     try:
         client = genai.Client(api_key=api_key)
         log.info("✅ Gemini-Client geladen.")
         return client, None
     except Exception as exc:
         log.warning("KI-Start fehlgeschlagen: %s", exc)
         return None, None

model, embedder = _init_ai()

# ---------------------------------------------------------
# 4. DATA MANAGEMENT
# ---------------------------------------------------------
def _generate_id(review: dict) -> str:
    unique = f"{review.get('text','')[:50]}{review.get('date','')}{review.get('app','')}{review.get('store','')}"
    return hashlib.sha256(unique.encode()).hexdigest()

def load_history() -> dict[str, dict]:
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        return {r["id"]: r for r in raw if "id" in r}
    except (json.JSONDecodeError, OSError) as exc:
        log.error("History laden fehlgeschlagen: %s", exc)
        return {}

def save_history(history_dict: dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    data_list = sorted(history_dict.values(), key=lambda x: x["date"], reverse=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

def load_analysis_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Cache laden fehlgeschlagen: %s", exc)
        return {}

def save_analysis_cache(data: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ---------------------------------------------------------
# 5. ANALYTICS
# ---------------------------------------------------------
def calculate_trends(reviews: list[dict]) -> dict:
    today = datetime.now().date()
    dated: list[tuple] = []
    breakdown: dict = {a["name"]: {"ios": [], "android": []} for a in APP_CONFIG}
    ios_total = android_total = 0

    for r in reviews:
        store = r.get("store", "")
        if store == "ios":
            ios_total += 1
        elif store == "android":
            android_total += 1

        try:
            r_date  = datetime.strptime(r["date"], "%Y-%m-%d").date()
            rating  = float(r["rating"])
            if r.get("text"):
                dated.append((r_date, rating))
            if r["app"] in breakdown and store in breakdown[r["app"]]:
                breakdown[r["app"]][store].append(rating)
        except (KeyError, ValueError):
            continue

    def _avg(days: int) -> float:
        cutoff = today - timedelta(days=days)
        vals   = [rat for d, rat in dated if d >= cutoff]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    final_breakdown = {
        app: {
            store: round(sum(ratings) / len(ratings), 2) if ratings else 0.0
            for store, ratings in stores.items()
        }
        for app, stores in breakdown.items()
    }
    all_ratings = [float(r["rating"]) for r in reviews if r.get("rating") is not None]
    overall     = round(sum(all_ratings) / len(all_ratings), 2) if all_ratings else 0.0

    return {
        "overall":      overall,
        "last_7d":      _avg(7),
        "last_30d":     _avg(30),
        "breakdown":    final_breakdown,
        "ios_total":    ios_total,
        "android_total":android_total,
    }

def prepare_chart_data(reviews: list[dict], days: int = 14) -> dict:
    today  = datetime.now().date()
    stats  = {
        (today - timedelta(days=i)).strftime("%Y-%m-%d"): {"pos": 0, "neg": 0, "neu": 0}
        for i in range(days)
    }
    for r in reviews:
        d   = r.get("date", "")
        rat = r.get("rating")
        if d in stats and rat is not None:
            rat = float(rat)
            if rat >= 4:   stats[d]["pos"] += 1
            elif rat <= 2: stats[d]["neg"] += 1
            else:          stats[d]["neu"] += 1

    labels = sorted(stats.keys())
    fmt    = [datetime.strptime(l, "%Y-%m-%d").strftime("%d.%m.") for l in labels]
    return {
        "labels": fmt,
        "pos":    [stats[d]["pos"] for d in labels],
        "neg":    [stats[d]["neg"] for d in labels],
        "neu":    [stats[d]["neu"] for d in labels],
    }

def _is_genuine_positive(review: dict) -> bool:
    bad_words = {"absturz","stürzt","fehler","schlecht","katastrophe","mies","flackern","unbrauchbar"}
    if float(review.get("rating", 0)) < 4:
        return False
    return not any(w in review.get("text", "").lower() for w in bad_words)

# ---------------------------------------------------------
# 6. LOCAL NLP FALLBACKS
# ---------------------------------------------------------
def _local_buzzwords(reviews: list[dict]) -> list[tuple[str, int]]:
    blob   = " ".join(r.get("text", "") for r in reviews).lower()
    blob   = re.sub(r"[^\w\säöüß]", "", blob)
    words  = blob.split()
    bigrams = [
        f"{words[i].capitalize()} {words[i+1].capitalize()}"
        for i in range(len(words) - 1)
        if len(words[i]) > 3 and len(words[i+1]) > 3
        and words[i] not in STOP_WORDS and words[i+1] not in STOP_WORDS
    ]
    return Counter(bigrams).most_common(12)

def _local_topics(texts: list[str]) -> list[str]:
    words = [
        w.capitalize()
        for t in texts
        for w in re.sub(r"[^\w\s]", "", t.lower()).split()
        if w not in STOP_WORDS and len(w) > 4
    ]
    return [w for w, _ in Counter(words).most_common(5)] or ["Allgemeines Feedback"]

# ---------------------------------------------------------
# 7. HYBRID AI ANALYSIS
# ---------------------------------------------------------
def _parse_json_response(text: str) -> Any:
    """Strip possible markdown fences before JSON parsing."""
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    return json.loads(clean)

def get_ai_data_hybrid(reviews: list[dict], cache: dict) -> tuple[list, list, dict]:
    topics      = cache.get("topics", [])
    buzzwords   = cache.get("buzzwords", [])
    summary     = cache.get("summary", "")
    top_reviews = cache.get("topReviews", [])
    bot_reviews = cache.get("bottomReviews", [])

    rich = [r for r in reviews if len(r.get("text", "")) > 40] or reviews

    if model:
        try:
            log.info("Starte KI-Analyse …")
            texts = [r["text"] for r in reviews[:100] if len(r.get("text", "")) > 10]

            # ── Buzzwords ──────────────────────────────────────────────────────
                        resp_buzz = model.models.generate_content(
                            model="gemini-1.5-flash",
                            contents=f"""Analysiere die Reviews. Identifiziere die 10 häufigsten spezifischen Probleme.
            Output: JSON-Liste [ {{"term": "Thema", "count": 12}}, ... ]
            Reviews: {json.dumps(texts, ensure_ascii=False)}"""
                        )
                        data_buzz = _parse_json_response(resp_buzz.text)
                        buzzwords = [(i["term"], i["count"]) for i in data_buzz if isinstance(i, dict)]

                        # ── Deep Analysis ──────────────────────────────────────────────────
                        sample = [
                            {"text": r["text"], "rating": r["rating"], "store": r["store"], "app": r["app"]}
                            for r in rich[:50]
                        ]
                        resp_deep = model.models.generate_content(
                            model="gemini-1.5-flash",
                            contents=f"""Analysiere diese App-Reviews.
            1. Erstelle 5 kurze Themen-Cluster-Labels (JSON-Liste von Strings).
            2. Schreibe ein Management-Summary auf Deutsch (2-3 Sätze, prägnant).
            3. Wähle die 3 besten positiven und 3 kritischsten negativen Reviews aus.
            Output JSON: {{"topics":["..."],"summary":"...","topReviews":[...],"bottomReviews":[...]}}
            Daten: {json.dumps(sample, ensure_ascii=False)}"""
                        )
                        data_deep   = _parse_json_response(resp_deep.text)
                        topics      = data_deep.get("topics", [])
                        summary     = data_deep.get("summary", "")
                        top_reviews = data_deep.get("topReviews", [])
                        bot_reviews = data_deep.get("bottomReviews", [])

                        log.info("✅ KI-Analyse erfolgreich.")
                        save_analysis_cache({
                            "topics": topics, "buzzwords": buzzwords, "summary": summary,
                            "topReviews": top_reviews, "bottomReviews": bot_reviews,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                        })
                    except Exception as exc:
                        log.warning("KI-Fehler (%s) – nutze Cache/Fallback.", exc)
    # ── Fallbacks ──────────────────────────────────────────────────────────────
    if not buzzwords:
        log.info("Nutze lokalen Buzzword-Fallback.")
        buzzwords = _local_buzzwords(reviews)
    if not topics:
        log.info("Nutze lokalen Topic-Fallback.")
        topics = _local_topics([r["text"] for r in reviews[:50]])

    ki_data = {"summary": summary, "topReviews": top_reviews, "bottomReviews": bot_reviews}
    return topics, buzzwords, ki_data

# ---------------------------------------------------------
# 8. SCRAPING
# ---------------------------------------------------------
def _fetch_ios(app_name: str, app_id: str, country: str = "de", count: int = 20) -> list[dict]:
    log.info("  → iOS %s …", app_name)
    try:
        url  = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostrecent/json"
        data = requests.get(url, timeout=10).json()
        out  = []
        for e in data.get("feed", {}).get("entry", [])[:count]:
            if "content" not in e:
                continue
            text = e["content"]["label"]
            out.append({
                "store":  "ios",
                "app":    app_name,
                "rating": int(e["im:rating"]["label"]),
                "text":   text,
                "date":   e.get("updated", {}).get("label", "")[:10],
                "id":     _generate_id({"app": app_name, "store": "ios", "text": text}),
            })
        return out
    except Exception as exc:
        log.warning("iOS-Scrape fehlgeschlagen für %s: %s", app_name, exc)
        return []

def _fetch_android(app_name: str, app_id: str, country: str = "de", count: int = 20) -> list[dict]:
    log.info("  → Android %s …", app_name)
    try:
        raw, _ = play_reviews(app_id, lang=country, country=country, sort=Sort.NEWEST, count=count)
        return [
            {
                "store":  "android",
                "app":    app_name,
                "rating": r["score"],
                "text":   r["content"],
                "date":   r["at"].strftime("%Y-%m-%d"),
                "id":     _generate_id({"app": app_name, "store": "android",
                                        "date": r["at"].strftime("%Y-%m-%d"), "text": r["content"]}),
            }
            for r in raw
        ]
    except Exception as exc:
        log.warning("Android-Scrape fehlgeschlagen für %s: %s", app_name, exc)
        return []

def get_fresh_reviews(count: int = 20) -> tuple[list[dict], list[dict]]:
    hist = load_history()
    new  = []
    log.info("=== Scrape Start ===")
    for app in APP_CONFIG:
        for r in _fetch_ios(app["name"], app["ios_id"], app["country"], count) + \
                 _fetch_android(app["name"], app["android_id"], app["country"], count):
            if r["id"] not in hist:
                hist[r["id"]] = r
                new.append(r)

    log.info("%d neue Reviews gefunden.", len(new))
    return sorted(hist.values(), key=lambda x: x["date"], reverse=True), new

# ---------------------------------------------------------
# 9. HTML DASHBOARD GENERATOR
# ---------------------------------------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>App Feedback Pro</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,500;0,700;1,300&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        /* ── TOKENS ──────────────────────────────────────────── */
        :root {{
            --bg:          #f4f3ef;
            --surface:     #ffffff;
            --surface-2:   #ede9e0;
            --text:        #1a1a1a;
            --text-muted:  #6b6760;
            --border:      #d8d4cc;
            --accent:      #c84b31;
            --accent-soft: #fdf0ed;
            --green:       #22a06b;
            --red:         #e53e3e;
            --blue:        #2563eb;
            --neutral:     #94a3b8;
            --buzz-base:   200, 75, 49;
            --shadow:      0 1px 4px rgba(0,0,0,.08), 0 4px 16px rgba(0,0,0,.04);
            --radius:      10px;
        }}
        [data-theme="dark"] {{
            --bg:          #141414;
            --surface:     #1e1e1e;
            --surface-2:   #272727;
            --text:        #f0ede8;
            --text-muted:  #8a8480;
            --border:      #333;
            --accent:      #e8623f;
            --accent-soft: #2a1810;
            --green:       #2ecc91;
            --red:         #fc6a6a;
            --shadow:      0 1px 4px rgba(0,0,0,.4);
            --buzz-base:   232, 98, 63;
        }}

        /* ── RESET & BASE ──────────────────────────────────────── */
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'DM Sans', sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 28px 20px 60px;
            line-height: 1.55;
        }}
        .container {{ max-width: 1080px; margin: 0 auto; }}

        /* ── HEADER ────────────────────────────────────────────── */
        header {{
            display: flex; justify-content: space-between; align-items: flex-end;
            margin-bottom: 36px;
            border-bottom: 2px solid var(--text);
            padding-bottom: 16px;
        }}
        .logo {{ font-size: 1.4rem; font-weight: 700; letter-spacing: -.3px; }}
        .logo span {{ color: var(--accent); }}
        .header-meta {{ font-size: .8rem; color: var(--text-muted); font-family: 'DM Mono', monospace; }}
        .theme-btn {{
            background: none; border: 1.5px solid var(--border);
            color: var(--text); padding: 7px 13px; border-radius: 6px;
            cursor: pointer; font-size: .85rem; transition: border-color .2s;
        }}
        .theme-btn:hover {{ border-color: var(--accent); }}

        /* ── CARDS ─────────────────────────────────────────────── */
        .card {{
            background: var(--surface); border-radius: var(--radius);
            border: 1px solid var(--border); box-shadow: var(--shadow);
        }}
        .card-pad {{ padding: 22px; }}

        /* ── KPI GRID ───────────────────────────────────────────── */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 16px; margin-bottom: 28px;
        }}
        .kpi-card {{ padding: 22px 24px; }}
        .kpi-label {{
            font-size: .7rem; text-transform: uppercase; letter-spacing: .08em;
            color: var(--text-muted); font-weight: 700; margin-bottom: 10px;
        }}
        .kpi-val {{ font-size: 2.4rem; font-weight: 700; line-height: 1; color: var(--accent); }}
        .breakdown-row {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px 0; border-bottom: 1px solid var(--border); font-size: .9rem;
        }}
        .breakdown-row:last-child {{ border-bottom: none; padding-bottom: 0; }}
        .store-scores {{ display: flex; gap: 12px; color: var(--text-muted); font-size: .85rem; }}

        /* ── CHART ─────────────────────────────────────────────── */
        .chart-wrap {{
            padding: 22px 24px 16px; margin-bottom: 28px;
        }}
        .chart-wrap h4 {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; color: var(--text-muted); margin-bottom: 14px; font-weight: 700; }}
        .chart-inner {{ height: 230px; position: relative; }}

        /* ── SUMMARY ────────────────────────────────────────────── */
        .summary-wrap {{
            padding: 24px 28px; margin-bottom: 28px;
            border-left: 4px solid var(--accent);
        }}
        .summary-wrap h3 {{ font-size: .85rem; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 12px; color: var(--text-muted); font-weight: 700; }}
        .summary-text {{ font-size: 1rem; line-height: 1.65; color: var(--text); }}
        .summary-placeholder {{
            font-style: italic; color: var(--text-muted); font-size: .95rem;
            background: var(--surface-2); border-radius: 6px; padding: 12px 16px;
        }}

        /* ── INSIGHTS ROW ───────────────────────────────────────── */
        .insights-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 36px; }}
        @media (max-width: 640px) {{ .insights-row {{ grid-template-columns: 1fr; }} }}
        .insights-panel {{ padding: 22px; }}
        .panel-title {{ font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; color: var(--text-muted); font-weight: 700; margin-bottom: 14px; }}

        /* ── TOPIC TAGS ─────────────────────────────────────────── */
        .topic-tags {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .tag {{
            background: var(--surface-2); border: 1px solid var(--border);
            padding: 5px 13px; border-radius: 20px;
            font-size: .875rem; color: var(--text);
        }}

        /* ── BUZZ TAGS ──────────────────────────────────────────── */
        .buzz-grid {{ display: flex; flex-direction: column; gap: 6px; }}
        .buzz-item {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 7px 12px; border-radius: 6px; cursor: pointer;
            background: rgba(var(--buzz-base), calc(.03 + var(--intensity) * .12));
            border: 1px solid rgba(var(--buzz-base), calc(.1 + var(--intensity) * .4));
            transition: transform .15s, border-color .15s;
            font-size: .875rem; font-weight: 500;
        }}
        .buzz-item:hover {{ transform: translateX(3px); border-color: var(--accent); }}
        .buzz-count {{
            background: rgba(var(--buzz-base), .15);
            color: var(--accent); font-size: .75rem; font-weight: 700;
            padding: 2px 8px; border-radius: 10px; font-family: 'DM Mono', monospace;
        }}

        /* ── REVIEW CARDS ───────────────────────────────────────── */
        .voices-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }}
        @media (max-width: 640px) {{ .voices-grid {{ grid-template-columns: 1fr; }} }}
        .voices-col h3 {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; color: var(--text-muted); font-weight: 700; margin-bottom: 12px; }}

        .review-card {{
            padding: 16px 18px; border-radius: var(--radius);
            border: 1px solid var(--border); background: var(--surface);
            margin-bottom: 12px; box-shadow: var(--shadow);
        }}
        .review-card.pos {{ border-top: 3px solid var(--green); }}
        .review-card.neg {{ border-top: 3px solid var(--red); }}
        .rev-meta {{
            display: flex; justify-content: space-between; align-items: center;
            font-size: .8rem; color: var(--text-muted); margin-bottom: 10px;
            padding-bottom: 8px; border-bottom: 1px solid var(--border);
        }}
        .icon-ios   {{ color: var(--text); }}
        .icon-android {{ color: #3ddc84; }}
        .rev-body   {{ font-size: .9rem; line-height: 1.55; }}
        .rev-body.clamped {{ display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }}
        .read-more  {{ color: var(--accent); cursor: pointer; font-size: .8rem; margin-top: 6px; font-weight: 600; display: none; }}

        /* ── EXPLORER ───────────────────────────────────────────── */
        .explorer-header {{ border-top: 2px solid var(--text); padding-top: 28px; margin-bottom: 20px; }}
        .explorer-header h2 {{ font-size: 1.1rem; font-weight: 700; letter-spacing: -.2px; }}
        .search-input {{
            width: 100%; padding: 11px 16px; font-size: .95rem; font-family: 'DM Sans', sans-serif;
            border: 1.5px solid var(--border); border-radius: var(--radius);
            background: var(--surface); color: var(--text);
            margin-bottom: 12px; transition: border-color .2s;
        }}
        .search-input:focus {{ outline: none; border-color: var(--accent); }}

        .filter-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }}
        .filter-group {{
            display: flex; align-items: center; gap: 4px;
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 8px; padding: 4px 6px;
        }}
        .filter-label {{ font-size: .65rem; text-transform: uppercase; font-weight: 700; color: var(--text-muted); padding: 0 6px; }}
        .filter-btn {{
            padding: 6px 13px; border: none; background: transparent;
            color: var(--text); border-radius: 5px; cursor: pointer; font-size: .85rem;
            font-family: 'DM Sans', sans-serif; transition: background .15s;
        }}
        .filter-btn:hover {{ background: var(--surface-2); }}
        .filter-btn.active {{ background: var(--accent); color: #fff; }}

        .explorer-item {{
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 16px 18px; font-size: .875rem;
            box-shadow: var(--shadow); line-height: 1.5;
        }}
        .explorer-meta {{
            display: flex; justify-content: space-between;
            font-size: .78rem; color: var(--text-muted);
            margin-bottom: 10px; padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
            font-family: 'DM Mono', monospace;
        }}
        #list-container {{ display: grid; gap: 10px; }}

        mark {{ background: #fde68a; color: #92400e; padding: 0 2px; border-radius: 2px; }}
        [data-theme="dark"] mark {{ background: #78350f; color: #fde68a; }}

        .copy-btn {{ float: right; cursor: pointer; opacity: .4; color: var(--text); font-size: .85rem; }}
        .copy-btn:hover {{ opacity: 1; color: var(--accent); }}

        /* ── ANIMATIONS ─────────────────────────────────────────── */
        @keyframes fadeUp {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:none; }} }}
        .kpi-card, .chart-wrap, .summary-wrap, .insights-panel {{ animation: fadeUp .4s ease both; }}
        .kpi-card:nth-child(2) {{ animation-delay: .05s; }}
        .kpi-card:nth-child(3) {{ animation-delay: .1s; }}
    </style>
</head>
<body>
<div class="container">

    <!-- HEADER -->
    <header>
        <div>
            <div class="logo">App Feedback<span>.</span></div>
            <div class="header-meta">Letztes Update: {update_time}</div>
        </div>
        <button class="theme-btn" onclick="toggleTheme()">🌗</button>
    </header>

    <!-- KPIs -->
    <div class="kpi-grid">
        <div class="card kpi-card">
            <div class="kpi-label">Gesamt Ø</div>
            <div class="kpi-val">{overall} ⭐</div>
        </div>
        <div class="card kpi-card">
            <div class="kpi-label">Details pro App</div>
            {breakdown_html}
        </div>
        <div class="card kpi-card">
            <div class="kpi-label">Reviews pro Store</div>
            <div style="margin-top:12px; font-size:1rem; display:flex; gap:16px;">
                <span><i class="fab fa-apple icon-ios"></i> {ios_total}</span>
                <span><i class="fab fa-android icon-android"></i> {android_total}</span>
            </div>
        </div>
    </div>

    <!-- CHART -->
    <div class="card chart-wrap">
        <h4>Bewertungsverlauf – letzte 14 Tage</h4>
        <div class="chart-inner"><canvas id="trendChart"></canvas></div>
    </div>

    <!-- KI SUMMARY -->
    <div class="card summary-wrap">
        <h3>🤖 KI-Analyse</h3>
        {summary_html}
    </div>

    <!-- INSIGHTS -->
    <div class="insights-row">
        <div class="card insights-panel">
            <div class="panel-title">🔥 Themen-Cluster</div>
            <div class="topic-tags">{topics_html}</div>
        </div>
        <div class="card insights-panel">
            <div class="panel-title">🚨 Häufigste Probleme</div>
            <div class="buzz-grid">{buzz_html}</div>
        </div>
    </div>

    <!-- VOICES -->
    <div class="voices-grid">
        <div class="voices-col">
            <h3>👍 Top Stimmen</h3>
            {top_html}
        </div>
        <div class="voices-col">
            <h3>⚠️ Kritische Stimmen</h3>
            {bot_html}
        </div>
    </div>

    <!-- EXPLORER -->
    <div class="explorer-header">
        <h2>🔎 Explorer</h2>
    </div>
    <input type="text" class="search-input" id="search" placeholder="Suche nach Stichworten …" oninput="filterData()">
    <div class="filter-row">
        <div class="filter-group">
            <span class="filter-label">App</span>
            <button class="filter-btn active" onclick="setFilter('app','all',this)">Alle</button>
            <button class="filter-btn" onclick="setFilter('app','Nordkurier',this)">NK</button>
            <button class="filter-btn" onclick="setFilter('app','Schwäbische',this)">SZ</button>
        </div>
        <div class="filter-group">
            <span class="filter-label">Store</span>
            <button class="filter-btn active" onclick="setFilter('store','all',this)">Alle</button>
            <button class="filter-btn" onclick="setFilter('store','ios',this)"><i class="fab fa-apple"></i></button>
            <button class="filter-btn" onclick="setFilter('store','android',this)"><i class="fab fa-android"></i></button>
        </div>
        <div class="filter-group">
            <span class="filter-label">Sort</span>
            <button class="filter-btn active" onclick="setSort('newest',this)">Neu</button>
            <button class="filter-btn" onclick="setSort('best',this)">Beste</button>
            <button class="filter-btn" onclick="setSort('worst',this)">Schlechteste</button>
        </div>
    </div>
    <div id="list-container"></div>
</div>

<script>
const REVIEWS = {js_reviews};
let filterApp   = 'all';
let filterStore = 'all';
let currentSort = 'newest';

// ── THEME ──────────────────────────────────────────────────────────────────
const saved = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', saved);

function toggleTheme() {{
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateChartColors();
}}

// ── CHART ──────────────────────────────────────────────────────────────────
const ctx = document.getElementById('trendChart').getContext('2d');
const chart = new Chart(ctx, {{
    type: 'bar',
    data: {{
        labels: {js_labels},
        datasets: [
            {{ label: 'Positiv (4-5★)', data: {js_pos}, backgroundColor: '#22a06b', borderRadius: 3 }},
            {{ label: 'Neutral (3★)',   data: {js_neu}, backgroundColor: '#94a3b8', borderRadius: 3 }},
            {{ label: 'Negativ (1-2★)',data: {js_neg}, backgroundColor: '#e53e3e', borderRadius: 3 }},
        ],
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        scales: {{
            x: {{ stacked: true, grid: {{ display: false }}, ticks: {{ color: '#888' }} }},
            y: {{ stacked: true, grid: {{ color: 'rgba(128,128,128,.15)' }}, ticks: {{ color: '#888', padding: 8 }} }},
        }},
        plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#888', boxWidth: 12 }} }} }},
    }},
}});

function updateChartColors() {{
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const c = isDark ? '#888' : '#666';
    chart.options.scales.x.ticks.color = c;
    chart.options.scales.y.ticks.color = c;
    chart.options.plugins.legend.labels.color = c;
    chart.update();
}}
updateChartColors();

// ── READ-MORE (event delegation – works for dynamically rendered cards too) ──
document.addEventListener('click', e => {{
    if (e.target.classList.contains('read-more')) {{
        const body = e.target.previousElementSibling;
        body.classList.toggle('clamped');
        e.target.textContent = body.classList.contains('clamped') ? 'Mehr anzeigen' : 'Weniger anzeigen';
    }}
}});

// Initialise read-more buttons for static cards (top/bot voices)
function initStaticReadMore() {{
    document.querySelectorAll('.rev-body.clamped').forEach(el => {{
        requestAnimationFrame(() => {{
            const btn = el.nextElementSibling;
            if (btn && el.scrollHeight > el.clientHeight + 4) btn.style.display = 'block';
        }});
    }});
}}
initStaticReadMore();

// ── FILTERS ────────────────────────────────────────────────────────────────
function setFilter(type, value, btn) {{
    if (type === 'app')   filterApp   = value;
    if (type === 'store') filterStore = value;
    btn.parentElement.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    filterData();
}}
function setSort(mode, btn) {{
    currentSort = mode;
    btn.parentElement.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    filterData();
}}
function setSearch(term) {{
    document.getElementById('search').value = term;
    filterData();
    document.getElementById('search').scrollIntoView({{ behavior: 'smooth' }});
}}
function copyText(text) {{
    navigator.clipboard.writeText(text).then(() => {{
        // subtle visual feedback instead of alert()
        const el = event.target;
        el.classList.replace('fa-copy', 'fa-check');
        setTimeout(() => el.classList.replace('fa-check', 'fa-copy'), 1500);
    }});
}}

// ── EXPLORER RENDER ────────────────────────────────────────────────────────
function escapeHtml(str) {{
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function filterData() {{
    const q   = document.getElementById('search').value.toLowerCase().trim();
    const terms = q.length >= 2 ? q.split(' ').filter(t => t.length > 1) : [];
    const container = document.getElementById('list-container');
    container.innerHTML = '';

    let filtered = REVIEWS.filter(r => {{
        if (filterApp   !== 'all' && r.app   !== filterApp)   return false;
        if (filterStore !== 'all' && r.store !== filterStore) return false;
        if (terms.length && !terms.every(t => (r.text + r.app + r.store).toLowerCase().includes(t))) return false;
        return true;
    }});

    if (currentSort === 'newest') filtered.sort((a,b) => a.date < b.date ? 1 : -1);
    if (currentSort === 'best')   filtered.sort((a,b) => b.rating - a.rating);
    if (currentSort === 'worst')  filtered.sort((a,b) => a.rating - b.rating);

    if (!filtered.length) {{
        container.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:30px;">Keine Ergebnisse</div>';
        return;
    }}

    filtered.slice(0, 50).forEach(r => {{
        const icon = r.store === 'ios'
            ? '<i class="fab fa-apple icon-ios"></i>'
            : '<i class="fab fa-android icon-android"></i>';

        let displayText = escapeHtml(r.text);
        if (terms.length) {{
            terms.forEach(term => {{
                const rx = new RegExp('(' + term.replace(/[.*+?^${{}}()|[\\]\\\\]/g,'\\\\$&') + ')', 'gi');
                displayText = displayText.replace(rx, '<mark>$1</mark>');
            }});
        }}

        const div = document.createElement('div');
        div.className = 'explorer-item';
        div.innerHTML = `
            <div class="explorer-meta">
                <span>${{icon}} <strong>${{r.app}}</strong> (${{r.store.toUpperCase()}}) · ${{r.rating}}⭐</span>
                <span>${{r.fmt_date || r.date}}</span>
            </div>
            <div>
                <span class="rev-body clamped">${{displayText}}</span>
                <span class="read-more">Mehr anzeigen</span>
                <i class="fas fa-copy copy-btn" title="Kopieren" onclick="copyText(${{JSON.stringify(r.text)}})"></i>
            </div>
        `;
        // show/hide read-more
        requestAnimationFrame(() => {{
            const body = div.querySelector('.rev-body');
            const btn  = div.querySelector('.read-more');
            if (body.scrollHeight > body.clientHeight + 4) btn.style.display = 'block';
        }});
        container.appendChild(div);
    }});
}}
filterData();
</script>
</body>
</html>"""


def _render_review_card(r: dict, cls: str) -> str:
    icon = '<i class="fab fa-apple icon-ios"></i>' if r.get("store") == "ios" else '<i class="fab fa-android icon-android"></i>'
    return f"""
    <div class="review-card {cls}">
        <div class="rev-meta">
            <span>{icon} <strong>{r.get('app', '–')}</strong></span>
            <span>{r.get('rating', '?')}★</span>
        </div>
        <div class="rev-body clamped">{r.get('text', '')}</div>
        <span class="read-more">Mehr anzeigen</span>
    </div>"""


def run_analysis_and_generate_html(full_history: list[dict], new_only: list[dict]) -> None:
    trends = calculate_trends(full_history)
    chart  = prepare_chart_data(full_history)
    cache  = load_analysis_cache()
    topics, buzzwords, ki_data = get_ai_data_hybrid(full_history, cache)

    # ── Format dates ──────────────────────────────────────────────────────
    for r in full_history:
        try:
            r["fmt_date"] = datetime.strptime(r["date"], "%Y-%m-%d").strftime("%d.%m.%Y")
        except (ValueError, KeyError):
            r["fmt_date"] = r.get("date", "")

    # ── Build top / bottom voice lists ────────────────────────────────────
    seen: set[str] = set()
    top_list, bot_list = [], []

    for r in sorted(
        [r for r in full_history if r.get("rating", 0) >= 4 and _is_genuine_positive(r)],
        key=lambda x: len(x["text"]), reverse=True,
    ):
        if len(top_list) >= 3: break
        if r["text"] not in seen:
            top_list.append(r); seen.add(r["text"])

    for r in sorted(
        [r for r in full_history if r.get("rating", 0) <= 2],
        key=lambda x: len(x["text"]), reverse=True,
    ):
        if len(bot_list) >= 3: break
        if r["text"] not in seen:
            bot_list.append(r); seen.add(r["text"])

    # ── Build HTML snippets ───────────────────────────────────────────────
    breakdown_html = "".join(
        f"""<div class="breakdown-row">
            <strong>{app}</strong>
            <div class="store-scores">
                <span><i class="fab fa-apple"></i> {stores.get('ios', 0)}⭐</span>
                <span><i class="fab fa-android icon-android"></i> {stores.get('android', 0)}⭐</span>
            </div>
        </div>"""
        for app, stores in trends["breakdown"].items()
    )

    summary_text = str(ki_data.get("summary", "")).strip().replace("{", "").replace("}", "")
    summary_html = (
        f'<p class="summary-text">{summary_text}</p>'
        if summary_text
        else '<p class="summary-placeholder">Noch keine KI-Analyse – wird beim nächsten Run mit aktivem API-Key erstellt.</p>'
    )

    topics_html = "".join(f'<span class="tag"># {t}</span>' for t in topics)

    max_c = buzzwords[0][1] if buzzwords else 1
    buzz_html = "".join(
        f"""<div class="buzz-item" style="--intensity:{min(1.0, w[1]/max_c):.2f};"
             onclick="setSearch('{w[0]}')">
            {w[0]} <span class="buzz-count">{w[1]}</span>
        </div>"""
        for w in buzzwords
    )

    top_html = "".join(_render_review_card(r, "pos") for r in top_list[:3])
    bot_html = "".join(_render_review_card(r, "neg") for r in bot_list[:3])

    # ── Render final HTML ─────────────────────────────────────────────────
    html = _HTML_TEMPLATE.format(
        update_time    = datetime.now().strftime("%d.%m.%Y %H:%M"),
        overall        = trends["overall"],
        ios_total      = trends["ios_total"],
        android_total  = trends["android_total"],
        breakdown_html = breakdown_html,
        summary_html   = summary_html,
        topics_html    = topics_html,
        buzz_html      = buzz_html,
        top_html       = top_html,
        bot_html       = bot_html,
        js_reviews     = json.dumps(full_history, ensure_ascii=False),
        js_labels      = json.dumps(chart["labels"]),
        js_pos         = json.dumps(chart["pos"]),
        js_neg         = json.dumps(chart["neg"]),
        js_neu         = json.dumps(chart["neu"]),
    )

    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    log.info("✅ Dashboard generiert → public/index.html")

# ---------------------------------------------------------
# 10. TEAMS NOTIFICATION
# ---------------------------------------------------------
def send_teams_notification(new_reviews: list[dict], webhook_url: str) -> None:
    if not new_reviews:
        return

    pos = sum(1 for r in new_reviews if r["rating"] >= 4)
    neu = sum(1 for r in new_reviews if r["rating"] == 3)
    neg = sum(1 for r in new_reviews if r["rating"] <= 2)

    card = {
        "@type":    "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "c84b31",
        "summary":  f"Neues App Feedback ({len(new_reviews)})",
        "sections": [
            {
                "activityTitle": f"🚀 **Neues Feedback ({len(new_reviews)})**",
                "facts": [
                    {"name": "Positiv:",  "value": str(pos)},
                    {"name": "Neutral:",  "value": str(neu)},
                    {"name": "Negativ:",  "value": str(neg)},
                ],
                "markdown": True,
            }
        ],
        "potentialAction": [
            {
                "@type": "OpenUri",
                "name":  "Zum Dashboard",
                "targets": [{"os": "default", "uri": "https://Hatozoro.github.io/feedback-agent/"}],
            }
        ],
    }

    for r in new_reviews[:10]:
        icon = "🍏" if r["store"] == "ios" else "🤖"
        card["sections"].append({
            "title": f"{icon} {r['app']} ({'⭐' * r['rating']})",
            "text":  r["text"],
            "markdown": True,
        })

    if len(new_reviews) > 10:
        card["sections"].append({
            "text": f"… und {len(new_reviews) - 10} weitere auf dem Dashboard.",
            "markdown": True,
        })

    try:
        response = requests.post(webhook_url, json=card, timeout=10)
        response.raise_for_status()
        log.info("✅ Teams-Nachricht gesendet.")
    except requests.RequestException as exc:
        log.error("Teams-Fehler: %s", exc)


# ---------------------------------------------------------
# 11. ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    full, new = get_fresh_reviews()
    save_history({r["id"]: r for r in full})
    run_analysis_and_generate_html(full, new)

    webhook = os.getenv("TEAMS_WEBHOOK_URL")
    if webhook:
        send_teams_notification(new, webhook)

    log.info("✅ Durchlauf beendet.")