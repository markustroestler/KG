# src/news/fetch_news.py
import json, time, hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable

import yfinance as yf
import newspaper
import yaml

# -------------------- Config --------------------

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def tickers_from_config(cfg: dict) -> list[str]:
    stocks = cfg.get("stocks") or []
    # optional: indices auch zulassen
    indices = cfg.get("indices") or []
    tickers = [*stocks, *indices]
    return [str(t).upper() for t in tickers]

def news_settings(cfg: dict) -> dict:
    n = cfg.get("news", {}) or {}
    return {
        "count": int(n.get("count", 25)),
        "sleep": float(n.get("sleep", 0.8)),
        "lang": str(n.get("lang", "en")),
        "out_dir": Path(n.get("out_dir", "data/news/raw")),
        "only_stocks": bool(n.get("only_stocks", True)),  # indices skippen?
    }

# -------------------- Utils --------------------

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    rid = rec.get("id") or (sha1(rec["url"]) if rec.get("url") else None)
                    if rid:
                        ids.add(rid)
                except Exception:
                    continue
    return ids

def fetch_fulltext(url: str, lang="en", timeout=15):
    # newspaper nutzt Requests mit Default UA – klappt für viele, nicht alle
    art = newspaper.Article(url=url, language=lang, memoize_articles=False)
    art.download()
    art.parse()
    return {
        "title_extracted": art.title or None,
        "text": art.text or "",
    }

def normalize_news_item(ticker: str, item: dict):
    c = item.get("content", {}) or {}
    # URL robust extrahieren
    url = None
    for key_path in [
        ("canonicalUrl", "url"),
        ("clickThroughUrl", "url"),
        ("link",),
        ("provider", "link"),
        ("relatedTickers",),  # not a url, skip automatically
    ]:
        try:
            v = c
            for k in key_path:
                v = v.get(k)
            if isinstance(v, str) and v.startswith("http"):
                url = v
                break
        except Exception:
            pass
    if not url and isinstance(item.get("link"), str) and item["link"].startswith("http"):
        url = item["link"]

    published = c.get("pubDate") or c.get("displayTime") or item.get("publisher")
    source = (c.get("provider") or {}).get("displayName") or (item.get("publisher") or {})

    title = c.get("title") or item.get("title")
    summary = c.get("summary") or c.get("description") or item.get("summary")

    rid = item.get("id") or (sha1(url) if url else None)

    return {
        "id": rid,
        "ticker": ticker,
        "title": title,
        "summary": summary,
        "url": url,
        "published_at": published,  # i.d.R. ISO8601 von Yahoo
        "source": source if isinstance(source, str) else None,
    }

# -------------------- Main --------------------

def process_ticker(ticker: str, cfg_news: dict):
    out_dir: Path = cfg_news["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{ticker.lower()}.jsonl"
    existing_ids = load_existing_ids(out_path)

    y = yf.Ticker(ticker)
    try:
        news_list = y.get_news(count=cfg_news["count"]) or []
    except Exception as e:
        return

    new_written = 0
    # Yahoo Cache Pfad setzen (verhindert Permission-Probleme in einigen Umgebungen)
    yf.set_tz_cache_location("src/news/yahoo_finance_cache")

    with out_path.open("a", encoding="utf-8") as f:
        for item in news_list:
            rec = normalize_news_item(ticker, item)
            rec_id = rec.get("id") or (sha1(rec["url"]) if rec.get("url") else None)
            if not rec_id or rec_id in existing_ids:
                continue

            # Volltext (best effort)
            fulltext_ok = False
            err = None
            if rec.get("url"):
                try:
                    full = fetch_fulltext(rec["url"], lang=cfg_news["lang"])
                    rec["text"] = full["text"]
                    # rec["title_extracted"] = full["title_extracted"]
                    fulltext_ok = bool(rec["text"])
                except Exception as e:
                    err = str(e)

            rec["fulltext_ok"] = fulltext_ok
            rec["error"] = err
            rec["fetched_at"] = datetime.now(timezone.utc).isoformat()

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing_ids.add(rec_id)
            new_written += 1

            time.sleep(cfg_news["sleep"])


def main():
    cfg = load_config(Path("config/config.yaml"))
    cfg_news = news_settings(cfg)

    tickers = tickers_from_config(cfg)
    if cfg_news["only_stocks"]:
        # optional: wenn du Indices in watchlist hast und die für News nicht willst
        tickers = [t.upper() for t in cfg.get("stocks", [])]

    if not tickers:
        print("⚠️ Keine Ticker in config.yaml gefunden (config.stocks / indices).")
        return

    print(f"\n🗞️  Hole News für {len(tickers)} Ticker …\n")
    for t in tickers:
        process_ticker(t, cfg_news)

if __name__ == "__main__":
    main()
