from pathlib import Path
import json, hashlib, re
from datetime import datetime
import yaml
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import XSD, RDF

EX = Namespace("http://example.org/")

def load_cfg() -> dict:
    with Path("config/config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def parse_dt_literals(s: str):
    if not s:
        return None, None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None, None
    return Literal(dt, datatype=XSD.dateTime), Literal(dt.date(), datatype=XSD.date)

SENT_MAP = {
    "positive": EX.Positive,
    "neutral":  EX.Neutral,
    "negative": EX.Negative,
}

def main():
    cfg = load_cfg()
    proc_dir = Path(cfg.get("news", {}).get("processed_dir", "data/news/processed"))
    kg_out   = Path(cfg.get("news", {}).get("kg_out", "data/kg_news.ttl"))
    append   = bool(cfg.get("news", {}).get("append", True))

    files = sorted(proc_dir.glob("*.jsonl"))
    if not files:
        print(f"⚠️ keine Dateien in {proc_dir}")
        return

    g_news = Graph()
    g_news.bind("ex", EX)
    g_news.bind("xsd", XSD)

    # minimale Klassen (optional)
    g_news.add((EX.News, RDF.type, EX.Class))
    g_news.add((EX.Positive, RDF.type, EX.Sentiment))
    g_news.add((EX.Neutral,  RDF.type, EX.Sentiment))
    g_news.add((EX.Negative, RDF.type, EX.Sentiment))

    added = 0
    for fp in files:
        ticker = Path(fp).stem.upper()
        asset = URIRef(EX[slug(ticker)])

        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                rid  = rec.get("id") or sha1(rec.get("url") or "")
                nuri = URIRef(EX[f"news_{slug(ticker)}_{sha1(rid)[:12]}"])

                # Typ + Links
                g_news.add((nuri, RDF.type, EX.News))
                g_news.add((nuri, EX.aboutAsset, asset))
                g_news.add((asset, EX.hasNews, nuri))

                # Sentiment
                sent = str(rec.get("sentiment") or "neutral").lower()
                g_news.add((nuri, EX.sentiment, SENT_MAP.get(sent, EX.Neutral)))
                try:
                    conf = float(rec.get("confidence") or 0.0)
                except Exception:
                    conf = 0.0
                g_news.add((nuri, EX.sentimentScore, Literal(conf, datatype=XSD.float)))

                # Metadaten
                url = rec.get("url")
                if url:
                    g_news.add((nuri, EX.url, Literal(url, datatype=XSD.anyURI)))

                # Datum/Zeit: BOTH dateTime + date
                pub_dt, pub_d = parse_dt_literals(rec.get("published_at"))
                if pub_dt:
                    g_news.add((nuri, EX.publishedAt, pub_dt))     # xsd:dateTime
                if pub_d:
                    g_news.add((nuri, EX.publishedDate, pub_d))    # xsd:date

                title = rec.get("title") or rec.get("title_extracted")
                if title:
                    g_news.add((nuri, EX.title, Literal(title, datatype=XSD.string)))
                summary = rec.get("summary")
                if summary:
                    g_news.add((nuri, EX.summary, Literal(summary, datatype=XSD.string)))

                added += 1

    if append and kg_out.exists():
        base = Graph(); base.parse(kg_out, format="turtle")
        for t in g_news:
            base.add(t)  # Set-Semantik → keine Duplikate
        base.serialize(destination=kg_out, format="turtle")
    else:
        kg_out.parent.mkdir(parents=True, exist_ok=True)
        g_news.serialize(destination=kg_out, format="turtle")

    print(f"✅ {added} News-Knoten nach {kg_out} geschrieben")

if __name__ == "__main__":
    main()
