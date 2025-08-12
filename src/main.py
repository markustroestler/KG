# main.py
import sys
from pathlib import Path
from rdflib import Graph

# Modulepfad
sys.path.append("src")

# Steps importieren (jeweils die vorhandenen main()/run()-Funktionen)
from embeddings.calibrate_scores import main as calibrate_scores
from recommendations.recommend import run as run_reco
from kg.build_from_config import main as build_kg
from news.fetch_news import main as fetch_news          # liest config/config.yaml
from news.process_news import main as process_news      # dein FinBERT-Script
from news.news_to_kg import main as write_news_to_kg    # News -> KG
from logic.analyse_trend import main as analyse_trend   # Trends -> KG
from datetime import date, timedelta
from features.indicators_to_kg import main as write_indicators_to_kg
from fundamentals.fetch_fundamentals import main as fetch_fundamentals
from embeddings.export_triples import main as export_kge_triples
from embeddings.train_kge import main as train_kge
from embeddings.score_up_today import main as score_up_today

def load_all_graphs(cfg: dict) -> Graph:
    g = Graph()
    kg_assets = Path(cfg.get("graph", {}).get("in") or cfg.get("graph", {}).get("out") or "data/kg_assets.ttl")
    kg_trends = Path(cfg.get("trend", {}).get("out", "data/kg_trends.ttl"))
    kg_news   = Path(cfg.get("news",  {}).get("kg_out", "data/kg_news.ttl"))
    kg_inds  = Path(cfg.get("features",{}).get("indicators_out","data/kg_indicators.ttl"))
    kg_funda = Path(cfg.get("fundamentals",{}).get("out","data/kg_fundamentals.ttl"))
    if kg_assets.exists(): g.parse(str(kg_assets), format="turtle")
    if kg_trends.exists(): g.parse(str(kg_trends), format="turtle")
    if kg_news.exists():   g.parse(str(kg_news),   format="turtle")
    if kg_inds.exists():  g.parse(str(kg_inds),  format="turtle")
    if kg_funda.exists(): g.parse(str(kg_funda), format="turtle")
    return g

def report_interesting(g, days: int = 30, strong_thresh: float = 0.6):
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    # zwei schlanke Queries -> in Python joinen
    spark_by_asset = fetch_spark(g, cutoff)
    news_by_asset  = fetch_news_counts(g, cutoff)

    assets = sorted(set(spark_by_asset) | set(news_by_asset))

    print(f"\n📊 Entwicklung & News (letzte {days} Tage):")
    for a in assets:
        spark = spark_by_asset.get(a, "")
        pos, neu, neg = news_by_asset.get(a, (0, 0, 0))
        trend = infer_trend_from_spark(spark)
        print(
            f"  {a:<10} "
            f"spark={spark:<{days}}  "
            f"trend={trend:<7}  "
            f"pos={pos:>2}  neu={neu:>2}  neg={neg:>2}"
        )

def fetch_spark(g, cutoff_iso: str):
    q = f"""
    PREFIX ex:  <http://example.org/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?asset (GROUP_CONCAT(?sym; separator="") AS ?spark)
    WHERE {{
      {{
        SELECT ?asset ?d (SAMPLE(?s0) AS ?sym)
        WHERE {{
          ?t a ex:DailyTrend ;
             ex:aboutAsset ?asset ;
             ex:toDate ?dRaw ;
             ex:symbol ?s0 .
          # robust: dateTime -> date
          BIND(STRDT(SUBSTR(STR(?dRaw),1,10), xsd:date) AS ?d)
          FILTER(?d >= "{cutoff_iso}"^^xsd:date)
        }}
        GROUP BY ?asset ?d
        ORDER BY ?asset ?d
      }}
    }}
    GROUP BY ?asset
    ORDER BY ?asset
    """
    return {str(a).split("/")[-1]: str(s or "") for a, s in g.query(q)}

def fetch_news_counts(g, cutoff_iso: str):
    q = f"""
    PREFIX ex:  <http://example.org/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?asset
           (SUM(IF(?sent = ex:Positive, 1, 0)) AS ?pos)
           (SUM(IF(?sent = ex:Neutral,  1, 0)) AS ?neu)
           (SUM(IF(?sent = ex:Negative, 1, 0)) AS ?neg)
    WHERE {{
      {{ ?asset ex:hasNews ?n }} UNION {{ ?n ex:aboutAsset ?asset }} .
      ?n ex:publishedDate ?dN ; ex:sentiment ?sent .
      BIND(STRDT(SUBSTR(STR(?dN),1,10), xsd:date) AS ?d)
      FILTER(?d >= "{cutoff_iso}"^^xsd:date)
    }}
    GROUP BY ?asset
    ORDER BY ?asset
    """
    return {
        str(a).split("/")[-1]: (int(pos), int(neu), int(neg))
        for a, pos, neu, neg in g.query(q)
    }

def infer_trend_from_spark(spark: str) -> str:
    """
    Bestimme Trend aus Sparkline.
    Mehr '+' als '−' => Up, mehr '−' => Down, sonst Neutral.
    """
    if not spark:
        return "Neutral"
    s = spark.replace("-", "−")  # vereinheitlichen
    plus  = s.count("+")
    minus = s.count("−")
    if plus > minus:
        return "Up"
    elif minus > plus:
        return "Down"
    else:
        return "Neutral"

def main():
    # print("🧱 Baue Knowledge Graph aus DJIA-Daten…")
    # build_kg()
    # print("✅ KG gebaut.")

    # print("🗞  Fetch Yahoo News…")
    # fetch_news()
    # print("✅ News geladen (raw).")

    # print("🧪 FinBERT Sentiment…")
    # process_news()
    # print("✅ News verarbeitet (processed).")

    # print("🔗 Schreibe News ins KG…")
    # write_news_to_kg()
    # print("✅ News im KG.")

    # print("📈 Analysiere Trends…")
    # analyse_trend()
    # print("✅ Trends im KG.")



    # # Bericht
    # import yaml
    # cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}
    # days = int(cfg.get("report", {}).get("news_window_days", 7))
    # strong = float(cfg.get("report", {}).get("strong_threshold", 0.6))

    # g_all = load_all_graphs(cfg)
    # print("🧮 Schreibe technische Indikatoren…")
    # write_indicators_to_kg()
    # print("✅ Indikatoren im KG.")

    # print("🏛️  Lade Fundamentals & Events…")
    # fetch_fundamentals()
    # print("✅ Fundamentals/Events im KG.")
    # report_interesting(g_all, days=days, strong_thresh=strong)
    # run_reco(g_all)   # <= new line

    # print("🧮 Export triples for embeddings…")
    # export_kge_triples()

    print("🧠 Train KGE model…")
    train_kge()

    print("🎯 Calibrate KGE scores to probabilities…")
    calibrate_scores()

    print("🔮 Score candidates for today…")
    score_up_today()


if __name__ == "__main__":
    main()
