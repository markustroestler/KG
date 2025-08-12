# src/logic/analyse_trend.py
from pathlib import Path
from datetime import date
import re
import yaml
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import XSD, RDF

EX = Namespace("http://example.org/")

def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def resolve_graph_path(cfg: dict, default="data/kg_assets.ttl") -> str:
    # bevorzugt graph.out, sonst graph.in, sonst Default
    return (
        cfg.get("graph", {}).get("out")
        or cfg.get("graph", {}).get("in")
        or default
    )

def tickers_from_config(cfg: dict) -> list[str]:
    wl = cfg.get("watchlist", {}) or {}
    stocks = wl.get("stocks") or cfg.get("stocks") or []
    return [str(t).upper() for t in stocks]

def query_prices(g: Graph, ticker: str) -> list[tuple[date, float]]:
    q = f"""
    PREFIX ex: <http://example.org/>
    SELECT ?d ?c
    WHERE {{
      ?p a ex:Price ;
         (ex:ofAsset|ex:ofCompany|ex:forCompany|ex:ofSymbol|ex:forSymbol) ex:{ticker} ;
         ex:onDate ?d ;
         (ex:closePrice|ex:close) ?c .
    }}
    ORDER BY ASC(?d)
    """
    rows = list(g.query(q))
    out = []
    for d, c in rows:
        d_py = d.toPython()
        if hasattr(d_py, "date"):          # datetime -> date
            d_py = d_py.date()
        elif isinstance(d_py, str):         # "2025-07-09T00:00:00" -> 2025-07-09
            d_py = date.fromisoformat(d_py[:10])
        out.append((d_py, float(c)))
    return out

def main():
    cfg = load_config(Path("config/config.yaml"))

    kg_path   = resolve_graph_path(cfg)
    out_path  = Path(cfg.get("trend", {}).get("out", "data/kg_trends.ttl"))
    append    = bool(cfg.get("trend", {}).get("append", True))

    g_src = Graph(); g_src.parse(kg_path, format="turtle")

    # Tickerliste (Config > Discovery)
    tickers = tickers_from_config(cfg)
    if not tickers:
        discover_q = """
        PREFIX ex: <http://example.org/>
        SELECT DISTINCT ?s WHERE { { ?s a ex:Company } UNION { ?s a ex:Index } }
        """
        tickers = [str(row[0]).split("/")[-1] for row in g_src.query(discover_q)]

    g_out = Graph(); g_out.bind("ex", EX)
    # optionale Schema-Hinweise
    g_out.add((EX.DailyTrend, RDF.type, EX.Class))
    g_out.add((EX.Up,   RDF.type, EX.Direction))
    g_out.add((EX.Down, RDF.type, EX.Direction))
    g_out.add((EX.Flat, RDF.type, EX.Direction))

    print(f"\n🧪 Erzeuge DailyTrends für {len(tickers)} Assets…\n")

    for tk in tickers:
        tk_local = slug(tk)
        prices = query_prices(g_src, tk_local)
        if len(prices) < 2:
            print(f"⏭️  {tk}: zu wenig Punkte")
            continue

        asset = URIRef(EX[tk_local])

        # paarweise Deltas (gestern->heute)
        for (d_prev, c_prev), (d_curr, c_curr) in zip(prices[:-1], prices[1:]):
            if d_curr <= d_prev:
                continue
            change = c_curr - c_prev
            pct = (change / c_prev * 100.0) if c_prev else 0.0
            if c_curr > c_prev:
                dir_uri, sym = EX.Up, "+"
            elif c_curr < c_prev:
                dir_uri, sym = EX.Down, "−"
            else:
                dir_uri, sym = EX.Flat, "="

            dt_uri = URIRef(EX[f"{tk_local}_dtrend_{d_curr.isoformat()}"])

            g_out.add((asset, EX.hasDailyTrend, dt_uri))
            g_out.add((dt_uri, RDF.type, EX.DailyTrend))
            g_out.add((dt_uri, EX.aboutAsset, asset))
            g_out.add((dt_uri, EX.fromDate, Literal(d_prev, datatype=XSD.date)))
            g_out.add((dt_uri, EX.toDate,   Literal(d_curr, datatype=XSD.date)))
            g_out.add((dt_uri, EX.direction, dir_uri))
            g_out.add((dt_uri, EX.symbol, Literal(sym, datatype=XSD.string)))
            g_out.add((dt_uri, EX.absChange, Literal(round(change, 6), datatype=XSD.float)))
            g_out.add((dt_uri, EX.pctChange, Literal(round(pct, 6), datatype=XSD.float)))

    # speichern
    if append and out_path.exists():
        base = Graph(); base.parse(out_path, format="turtle")
        for t in g_out: base.add(t)
        base.serialize(destination=out_path, format="turtle")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        g_out.serialize(destination=out_path, format="turtle")

    print(f"\n✅ DailyTrends gespeichert in {out_path}")

if __name__ == "__main__":
    main()
