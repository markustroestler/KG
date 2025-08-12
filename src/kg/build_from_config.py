# src/kg/build_from_config.py
import re, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yaml
import yfinance as yf
import pandas as pd
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD

EX = Namespace("http://example.org/")

CFG_PATH_DEFAULT = Path("config/config.yaml")
OUT_PATH_DEFAULT = Path("data/kg_assets.ttl")

def load_cfg(path: Path = CFG_PATH_DEFAULT) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def fetch_prices(ticker: str, start=None, end=None, interval="1d") -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end, interval=interval,
        auto_adjust=False, progress=False, ignore_tz=True,
    )
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=-1)
        except Exception:
            df.columns = [c[-1] if isinstance(c, tuple) else str(c) for c in df.columns]

    colmap = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ("close", "adj close", "adjclose"): colmap["close"] = c
        elif lc == "open":   colmap["open"] = c
        elif lc == "high":   colmap["high"] = c
        elif lc == "low":    colmap["low"] = c
        elif lc == "volume": colmap["volume"] = c
    for req in ["close", "open", "high", "low", "volume"]:
        if req not in colmap:
            raise ValueError(f"Missing {req}; got {list(df.columns)}")

    out = df[[colmap["close"], colmap["high"], colmap["low"], colmap["open"], colmap["volume"]]].copy()
    out.columns = ["Close", "High", "Low", "Open", "Volume"]

    out = out.reset_index()
    if "Date" not in out.columns:
        out.rename(columns={out.columns[0]: "Date"}, inplace=True)
    out = out.dropna(subset=["Close", "Open", "High", "Low", "Volume"])
    return out

def add_prices_to_graph(g: Graph, symbol: str, cls: URIRef, df: pd.DataFrame):
    local = slug(symbol)
    asset_uri = URIRef(EX[local])
    g.add((asset_uri, RDF.type, cls))
    g.add((asset_uri, EX.symbol, Literal(symbol)))

    for _, row in df.iterrows():
        # stelle sicher, dass es ein reines Datum ist
        d = row["Date"]
        d = d.date() if hasattr(d, "date") else d  # pandas.Timestamp -> date
        date_str = str(d)
        price_uri = URIRef(EX[f"{local}_{date_str}"])

        g.add((price_uri, RDF.type, EX.Price))
        g.add((price_uri, EX.onDate, Literal(d, datatype=XSD.date)))
        g.add((price_uri, EX.ofAsset, asset_uri))
        g.add((price_uri, EX.openPrice,  Literal(float(row["Open"]),  datatype=XSD.float)))
        g.add((price_uri, EX.highPrice,  Literal(float(row["High"]),  datatype=XSD.float)))
        g.add((price_uri, EX.lowPrice,   Literal(float(row["Low"]),   datatype=XSD.float)))
        g.add((price_uri, EX.closePrice, Literal(float(row["Close"]), datatype=XSD.float)))
        vol = float(row["Volume"])
        if not math.isnan(vol):
            g.add((price_uri, EX.volume, Literal(int(vol), datatype=XSD.integer)))

def _resolve_output(cfg: dict) -> Path:
    # akzeptiere graph.out | graph.in | fallback
    return Path(
        cfg.get("graph", {}).get("stocks")
        or str(OUT_PATH_DEFAULT)
    )

def _tickers_from_cfg(cfg: dict) -> list[str]:
    stocks = cfg.get("stocks") or cfg.get("stocks") or []
    return [str(t).upper() for t in stocks]

def _indices_from_cfg(cfg: dict) -> list[str]:
    idx = cfg.get("indices") or cfg.get("indices") or []
    return [str(t) for t in idx]

def build_kg_from_config(cfg: dict) -> Path:
    fetch_cfg = cfg.get("fetch", {})  # optionaler Block
    lookback_days = int(fetch_cfg.get("lookback_days", cfg.get("lookback_days", 90)))
    interval      = str(fetch_cfg.get("interval",      cfg.get("interval", "1d")))

    tickers = _tickers_from_cfg(cfg)
    indices = _indices_from_cfg(cfg)
    out_path = _resolve_output(cfg)

    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()

    g = Graph()
    g.bind("ex", EX)

    # minimale Schema-Hinweise (optional)
    g.add((EX.Company, RDF.type, EX.Class))
    g.add((EX.Index,   RDF.type, EX.Class))
    g.add((EX.Price,   RDF.type, EX.Class))

    # equities
    for t in tickers:
        df = fetch_prices(t, start=start, interval=interval)
        if df.empty:
            continue
        add_prices_to_graph(g, t, EX.Company, df)

    # indices
    for idx in indices:
        df = fetch_prices(idx, start=start, interval=interval)
        if df.empty:
            continue
        add_prices_to_graph(g, idx, EX.Index, df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format="turtle")
    print(f"\n✅ {len(g)} triples saved → {out_path}")
    return out_path

# kompatibler Entry-Point ohne CLI-Args
def build_kg_from_djia():
    cfg = load_cfg()
    return build_kg_from_config(cfg)

def main():
    build_kg_from_djia()
