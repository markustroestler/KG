from pathlib import Path
import re, yaml, math, datetime as dt
import yfinance as yf
import pandas as pd
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import XSD, RDF

EX = Namespace("http://example.org/")
def slug(s:str)->str: return re.sub(r"[^A-Za-z0-9]+","_",str(s)).strip("_")
def load_cfg(): return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}


def add_earnings_events(g, y, tk: str, asset):
    """
    Fill ex:EarningsEvent using multiple yfinance fallbacks:
    - get_earnings_dates()/earnings_dates()  (hist + future)
    - calendar['Earnings Date']              (next only)
    """
    from rdflib import Literal, URIRef
    from rdflib.namespace import XSD, RDF
    EX = Namespace("http://example.org/")

    df = None
    # 1) Try the tabular earnings dates API (names differ by yfinance version)
    try:
        if hasattr(y, "get_earnings_dates"):
            df = y.get_earnings_dates(limit=16)
        else:
            df = y.earnings_dates(limit=16)
    except Exception:
        df = None

    def _emit_row(date_val, eps_est=None, eps_act=None, surprise=None):
        if date_val is None: return
        d = pd.to_datetime(date_val).date()
        ev = URIRef(EX[f"{slug(tk)}_earn_{d}"])
        g.add((ev, RDF.type, EX.EarningsEvent))
        g.add((ev, EX.aboutAsset, asset))
        g.add((ev, EX.onDate, Literal(d, datatype=XSD.date)))
        if eps_est is not None and pd.notna(eps_est):
            g.add((ev, EX.epsEstimate, Literal(float(eps_est), datatype=XSD.float)))
        if eps_act is not None and pd.notna(eps_act):
            g.add((ev, EX.epsActual, Literal(float(eps_act), datatype=XSD.float)))
        if surprise is not None and pd.notna(surprise):
            g.add((ev, EX.surprisePct, Literal(float(surprise), datatype=XSD.float)))

    emitted = 0
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.reset_index()
        # Normalize column names
        ren = {}
        for col in df.columns:
            lc = str(col).lower()
            if lc in ("index", "earnings date", "date"): ren[col] = "Date"
            elif "estimate" in lc:                       ren[col] = "EPS Estimate"
            elif "reported" in lc:                       ren[col] = "Reported EPS"
            elif "surprise" in lc:                       ren[col] = "Surprise(%)"
        df = df.rename(columns=ren)
        for _, r in df.iterrows():
            _emit_row(r.get("Date"), r.get("EPS Estimate"), r.get("Reported EPS"), r.get("Surprise(%)"))
            emitted += 1

    # 2) Fallback: next earnings date from calendar
    if emitted == 0:
        try:
            cal = y.calendar
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                val = None
                if "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].iloc[0]
                elif "Earnings Date" in cal.columns:
                    val = cal["Earnings Date"].iloc[0]
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    val = val[0]
                if val is not None:
                    _emit_row(val)
        except Exception:
            pass

# --- Helpers ---------------------------------------------------------------

def _fi_get(fi, key):
    # fast_info can be a dict-like or an object with attributes
    try:
        if hasattr(fi, key):
            v = getattr(fi, key)
            return float(v) if v is not None else None
    except Exception:
        pass
    try:
        v = fi.get(key)  # dict style
        return float(v) if v is not None else None
    except Exception:
        return None

def add_fundamental_snapshot(g: Graph, y: yf.Ticker, tk: str, asset: URIRef, today: dt.date):
    """Market cap, beta, dividend yield + sector/industry."""
    snap = URIRef(EX[f"{slug(tk)}_fund_{today}"])
    g.add((snap, RDF.type, EX.FundamentalSnapshot))
    g.add((snap, EX.aboutAsset, asset))
    g.add((snap, EX.onDate, Literal(today, datatype=XSD.date)))

    # fast_info (robust to versions)
    try:
        fi = getattr(y, "fast_info", {}) or {}
        mcap = _fi_get(fi, "market_cap")
        beta = _fi_get(fi, "beta")
        dyld = _fi_get(fi, "dividend_yield")
        if mcap is not None: g.add((snap, EX.marketCap, Literal(mcap, datatype=XSD.double)))
        if beta is not None:  g.add((snap, EX.beta, Literal(beta, datatype=XSD.float)))
        if dyld is not None:  g.add((snap, EX.dividendYield, Literal(dyld, datatype=XSD.float)))
    except Exception:
        pass

    # sector / industry (get_info preferred over deprecated .info)
    sector, industry = None, None
    try:
        gi = y.get_info()
        sector = gi.get("sector")
        industry = gi.get("industry")
    except Exception:
        try:
            gi = getattr(y, "info", {}) or {}
            sector = sector or gi.get("sector")
            industry = industry or gi.get("industry")
        except Exception:
            pass

    if sector:   g.add((asset, EX.sector,   Literal(sector,   datatype=XSD.string)))
    if industry: g.add((asset, EX.industry, Literal(industry, datatype=XSD.string)))


def add_corporate_actions(g: Graph, y: yf.Ticker, tk: str, asset: URIRef):
    """Dividends + Stock Splits (yfinance versions vary: actions or individual series)."""
    # Unified emitter
    def emit_div(d, amt):
        ev = URIRef(EX[f"{slug(tk)}_div_{d}"])
        g.add((ev, RDF.type, EX.DividendEvent))
        g.add((ev, EX.aboutAsset, asset))
        g.add((ev, EX.onDate, Literal(d, datatype=XSD.date)))
        g.add((ev, EX.amount, Literal(float(amt), datatype=XSD.float)))

    def emit_split(d, ratio):
        ev = URIRef(EX[f"{slug(tk)}_split_{d}"])
        g.add((ev, RDF.type, EX.SplitEvent))
        g.add((ev, EX.aboutAsset, asset))
        g.add((ev, EX.onDate, Literal(d, datatype=XSD.date)))
        g.add((ev, EX.ratio, Literal(str(ratio), datatype=XSD.string)))

    # 1) Newer yfinance: .actions table
    try:
        acts = y.actions or pd.DataFrame()
        if isinstance(acts, pd.DataFrame) and not acts.empty:
            acts = acts.reset_index().rename(columns={"index": "Date"})
            for _, r in acts.iterrows():
                d = pd.to_datetime(r["Date"]).date()
                if "Dividends" in r and pd.notna(r["Dividends"]) and float(r["Dividends"]) != 0.0:
                    emit_div(d, r["Dividends"])
                if "Stock Splits" in r and pd.notna(r["Stock Splits"]) and float(r["Stock Splits"]) != 0.0:
                    emit_split(d, r["Stock Splits"])
            return
    except Exception:
        pass

    # 2) Fallback: separate series
    try:
        divs = y.dividends or pd.Series(dtype="float64")
        if not divs.empty:
            for d, amt in divs.items():
                d = pd.to_datetime(d).date()
                if float(amt) != 0.0:
                    emit_div(d, amt)
    except Exception:
        pass

    try:
        splits = y.splits or pd.Series(dtype="float64")
        if not splits.empty:
            for d, ratio in splits.items():
                d = pd.to_datetime(d).date()
                if float(ratio) != 0.0:
                    emit_split(d, ratio)
    except Exception:
        pass

def add_analyst_consensus(g: Graph, y: yf.Ticker, tk: str, asset: URIRef, today: dt.date, window_days: int):
    """
    Prefer yfinance.get_recommendations_summary() which returns rows like:
      period  strongBuy  buy  hold  sell  strongSell   (periods: '0m','-1m',...)
    Aggregate the last N months that cover `window_days` (ceil(window_days/30)).
    Fallback to historical recommendations mapping if summary not present.
    """
    # --- try the monthly summary first ---
    df = None
    try:
        if hasattr(y, "get_recommendations_summary"):
            df = y.get_recommendations_summary(as_dict=False)
        elif hasattr(y, "recommendations_summary"):
            # some versions expose as a property or method
            rs = y.recommendations_summary
            df = rs() if callable(rs) else rs
    except Exception:
        df = None

    def emit_summary_counts(scounts: dict):
        node = URIRef(EX[f"{slug(tk)}_cons_{today}"])
        g.add((node, RDF.type, EX.AnalystConsensus))
        g.add((node, EX.aboutAsset, asset))
        g.add((node, EX.onDate, Literal(today, datatype=XSD.date)))
        g.add((node, EX.windowDays, Literal(int(window_days), datatype=XSD.integer)))
        # expose both coarse and fine buckets
        for k in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
            g.add((node, URIRef(EX[k]), Literal(int(scounts.get(k, 0)), datatype=XSD.integer)))

    if isinstance(df, pd.DataFrame) and not df.empty:
        # normalize cols
        cols = {c: str(c).strip() for c in df.columns}
        df = df.rename(columns=cols)
        if "period" in df.columns:
            # how many months to include?
            months = max(1, int(math.ceil(window_days / 30.0)))
            # keep 0m, -1m, ... up to -(months-1)m
            def period_to_int(s: str) -> int:
                s = str(s).strip()
                if s == "0m": return 0
                m = re.match(r"-(\d+)m$", s)
                return int(m.group(1)) if m else 9999
            df["_m"] = df["period"].map(period_to_int)
            df = df[df["_m"] <= (months - 1)]
        # sum across selected months (or all rows if no period col)
        agg = {k: int(df[k].sum()) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"] if k in df.columns}
        if agg:
            emit_summary_counts(agg)
            return  # success

    # --- fallback: historical recommendations -> bucketize to buy/hold/sell ---
    try:
        rec = y.recommendations
    except Exception:
        rec = None

    if isinstance(rec, pd.DataFrame) and not rec.empty:
        since = pd.Timestamp(today - dt.timedelta(days=window_days))
        # normalize index/column for date
        if rec.index.name is None and "Date" in rec.columns:
            rec["Date"] = pd.to_datetime(rec["Date"]); rec = rec.set_index("Date")
        rec = rec[rec.index >= since]
        if not rec.empty:
            # pick a target-grade-like column
            possible_cols = ["To Grade", "toGrade", "to grade", "to_grade", "Grade"]
            col = next((c for c in possible_cols if c in rec.columns), None)
            if col:
                def bucket(x: str):
                    x = (str(x) or "").lower()
                    if any(k in x for k in ["strong buy", "strongbuy"]):           return "strongBuy"
                    if any(k in x for k in ["buy", "outperform", "overweight", "add", "accumulate"]): return "buy"
                    if any(k in x for k in ["sell", "underperform", "underweight", "reduce"]):        return "sell"
                    return "hold"
                counts = (
                    rec.reset_index()
                       .assign(bucket=lambda d: d[col].map(bucket))
                       .groupby("bucket")
                       .size()
                       .to_dict()
                )
                # synthesize strongSell as 0 (we cannot infer it cleanly from grades)
                agg = {
                    "strongBuy": int(counts.get("strongBuy", 0)),
                    "buy":       int(counts.get("buy", 0)),
                    "hold":      int(counts.get("hold", 0)),
                    "sell":      int(counts.get("sell", 0)),
                    "strongSell": 0,
                }
                emit_summary_counts(agg)


def main():
    cfg = load_cfg()
    out = Path(cfg.get("fundamentals", {}).get("out", "data/kg_fundamentals.ttl"))
    window = int(cfg.get("fundamentals", {}).get("consensus_window_days", 180))
    tickers = [str(t).upper() for t in (cfg.get("stocks") or [])]

    g = Graph(); g.bind("ex", EX)
    # Klassen registrieren (optional)
    for cls in [EX.FundamentalSnapshot, EX.DividendEvent, EX.SplitEvent, EX.EarningsEvent, EX.AnalystConsensus]:
        g.add((cls, RDF.type, EX.Class))

    today = dt.date.today()

    for tk in tickers:
        y = yf.Ticker(tk)
        asset = URIRef(EX[slug(tk)])

        # Helpers (robust, version-tolerant)
        add_fundamental_snapshot(g, y, tk, asset, today)
        add_corporate_actions(g, y, tk, asset)
        add_earnings_events(g, y, tk, asset)              # your existing robust helper
        add_analyst_consensus(g, y, tk, asset, today, window)

    out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out), format="turtle")
    print(f"✅ Fundamentals/Events gespeichert → {out}")


if __name__ == "__main__":
    main()
