from pathlib import Path
import re, yaml
import pandas as pd
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import XSD, RDF

EX = Namespace("http://example.org/")

def slug(s:str)->str: return re.sub(r"[^A-Za-z0-9]+","_",str(s)).strip("_")

def load_cfg():
    return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}

def query_prices_df(g: Graph, tk: str) -> pd.DataFrame:
    q = f"""
    PREFIX ex:<http://example.org/>
    SELECT ?d ?c WHERE {{
      ?p a ex:Price ;
         ex:ofAsset ex:{tk} ;
         ex:onDate ?d ;
         ex:closePrice ?c .
    }} ORDER BY ?d
    """
    rows = [(r.d.toPython(), float(r.c)) for r in g.query(q)]
    return pd.DataFrame(rows, columns=["Date","AdjClose"]).assign(Date=lambda x: pd.to_datetime(x["Date"]))

def rsi(series: pd.Series, period=14):
    delta = series.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down.replace(0, pd.NA)
    return 100 - (100/(1+rs))

def main():
    cfg = load_cfg()
    kg_prices = Path(cfg.get("graph",{}).get("in") or cfg.get("graph",{}).get("out") or "data/kg_assets.ttl")
    out = Path(cfg.get("features",{}).get("indicators_out","data/kg_indicators.ttl"))
    ma_wins = cfg.get("features",{}).get("ma_windows",[20,50,200])
    rsi_period = int(cfg.get("features",{}).get("rsi_period",14))
    vol_win = int(cfg.get("features",{}).get("vol_window",20))

    g = Graph(); g.parse(str(kg_prices), format="turtle")

    # Ticker entdecken
    tq = "PREFIX ex:<http://example.org/> SELECT DISTINCT ?s WHERE { ?p a ex:Price ; ex:ofAsset ?s . }"
    tickers = sorted({ str(r[0]).split("/")[-1] for r in g.query(tq) })

    g_out = Graph(); g_out.bind("ex", EX)
    g_out.add((EX.DailyIndicator, RDF.type, EX.Class))

    for tk in tickers:
        df = query_prices_df(g, tk)
        if df.empty: continue
        df = df.set_index("Date").sort_index()
        ret = df["AdjClose"].pct_change()
        ma = {w: df["AdjClose"].rolling(w).mean() for w in ma_wins}
        vol = ret.rolling(vol_win).std()
        r = rsi(df["AdjClose"], rsi_period)

        asset = URIRef(EX[tk])
        for dt, row in df.iterrows():
            di = URIRef(EX[f"{tk}_ind_{dt.date()}"])
            g_out.add((di, RDF.type, EX.DailyIndicator))
            g_out.add((di, EX.aboutAsset, asset))
            g_out.add((di, EX.onDate, Literal(dt.date(), datatype=XSD.date)))
            # values (nur wenn vorhanden)
            v = ret.get(dt);        
            if pd.notna(v):  g_out.add((di, EX.return1d, Literal(float(v), datatype=XSD.float)))
            for w,s in ma.items():
                val = s.get(dt)
                if pd.notna(val): g_out.add((di, URIRef(EX[f"ma{w}"]), Literal(float(val), datatype=XSD.float)))
            vv = vol.get(dt);       
            if pd.notna(vv): g_out.add((di, EX.vol20, Literal(float(vv), datatype=XSD.float)))
            rr = r.get(dt);         
            if pd.notna(rr): g_out.add((di, EX.rsi14, Literal(float(rr), datatype=XSD.float)))

    out.parent.mkdir(parents=True, exist_ok=True)
    g_out.serialize(destination=str(out), format="turtle")
    print(f"✅ Indicators gespeichert → {out}")

if __name__ == "__main__":
    main()
