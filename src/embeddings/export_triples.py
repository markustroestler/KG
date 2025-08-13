# embeddings/export_triples.py (erweitert)
from pathlib import Path
from datetime import date, timedelta
import yaml, math
from rdflib import Graph, Namespace, URIRef, RDF, Literal
from rdflib.namespace import XSD
from collections import Counter

EX = Namespace("http://example.org/")

def _cfg():
    return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}

def _d_ent(d: date) -> URIRef:
    return URIRef(EX[f"d_{d.isoformat()}"])

def _short(u: URIRef | str) -> str:
    base = str(EX); s = str(u)
    return "ex:" + s[len(base):] if s.startswith(base) else s

def _emit_feature_edges(triples, A_short: str, D_short: str, f: dict):
    def add(rel: str): triples.add((A_short, rel, D_short))
    add(f"ex:hasTrend5d_{f['trend5d'].replace('trend_','')}")
    add(f"ex:hasNewsPos7d_{f['npos'].replace('npos_','')}")
    add(f"ex:hasNewsNeg7d_{f['nneg'].replace('nneg_','')}")
    add(f"ex:hasRSI_{f['rsi'].replace('rsi_','')}")
    add(f"ex:hasVol_{f['vol'].replace('vol_','')}")
    # add(f"ex:hasConsensus_{f['cons'].replace('cons_','')}")


# ---------- Bucketing ----------
def bucket(val: float | None, cuts: list[float], labels: list[str]) -> str:
    if val is None or math.isnan(val): return labels[0]
    for c, lab in zip(cuts, labels):
        if val <= c: return lab
    return labels[-1]

# ---------- Queries ----------
def q_prices_for_asset(g: Graph, tk: str):
    q = f"""PREFIX ex:<http://example.org/>
    SELECT ?d ?c WHERE {{
      ?p a ex:Price ; ex:ofAsset ex:{tk} ; ex:onDate ?d ; ex:closePrice ?c .
    }} ORDER BY ?d"""
    out = []
    for r in g.query(q):
        d = r.d.toPython(); d = d.date() if hasattr(d, "date") else d
        out.append((d, float(r.c)))
    return out

def q_news_counts(g: Graph, tk: str, d: date, win: int):
    q = f"""PREFIX ex:<http://example.org/>
    SELECT ?sent (COUNT(*) AS ?n) WHERE {{
      ?n a ex:News ; ex:aboutAsset ex:{tk} ; ex:publishedDate ?pd ; ex:sentiment ?sent .
      FILTER (?pd >= "{(d - timedelta(days=win-1)).isoformat()}"^^xsd:date && ?pd <= "{d.isoformat()}"^^xsd:date)
    }} GROUP BY ?sent"""
    pos = neg = 0
    for r in g.query(q, initNs={"xsd": XSD}):
        s = str(r.sent).split("/")[-1].lower()  # Positive/Negative/Neutral
        n = int(r.n.toPython())
        if "positive" in s: pos += n
        elif "negative" in s: neg += n
    return pos, neg

def q_indicators_on(g: Graph, tk: str, d: date):
    q = f"""PREFIX ex:<http://example.org/>
    SELECT ?rsi ?vol WHERE {{
      ?di a ex:DailyIndicator ; ex:aboutAsset ex:{tk} ; ex:onDate "{d.isoformat()}"^^xsd:date .
      OPTIONAL {{ ?di ex:rsi14 ?rsi. }} OPTIONAL {{ ?di ex:vol20 ?vol. }}
    }}"""
    rsi = vol = None
    for r in g.query(q, initNs={"xsd": XSD}):
        rsi = float(r.rsi) if r.rsi else None
        vol = float(r.vol) if r.vol else None
    return rsi, vol

def q_consensus_latest_on_or_before(g: Graph, tk: str, d: date):
    q = f"""PREFIX ex:<http://example.org/>
    SELECT ?sb ?b ?h ?s ?ss ?on WHERE {{
      ?c a ex:AnalystConsensus ; ex:aboutAsset ex:{tk} ; ex:onDate ?on .
      OPTIONAL {{ ?c ex:strongBuy ?sb }} OPTIONAL {{ ?c ex:buy ?b }}
      OPTIONAL {{ ?c ex:hold ?h }}       OPTIONAL {{ ?c ex:sell ?s }}
      OPTIONAL {{ ?c ex:strongSell ?ss }}
      FILTER (?on <= "{d.isoformat()}"^^xsd:date)
    }} ORDER BY DESC(?on) LIMIT 1"""
    for r in g.query(q, initNs={"xsd": XSD}):
        vals = {k: int((getattr(r, k) or 0)) for k in ["sb","b","h","s","ss"]}
        tot = sum(vals.values()) or 1
        score = (2*vals["sb"] + 1*vals["b"] - 1*vals["s"] - 2*vals["ss"]) / tot  # grob: Buy(+)…Sell(−)
        lab = "cons_buy" if score > 0.2 else "cons_sell" if score < -0.2 else "cons_hold"
        return lab
    return "cons_hold"

# ---------- Features + Labels ----------
def features_for(g_assets: Graph, g_news: Graph, g_ind: Graph, g_fund: Graph, tk: str, d: date, closes_by_day: dict):
    # Trend 5d (über Close)
    def trend5d():
        back = [closes_by_day.get(d - timedelta(days=i)) for i in range(0,6)]
        back = [x for x in back if x is not None]
        if len(back) < 2: return "trend_flat"
        if back[-1] > back[0]: return "trend_up"
        if back[-1] < back[0]: return "trend_down"
        return "trend_flat"

    # News counts (7d)
    pos7, neg7 = q_news_counts(g_news, tk, d, 7)
    npos = bucket(pos7, [0,2,5], ["npos_0","npos_1_2","npos_3_5","npos_6p"])
    nneg = bucket(neg7, [0,2,5], ["nneg_0","nneg_1_2","nneg_3_5","nneg_6p"])

    # RSI/Vol Buckets
    rsi, vol = q_indicators_on(g_ind, tk, d)
    rsi_b = ("rsi_low" if (rsi or 0) < 30 else "rsi_high" if (rsi or 0) > 70 else "rsi_mid")
    vol_b = bucket(vol if vol is not None else float("nan"), [0.01,0.02,0.04], ["vol_low","vol_mid","vol_high","vol_very_high"])

    # Analystenkonsens
    # cons_b = q_consensus_latest_on_or_before(g_fund, tk, d)

    return {
        "trend5d": trend5d(),
        "npos": npos, "nneg": nneg,
        "rsi": rsi_b, "vol": vol_b,
        # "cons": cons_b,
    }

def forward_label(closes_by_day: dict, d: date, H=14, tau=0.02):
    c0 = closes_by_day.get(d); cH = closes_by_day.get(d + timedelta(days=H))
    if c0 is None or cH is None: return None
    ret = (cH / c0) - 1.0
    if ret >= tau:  return "up"
    if ret <= -tau: return "down"
    return None

def print_relation_counts(name, triples):
    rel_counts = Counter(r for _, r, _ in triples)
    print(f"\n--- {name} Relation Counts ---")
    for rel, cnt in rel_counts.most_common():
        print(f"{rel:35} {cnt}")
    print(f"Total triples in {name}: {len(triples)}")

def read_tsv(p):
    return [tuple(line.strip().split("\t")) for line in Path(p).read_text().splitlines() if line.strip()]


def main():
    cfg = _cfg()
    e = cfg.get("embeddings", {}) or {}
    out_dir = Path(e.get("out_dir", "data/kge"))
    H = int(e.get("horizon_days", 14))
    tau = float(e.get("target_threshold", 0.02))
    window_days = int(cfg.get("embeddings", {}).get("window_days", 180))

    # KGs laden
    kg_assets = Path(cfg.get("graph", {}).get("in") or cfg.get("graph", {}).get("out") or "data/kg_assets.ttl")
    kg_news   = Path(cfg.get("news",  {}).get("kg_out", "data/kg_news.ttl"))
    kg_ind    = Path(cfg.get("features",{}).get("indicators_out", "data/kg_indicators.ttl"))
    kg_fund   = Path(cfg.get("fundamentals",{}).get("out", "data/kg_fundamentals.ttl"))

    gA = Graph(); gA.parse(kg_assets, format="turtle")
    gN = Graph(); 
    if kg_news.exists(): gN.parse(kg_news, format="turtle")
    gI = Graph(); 
    if kg_ind.exists():  gI.parse(kg_ind, format="turtle")
    gF = Graph(); 
    if kg_fund.exists(): gF.parse(kg_fund, format="turtle")

    # Ticker entdecken
    tq = "PREFIX ex:<http://example.org/> SELECT DISTINCT ?s WHERE { ?p a ex:Price ; ex:ofAsset ?s . }"
    tickers = sorted({ str(r[0]).split("/")[-1] for r in gA.query(tq) })
    print(tickers)

    triples = set()

    # Iterate über (A,D)
    for tk in tickers:
        prices = q_prices_for_asset(gA, tk)  # Liste (d, close)
        if not prices: continue
        closes = {d:c for d,c in prices}

        # Beobachtungszeitraum (letzte window_days abdeckend)
        days = [d for d,_ in prices]
        if not days: continue
        start = max(days[0], (date.today() - timedelta(days=window_days)))
        obs_days = [d for d in days if start <= d <= date.today()]

        for d in obs_days:
            # Label
            y = forward_label(closes, d, H=H, tau=tau)
            if not y: 
                continue  # optional: nur klare up/down Beispiele

            A = URIRef(EX[tk]); D = _d_ent(d)
            triples.add((_short(A), "ex:instanceOf", "ex:Asset"))
            triples.add((_short(D), "ex:instanceOf", "ex:Day"))

            # Zielrelation
            rel = "ex:risesWithin_%dd" % H if y == "up" else "ex:fallsWithin_%dd" % H
            triples.add((_short(A), rel, _short(D)))

            # Features
            f = features_for(gA, gN, gI, gF, tk, d, closes)
            _emit_feature_edges(triples, _short(A), _short(D), f)

   # Tag-Entitäten komplettieren für Sichtbarkeit
    for h, r, t in list(triples):
        if t.startswith("ex:d_"):
            triples.add((t, "ex:instanceOf", "ex:Day"))

    # Splits (zeitbasiert)
    all_days = sorted({t for _,_,t in triples if t.startswith("ex:d_")})
    if not all_days:
        print("⚠️ Keine Triples erzeugt – prüfe Daten/Schwellen.")
        return
    n = len(all_days)
    cut1, cut2 = int(n*0.7), int(n*0.85)
    train_days = set(all_days[:cut1]); valid_days = set(all_days[cut1:cut2]); test_days = set(all_days[cut2:])

    out_dir.mkdir(parents=True, exist_ok=True)

    day_typings = {(t, "ex:instanceOf", "ex:Day")
               for _,_,t in triples if t.startswith("ex:d_")}

    today = date.today()
    day_typings.add((f"ex:d_{today.isoformat()}", "ex:instanceOf", "ex:Day"))
    asset_typings = {(f"ex:{tk}", "ex:instanceOf", "ex:Asset") for tk in tickers}

    # 3) Beim Schreiben: alle Day/Asset-Typings NUR in train.tsv mitschreiben
    def write(fname, days_set, include_typings=False):
        p = out_dir / fname
        with p.open("w", encoding="utf-8") as f:
            if include_typings and fname == "train.tsv":
                for h,r,t in sorted(day_typings):
                    f.write(f"{h}\t{r}\t{t}\n")
                for h,r,t in sorted(asset_typings):
                    f.write(f"{h}\t{r}\t{t}\n")
            for h,r,t in sorted(triples):
                if t in days_set:
                    f.write(f"{h}\t{r}\t{t}\n")

    p_train = write("train.tsv", train_days, include_typings=True)
    p_valid = write("valid.tsv", valid_days)
    p_test  = write("test.tsv",  test_days)

    # in TXT for debug
    p_trainTXT = write("train.txt", train_days, include_typings=True)
    p_validTXT = write("valid.txt", valid_days)
    p_testTXT  = write("test.txt",  test_days)

    print_relation_counts("ALL", triples)

    train_triples = read_tsv(out_dir/"train.tsv")
    valid_triples = read_tsv(out_dir/"valid.tsv")
    test_triples  = read_tsv(out_dir/"test.tsv")
    # Für Splits: train_days / valid_days / test_days sind Sets mit Days,
    # aber du musst die Triples filtern:
    # train_triples = [t for t in triples if t[2] in train_days or t[0] in train_days]
    # valid_triples = [t for t in triples if t[2] in valid_days or t[0] in valid_days]
    # test_triples  = [t for t in triples if t[2] in test_days  or t[0] in test_days]

    # print_relation_counts("TRAIN", train_triples)
    # print_relation_counts("VALID", valid_triples)
    # print_relation_counts("TEST",  test_triples)

if __name__ == "__main__":
    main()
