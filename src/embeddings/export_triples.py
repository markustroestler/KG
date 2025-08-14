# export_triples.py – kompakt & MRR-orientiert
from pathlib import Path
from datetime import date, timedelta
import yaml, math, random, re
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import XSD

random.seed(42)
EX = Namespace("http://example.org/")

# -------------------- Config / Utils --------------------
def _cfg():
    try:
        return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}

def _short(u: URIRef | str) -> str:
    s = str(u); base = str(EX)
    return "ex:" + s[len(base):] if s.startswith(base) else s

# NEW: symbol → ex:LABEL (ohne ^ . -)
def _sym_to_ex(s: str) -> str:
    s = str(s).upper()
    s = s.replace("^", "")
    s = re.sub(r"[^A-Z0-9]+", "_", s).strip("_")
    return s  # ohne "ex:"; für SPARQL nutzen wir ex:{...}

def _trading_days(closes: dict):
    return sorted(closes.keys())

def ret_trailing_td(closes: dict, d, k: int):
    """Trailing Return über k Handelstage bis zum letzten verfügbaren Tag ≤ d."""
    days = sorted(closes.keys())
    # auf letzten bekannten Tag ≤ d snappen
    prior = [x for x in days if x <= d]
    if not prior:
        return None
    d_eff = prior[-1]
    i = days.index(d_eff)
    j = i - k
    if j < 0:
        return None
    c0, c1 = closes[days[j]], closes[days[i]]
    if c0 is None or c1 is None or c0 == 0:
        return None
    return (c1 / c0) - 1.0

def bucket_rel_idx(diff: float | None):
    if diff is None: return "na"
    if diff <= -0.02: return "under"
    if diff >=  0.02: return "out"
    return "in"

def prev_trading_day(closes: dict, d, max_back=5):
    days = _trading_days(closes)
    if d not in closes: return None
    i = days.index(d)
    if i == 0: return None
    return days[i-1]

def bucket_count(n: int):
    if n <= 0: return "0"
    if n <= 2: return "1_2"
    if n <= 5: return "3_5"
    return "6p"

def dist_to_high_52w_bucket(closes: dict, d):
    days = _trading_days(closes)
    if d not in closes: return "na"
    i = days.index(d)
    # 252 Handelstage Rückblick (≈ 52 Wochen)
    j = max(0, i - 251)
    window = [closes[x] for x in days[j:i+1] if closes.get(x) is not None]
    if not window: return "na"
    c = closes[d]; hi = max(window)
    if not c or not hi or hi == 0: return "na"
    dist = (hi - c) / hi
    if dist < 0.05:  return "near"
    if dist < 0.15:  return "mid"
    return "far"

def bucket_cont(x: float | None, cuts: list[float], labels: list[str]) -> str:
    if x is None or math.isnan(x): return labels[0]
    for c, lab in zip(cuts, labels):
        if x <= c: return lab
    return labels[-1]

# -------------------- Queries --------------------
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

def q_indicators_on(g: Graph | None, tk: str, d: date):
    if g is None: return None, None
    q = f"""PREFIX ex:<http://example.org/>
    SELECT ?rsi ?vol WHERE {{
      ?di a ex:DailyIndicator ; ex:aboutAsset ex:{tk} ;
          ex:onDate "{d.isoformat()}"^^xsd:date .
      OPTIONAL {{ ?di ex:rsi14 ?rsi. }} OPTIONAL {{ ?di ex:vol20 ?vol. }}
    }}"""
    rsi = vol = None
    for r in g.query(q, initNs={"xsd": XSD}):
        rsi = float(r.rsi) if r.rsi else None
        vol = float(r.vol) if r.vol else None
    return rsi, vol

def q_first_news_date(g: Graph | None, tk: str):
    if g is None: return None
    q = f"""PREFIX ex:<http://example.org/> PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>
    SELECT (MIN(?d) AS ?minD) WHERE {{
      {{ ?news a ex:News ; ex:aboutAsset ex:{tk} . }}
      UNION {{ ex:{tk} ex:hasNews ?news . }}
      OPTIONAL {{ ?news ex:publishedDate ?pd_d . }}
      OPTIONAL {{ ?news ex:publishedAt  ?pd_dt . BIND(xsd:date(?pd_dt) AS ?pd_dt_d) }}
      BIND(COALESCE(?pd_d, ?pd_dt_d) AS ?d)
      FILTER(bound(?d))
    }}"""
    for r in g.query(q, initNs={"xsd": XSD}):
        if r.minD:
            v = r.minD.toPython()
            return v.date() if hasattr(v, "date") else v
    return None

def q_news_stats(g: Graph | None, tk: str, d: date, win: int = 7):
    # zählt pos/neg/neutral im Fenster [d-(win-1), d]
    if g is None:
        return {"pos": 0, "neg": 0, "neu": 0, "tot": 0}
    q = f"""PREFIX ex:<http://example.org/> PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>
    SELECT ?sent (COUNT(*) AS ?n) WHERE {{
      {{ ?news a ex:News ; ex:aboutAsset ex:{tk} . }}
      UNION {{ ex:{tk} ex:hasNews ?news . }}
      OPTIONAL {{ ?news ex:publishedDate ?pd_d . }}
      OPTIONAL {{ ?news ex:publishedAt  ?pd_dt . BIND(xsd:date(?pd_dt) AS ?pd_dt_d) }}
      BIND(COALESCE(?pd_d, ?pd_dt_d) AS ?pdate)
      FILTER (?pdate >= "{(d - timedelta(days=win-1)).isoformat()}"^^xsd:date &&
              ?pdate <= "{d.isoformat()}"^^xsd:date)
      OPTIONAL {{ ?news ex:sentiment ?sent . }}
    }} GROUP BY ?sent"""
    pos = neg = neu = tot = 0
    for r in g.query(q, initNs={"xsd": XSD}):
        c = int(r.n.toPython()); tot += c
        s = (str(r.sent).lower() if r.sent else "")
        if "positive" in s: pos += c
        elif "negative" in s: neg += c
        elif "neutral"  in s: neu += c
    return {"pos": pos, "neg": neg, "neu": neu, "tot": tot}

# -------------------- Features & Label --------------------
def news_coverage_flag(first_news_date: date | None, d: date, win: int) -> str:
    if not first_news_date:
        return "none"
    return "full" if (d - timedelta(days=win-1)) >= first_news_date else "partial"

def features_for(gN: Graph | None, gI: Graph | None,
                 tk: str, d: date, closes_by_day: dict, first_news_date: date | None, win: int):
    # Trend 5d (auf Close)
    back = [closes_by_day.get(d - timedelta(days=i)) for i in range(0, 6)]
    back = [x for x in back if x is not None]
    if len(back) < 2:
        trend = "flat"
    else:
        trend = "up" if back[-1] > back[0] else "down" if back[-1] < back[0] else "flat"

    # News: Pos/Neg als Buckets + Coverage
    ns = q_news_stats(gN, tk, d, win=win)
    news_pos_b = bucket_count(ns["pos"])   # 0 | 1_2 | 3_5 | 6p
    news_neg_b = bucket_count(ns["neg"])
    news_cov   = news_coverage_flag(first_news_date, d, win)

    # RSI / Vol (robust gebucketed)
    rsi, vol = q_indicators_on(gI, tk, d)
    rsi_b = "low" if (rsi or 0) < 30 else "high" if (rsi or 0) > 70 else "mid"
    vol_b = bucket_cont(vol if vol is not None else float("nan"),
                        [0.01, 0.02, 0.04], ["low", "mid", "high", "very_high"])

    return {
        "trend5d": trend,
        "news_pos": news_pos_b,
        "news_neg": news_neg_b,
        "news_cov": news_cov,
        "rsi": rsi_b,
        "vol": vol_b,
    }

def forward_label(closes_by_day: dict, d: date, H=14, tau=0.01):
    c0 = closes_by_day.get(d); cH = closes_by_day.get(d + timedelta(days=H))
    if c0 is None or cH is None: return None
    ret = (cH / c0) - 1.0
    if ret >= tau:  return "up"
    if ret <= -tau: return "down"
    return None

# -------------------- Build + Write --------------------
def main():
    cfg = _cfg()
    e = cfg.get("embeddings", {}) or {}
    out_dir = Path(e.get("out_dir", "data/kge"))
    H = int(e.get("horizon_days", 14))
    tau = float(e.get("target_threshold", 0.01))
    window_days = int(e.get("window_days", 365 * 3))
    win_news = int(e.get("news_window_days", 7))
    max_class_ratio = e.get("max_class_ratio", 1.0)  # 1.0 = strikt balanciert; >1 erlaubt leichte Mehrheit

    # KGs laden
    kg_assets = Path(cfg.get("graph", {}).get("in") or cfg.get("graph", {}).get("out") or "data/kg_assets.ttl")
    kg_news   = Path(cfg.get("news",  {}).get("kg_out", "data/kg_news.ttl"))
    kg_ind    = Path(cfg.get("features",{}).get("indicators_out", "data/kg_indicators.ttl"))

    gA = Graph(); gA.parse(kg_assets, format="turtle")
    gN = Graph() if kg_news.exists() else None
    if gN is not None: gN.parse(kg_news, format="turtle")
    gI = Graph() if kg_ind.exists() else None
    if gI is not None: gI.parse(kg_ind, format="turtle")

    # ---- Ticker aus config (falls vorhanden), sonst aus KG ----
    cfg_stocks = [ _sym_to_ex(s) for s in (cfg.get("stocks") or []) ]  # ohne "ex:"
    # Alle mit Preisen im KG
    tq = "PREFIX ex:<http://example.org/> SELECT DISTINCT ?s WHERE { ?p a ex:Price ; ex:ofAsset ?s . }"
    present = { str(r[0]).split("/")[-1] for r in gA.query(tq) }
    tickers = sorted([t for t in cfg_stocks if t in present]) if cfg_stocks else sorted(present)

    # ---- Index-Closes laden (aus config.indices + aus index_map)  # NEW ----
    cfg_indices = [ _sym_to_ex(s) for s in (cfg.get("indices") or []) ]
    idx_map_cfg = ((cfg.get("features") or {}).get("index_map") or {})
    idx_of = { _sym_to_ex(k): _sym_to_ex(v) for k, v in idx_map_cfg.items() if v }  # Asset(ex) -> Index(ex)
    # sicherstellen, dass alle gemappten Indizes geladen werden
    need_indices = sorted(set(cfg_indices) | set(idx_of.values()))
    index_closes = {}
    for idx in need_indices:
        ps = q_prices_for_asset(gA, idx)
        if ps:
            index_closes[idx] = {d: c for d, c in ps}

    # ---- Peers vorbereiten: einfach "alle anderen Stocks"  # NEW ----
    peers_of = { tk: [p for p in tickers if p != tk] for tk in tickers }

    # ---- Alle Asset-Closes (für peersUp1d)  # NEW ----
    asset_closes = {}
    for tk in tickers:
        ps = q_prices_for_asset(gA, tk)
        if ps:
            asset_closes[tk] = {d: c for d, c in ps}

    # Frühestes News-Datum je Ticker (für Coverage)
    first_news_by_tk = {}
    if gN is not None:
        for tk in tickers:
            first_news_by_tk[tk] = q_first_news_date(gN, tk)

    # Beispiele sammeln pro Ticker
    examples_by_tk: dict[str, list[tuple[date, str, dict]]] = {tk: [] for tk in tickers}
    for tk in tickers:
        prices = q_prices_for_asset(gA, tk)
        if not prices: continue
        closes = {d: c for d, c in prices}
        days = [d for d, _ in prices]
        if not days: continue

        start = max(days[0], (date.today() - timedelta(days=window_days)))
        obs_days = [d for d in days if start <= d <= date.today()]

        for d in obs_days:
            y = forward_label(closes, d, H=H, tau=tau)
            if y is None:
                continue  # nur klare up/down Beispiele

            # Basis-Features
            feats = features_for(gN, gI, tk, d, closes, first_news_by_tk.get(tk), win=win_news)

            # --- NEW: relToIndex20d ---
            rel_idx = "na"
            idx_sym = idx_of.get(tk)
            if idx_sym and idx_sym in index_closes:
                r_a20 = ret_trailing_td(closes, d, int(e.get("rel_to_index_window_days", 20)))
                r_i20 = ret_trailing_td(index_closes[idx_sym], d, int(e.get("rel_to_index_window_days", 20)))
                rel_idx = bucket_rel_idx(None if (r_a20 is None or r_i20 is None) else (r_a20 - r_i20))

            # --- NEW: peersUp1d (am Vortag) ---
            up_bucket = "0"
            prev_d = prev_trading_day(closes, d)
            if prev_d:
                up_cnt = 0
                for p in peers_of.get(tk, []):
                    pc = asset_closes.get(p)
                    if not pc: continue
                    prev_p = prev_trading_day(pc, prev_d)
                    if prev_p and pc.get(prev_p) and pc.get(prev_d):
                        r = pc[prev_d]/pc[prev_p] - 1.0
                        if r > 0: up_cnt += 1
                up_bucket = bucket_count(up_cnt)

            # --- NEW: distToHigh52w ---
            dist52 = dist_to_high_52w_bucket(closes, d)

            feats.update({
                "relToIndex20d": rel_idx,
                "peersUp1d": up_bucket,
                "distToHigh52w": dist52,
            })

            examples_by_tk[tk].append((d, y, feats))

    # Balancing (Downsampling Majority pro Ticker)
    triples = set()
    def add(h, r, t): triples.add((h, r, t))
    for tk, rows in examples_by_tk.items():
        if not rows: continue
        ups   = [r for r in rows if r[1] == "up"]
        downs = [r for r in rows if r[1] == "down"]
        if not ups or not downs or max_class_ratio is None:
            picked = rows
        else:
            maj, minr = (ups, downs) if len(ups) > len(downs) else (downs, ups)
            k = min(len(maj), int(math.ceil(len(minr) * float(max_class_ratio))))
            maj = random.sample(maj, k) if len(maj) > k else maj
            picked = minr + maj

        A = _short(URIRef(EX[tk]))
        # Emit (genau eine Kante je Feature-Art)
        for d, y, f in picked:
            D = f"ex:d_{d.isoformat()}"
            add(A, "ex:instanceOf", "ex:Asset")  # Typisierung (nur in train schreiben)
            add(D, "ex:instanceOf", "ex:Day")

            rel = f"ex:{'risesWithin' if y=='up' else 'fallsWithin'}_{H}d"
            add(A, rel, D)

            add(A, f"ex:hasTrend5d_{f['trend5d']}", D)
            add(A, f"ex:hasRSI_{f['rsi']}", D)
            add(A, f"ex:hasVol_{f['vol']}", D)
            add(A, f"ex:hasNews7d_pos_{f['news_pos']}", D)   # Stärke positiv
            add(A, f"ex:hasNews7d_neg_{f['news_neg']}", D)   # Stärke negativ
            add(A, f"ex:hasNewsCoverage7d_{f['news_cov']}", D)  # full/partial/none

            # --- NEW: drei neue Feature-Kanten ---
            add(A, f"ex:relToIndex20d_{f['relToIndex20d']}", D)      # under|in|out|na
            add(A, f"ex:peersUp1d_{f['peersUp1d']}", D)              # 0|1_2|3_5|6p
            add(A, f"ex:distToHigh52w_{f['distToHigh52w']}", D)      # near|mid|far|na

    if not triples:
        print("⚠️ Keine Triples erzeugt – prüfe Daten/Schwellen.")
        return

    # Typisierungen extrahieren (nur in train.tsv)
    day_typings   = {(t, "ex:instanceOf", "ex:Day")   for _, _, t in triples if str(t).startswith("ex:d_")}
    asset_typings = {(h, "ex:instanceOf", "ex:Asset") for h, _, _ in triples if not str(h).startswith("ex:d_")}

    # Zeit-Splits
    all_days = sorted({t for _, _, t in triples if str(t).startswith("ex:d_")})
    n = len(all_days)
    cut1, cut2 = int(n * 0.7), int(n * 0.85)

    def labels_present(days_subset: set[str]) -> bool:
        labs = {r for h, r, t in triples if t in days_subset and ("risesWithin_" in r or "fallsWithin_" in r)}
        return any("risesWithin_" in x for x in labs) and any("fallsWithin_" in x for x in labs)

    while cut1 < n and not labels_present(set(all_days[:cut1])):
        cut1 += max(1, n // 50)  # kleine Schritte
    cut1 = min(cut1, n - 2)
    cut2 = max(cut2, cut1 + 1)

    train_days = set(all_days[:cut1])
    valid_days = set(all_days[cut1:cut2])
    test_days  = set(all_days[cut2:])

    out_dir.mkdir(parents=True, exist_ok=True)

    def write(fname: str, days_set: set[str], include_typings: bool = False):
        p = out_dir / fname
        with p.open("w", encoding="utf-8") as f:
            if include_typings and fname == "train.tsv":
                for h, r, t in sorted(day_typings):   f.write(f"{h}\t{r}\t{t}\n")
                for h, r, t in sorted(asset_typings): f.write(f"{h}\t{r}\t{t}\n")
            for h, r, t in sorted(triples):
                if t in days_set:
                    f.write(f"{h}\t{r}\t{t}\n")

    write("train.tsv", train_days, include_typings=True)
    write("valid.tsv", valid_days)
    write("test.tsv",  test_days)

    # Kurz-Stats
    from collections import Counter
    def rel_counts(days_set): return Counter(r for h, r, t in triples if t in days_set)
    print("[Split] train/valid/test days:", len(train_days), len(valid_days), len(test_days))
    for name, days in [("train", train_days), ("valid", valid_days), ("test", test_days)]:
        lbl = {k: v for k, v in rel_counts(days).items() if "Within_" in k}
        print(f"[Labels {name}]", lbl)

if __name__ == "__main__":
    main()
