# src/reco/recommend.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import yaml
from rdflib import Graph

EX = "http://example.org/"

@dataclass
class Weights:
    trend: float
    news: float
    consensus: float
    penalty_near_earnings: float

@dataclass
class Metrics:
    ticker: str
    plus: int = 0
    minus: int = 0
    equal: int = 0
    pos_news: int = 0
    neu_news: int = 0
    neg_news: int = 0
    net_now: int | None = None
    net_3m: int | None = None
    next_earn_date: date | None = None

    @property
    def trend_ratio(self) -> float:
        total = self.plus + self.minus
        if total <= 0: return 0.0
        return (self.plus - self.minus) / total  # -1..+1

    @property
    def news_ratio(self) -> float:
        total = self.pos_news + self.neg_news
        if total <= 0: return 0.0
        return (self.pos_news - self.neg_news) / total  # -1..+1

    @property
    def consensus_momentum(self) -> float:
        if self.net_now is None: return 0.0
        # If we have a 3m snapshot, use momentum; else use net_now alone
        if self.net_3m is not None:
            return (self.net_now - self.net_3m) / max(1, abs(self.net_3m) + abs(self.net_now))
        # map to -1..+1 by dividing by a reasonable scale (40 analysts ~= 1.0)
        return max(-1.0, min(1.0, self.net_now / 40.0))

def _read_cfg() -> dict:
    return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}

def _weights(cfg: dict) -> Weights:
    w = cfg.get("reco", {}).get("weights", {})
    return Weights(
        trend=float(w.get("trend", 0.4)),
        news=float(w.get("news", 0.3)),
        consensus=float(w.get("consensus", 0.3)),
        penalty_near_earnings=float(w.get("penalty_near_earnings", -0.5)),
    )

def _cutoffs(cfg: dict) -> tuple[str, str]:
    tw = int(cfg.get("reco", {}).get("trend_window_days", 30))
    nw = int(cfg.get("reco", {}).get("news_window_days", 14))
    trend_cutoff = (date.today() - timedelta(days=tw)).isoformat()
    news_cutoff  = (date.today() - timedelta(days=nw)).isoformat()
    return trend_cutoff, news_cutoff

def _fetch_trend_counts(g: Graph, cutoff_iso: str) -> dict[str, Metrics]:
    q = f"""
    PREFIX ex:  <{EX}>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?asset ?sym (COUNT(?t) AS ?cnt)
    WHERE {{
      ?t a ex:DailyTrend ; ex:aboutAsset ?asset ; ex:toDate ?d ; ex:symbol ?sym .
      FILTER(?d >= "{cutoff_iso}"^^xsd:date)
    }}
    GROUP BY ?asset ?sym
    """
    out: dict[str, Metrics] = {}
    for asset, sym, cnt in g.query(q):
        tk = str(asset).split("/")[-1]
        m = out.setdefault(tk, Metrics(ticker=tk))
        s = str(sym) if sym is not None else ""
        n = int(cnt)
        if s == "+": m.plus += n
        elif s in ("−", "-"): m.minus += n
        else: m.equal += n
    return out

def _fetch_news_counts(g: Graph, cutoff_iso: str) -> dict[str, tuple[int,int,int]]:
    q = f"""
    PREFIX ex:  <{EX}>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?asset
           (SUM(IF(?sent = ex:Positive, 1, 0)) AS ?pos)
           (SUM(IF(?sent = ex:Neutral,  1, 0)) AS ?neu)
           (SUM(IF(?sent = ex:Negative, 1, 0)) AS ?neg)
    WHERE {{
      {{ ?asset ex:hasNews ?n }} UNION {{ ?n ex:aboutAsset ?asset }} .
      ?n ex:publishedDate ?d ; ex:sentiment ?sent .
      BIND(IF(datatype(?d)=xsd:dateTime, xsd:date(?d), ?d) AS ?d1)
      FILTER(?d1 >= "{cutoff_iso}"^^xsd:date)
    }}
    GROUP BY ?asset
    """
    out = {}
    for asset, pos, neu, neg in g.query(q):
        tk = str(asset).split("/")[-1]
        out[tk] = (int(pos), int(neu), int(neg))
    return out

def _fetch_next_earnings(g: Graph) -> dict[str, date]:
    today = date.today().isoformat()
    q = f"""
    PREFIX ex:  <{EX}>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?asset (MIN(?d) AS ?next)
    WHERE {{
      ?e a ex:EarningsEvent ; ex:aboutAsset ?asset ; ex:onDate ?d .
      FILTER(?d >= "{today}"^^xsd:date)
    }}
    GROUP BY ?asset
    """
    out = {}
    for asset, nxt in g.query(q):
        tk = str(asset).split("/")[-1]
        if nxt is not None:
            out[tk] = nxt.toPython()
    return out

def _fetch_consensus_net(g: Graph) -> dict[str, dict[str,int]]:
    """
    Reads EX.RecoSummary nodes (period "0m","-1m","-2m","-3m") if present.
    Falls back to EX.AnalystConsensus node (onDate=today) as '0m' only.
    Returns: {ticker: {"0m": net, "-3m": net or None}}
    net = (strongBuy + buy) - (sell + strongSell)
    """
    out: dict[str, dict[str,int]] = {}
    # Try detailed RecoSummary first
    q1 = f"""
    PREFIX ex: <{EX}>
    SELECT ?asset ?period ?sb ?b ?h ?s ?ss
    WHERE {{
      ?r a ex:RecoSummary ;
         ex:aboutAsset ?asset ;
         ex:period ?period ;
         ex:strongBuy ?sb ; ex:buy ?b ; ex:hold ?h ; ex:sell ?s ; ex:strongSell ?ss .
      FILTER(?period IN ("0m","-3m"))
    }}
    """
    got_any = False
    for asset, period, sb, b, _, s, ss in g.query(q1):
        got_any = True
        tk = str(asset).split("/")[-1]
        net = int(sb) + int(b) - (int(s) + int(ss))
        out.setdefault(tk, {})[str(period)] = net

    if got_any:
        return out

    # Fallback to AnalystConsensus (single snapshot)
    q2 = f"""
    PREFIX ex: <{EX}>
    SELECT ?asset ?buy ?hold ?sell
    WHERE {{
      ?n a ex:AnalystConsensus ;
         ex:aboutAsset ?asset ;
         ex:buy ?buy ; ex:hold ?hold ; ex:sell ?sell .
    }}
    """
    for asset, b, _, s in g.query(q2):
        tk = str(asset).split("/")[-1]
        net = int(b) - int(s)
        out.setdefault(tk, {})["0m"] = net
    return out

def _score(m: Metrics, w: Weights, earnings_buffer_days: int) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []
    # Trend
    t = m.trend_ratio
    score += w.trend * t
    reasons.append(f"trend {t:+.2f}")
    # News
    n = m.news_ratio
    score += w.news * n
    reasons.append(f"news {n:+.2f}")
    # Consensus
    c = m.consensus_momentum
    score += w.consensus * c
    reasons.append(f"cons {c:+.2f}")
    # Earnings proximity penalty
    if m.next_earn_date:
        days = (m.next_earn_date - date.today()).days
        if days >= 0 and days <= earnings_buffer_days:
            score += w.penalty_near_earnings
            reasons.append(f"earnings in {days}d (penalty {w.penalty_near_earnings:+.2f})")
    return score, reasons

def run(g: Graph):
    cfg = _read_cfg()
    w = _weights(cfg)
    trend_cutoff, news_cutoff = _cutoffs(cfg)
    earnings_buffer = int(cfg.get("reco", {}).get("earnings_buffer_days", 5))

    metrics = _fetch_trend_counts(g, trend_cutoff)
    news = _fetch_news_counts(g, news_cutoff)
    for tk, (p, n, ng) in news.items():
        m = metrics.setdefault(tk, Metrics(ticker=tk))
        m.pos_news, m.neu_news, m.neg_news = p, n, ng

    # consensus
    cons = _fetch_consensus_net(g)
    for tk, per in cons.items():
        m = metrics.setdefault(tk, Metrics(ticker=tk))
        m.net_now = per.get("0m")
        m.net_3m  = per.get("-3m")

    # earnings
    nxt = _fetch_next_earnings(g)
    for tk, d in nxt.items():
        m = metrics.setdefault(tk, Metrics(ticker=tk))
        m.next_earn_date = d

    # score & print
    rows = []
    for tk, m in metrics.items():
        score, reasons = _score(m, w, earnings_buffer)
        rows.append((score, tk, m, reasons))

    rows.sort(reverse=True, key=lambda x: x[0])

    print("\n🔎 Recommendations (rule-based, higher is better):")
    for score, tk, m, reasons in rows:
        trend_str = f"{m.plus}+/{m.minus}- ({m.trend_ratio:+.2f})"
        news_str  = f"{m.pos_news}+/ {m.neg_news}- ({m.news_ratio:+.2f})"
        cons_str  = f"net now={m.net_now} vs -3m={m.net_3m}" if m.net_now is not None else "cons=?"
        earn_str  = f"next ER={m.next_earn_date}" if m.next_earn_date else "next ER=?"
        print(f"  {tk:<10} score={score:+.2f}  trend={trend_str}  news={news_str}  {cons_str}  {earn_str}")
        print(f"      why: " + ", ".join(reasons))
