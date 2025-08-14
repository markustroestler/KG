# embeddings/score_up_today.py  (Ansatz A: Rank von d* pro Asset)
from pathlib import Path
from datetime import datetime, timedelta
import re, yaml, torch, numpy as np, joblib, os, random
from datetime import date as _date

from pykeen.triples import TriplesFactory

# ---------------- utils ----------------
def _set_seeds(s):
    os.environ["PYTHONHASHSEED"]=str(s); random.seed(s); np.random.seed(s); torch.manual_seed(s)

def _cfg():
    return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").upper()

def _to_ex_asset(sym: str) -> str:
    return f"ex:{_slug(sym)}"

def _to_ex_day(d: str) -> str:
    # expects YYYY-MM-DD
    return f"ex:d_{d}"

def _parse_day_label(lbl: str):
    m = re.match(r"ex:d_(\d{4}-\d{2}-\d{2})$", str(lbl))
    return datetime.strptime(m.group(1), "%Y-%m-%d").date() if m else None

def _load_model_and_tf(model_dir: Path):
    model = torch.load(model_dir / "trained_model.pkl", map_location="cpu", weights_only=False)
    model.eval()
    tf = TriplesFactory.from_path_binary(model_dir / "training_triples")
    return model, tf

def _load_calibrator(model_dir: Path):
    p = model_dir / "calibrator.joblib"
    return joblib.load(p) if p.exists() else None

# -------------- core: rank day per asset --------------
def rank_day_for_asset(model, tf: TriplesFactory, rel_label: str, asset_label: str,
                       day_labels: list[str], target_day_label: str):
    e2i, r2i = tf.entity_to_id, tf.relation_to_id
    if rel_label not in r2i or asset_label not in e2i or target_day_label not in e2i:
        return None

    r_id = r2i[rel_label]
    a_id = e2i[asset_label]
    t_star = e2i[target_day_label]

    # Kandidaten-HRT (Asset fix, Relation fix, alle Days)
    t_ids = [e2i[d] for d in day_labels if d in e2i]
    if not t_ids or t_star not in t_ids:
        return None

    h = torch.full((len(t_ids),), a_id, dtype=torch.long)
    r = torch.full((len(t_ids),), r_id, dtype=torch.long)
    t = torch.tensor(t_ids, dtype=torch.long)
    hrt = torch.stack([h, r, t], dim=1)

    with torch.no_grad():
        scores = model.score_hrt(hrt).view(-1).cpu().numpy()

    # Rang (desc): 1 = bester
    idx_star = t_ids.index(t_star)
    s_star = scores[idx_star]
    rank = int((scores > s_star).sum() + 1)
    n = len(scores)
    percentile = 1.0 - (rank - 1) / n
    # z-Score ggü. Asset-Historie
    mu, sigma = float(scores.mean()), float(scores.std() + 1e-12)
    z = (s_star - mu) / sigma
    return {
        "rank": rank, "n": n, "percentile": percentile, "z": z,
        "score": float(s_star), "asset": asset_label, "day": target_day_label,
    }

def main(target_date: str | None = None, top_k: int = 20):
    cfg = _cfg(); e = cfg.get("embeddings", {}) or {}
    _set_seeds(int(e.get("seed", 42)))
    out = Path(e.get("out_dir", "data/kge")); model_dir = out / "model"

    # --- load model + TF + calibrator ---
    model, tf = _load_model_and_tf(model_dir)
    cal = _load_calibrator(model_dir)
    rel = f"ex:{e.get('target_relation', 'risesWithin_14d')}"

    # --- target day entity ---
    if not target_date:
        target_date = (_date.today().isoformat()).strip()
    if not target_date:
        # fallback: latest day in TF
        day_labels_all = sorted([lbl for lbl in tf.entity_to_id if str(lbl).startswith("ex:d_")])
        if not day_labels_all:
            print("⚠️ No day entities in model."); return
        target_day = day_labels_all[-1]
        print(f"⚠️ Using latest available training day: {str(target_day).replace('ex:d_','')}")
    else:
        target_day = _to_ex_day(target_date)

    if target_day not in tf.entity_to_id:
        # fallback auf nächsten kleineren Tag
        all_days = sorted([lbl for lbl in tf.entity_to_id if str(lbl).startswith("ex:d_")],
                          key=lambda x: _parse_day_label(x))
        d_star = _parse_day_label(target_day)
        le_days = [d for d in all_days if _parse_day_label(d) and _parse_day_label(d) <= d_star]
        if not le_days:
            print("⚠️ Target day not in TF and no earlier day found."); return
        target_day = le_days[-1]
        print(f"⚠️ Using nearest available ≤ target: {str(target_day).replace('ex:d_','')}")

    # --- candidate day list per asset (≤ target, optional lookback) ---
    lookback_days = int(e.get("rank_window_days", 365))  # e.g., 365 Handelstage
    d_star = _parse_day_label(target_day)
    all_days = [lbl for lbl in tf.entity_to_id if str(lbl).startswith("ex:d_")]
    # Filter days ≤ d* and within lookback
    days_filtered = []
    lb_cut = d_star - timedelta(days=lookback_days)
    for lbl in all_days:
        d = _parse_day_label(lbl)
        if d and (lb_cut <= d <= d_star):
            days_filtered.append(lbl)
    days_filtered.sort(key=lambda x: _parse_day_label(x))
    if not days_filtered:
        print("⚠️ No candidate days after filtering."); return

    # --- assets from config (map to ex:) ---
    assets_cfg = [_to_ex_asset(s) for s in (cfg.get("stocks") or [])]
    assets = [a for a in assets_cfg if a in tf.entity_to_id]
    if not assets:
        print("⚠️ No matching assets from config found in TF."); return

    print("\n--- Debug Info ---")
    print("Relation:", rel)
    print("Target day:", str(target_day).replace("ex:d_", ""))
    print("Candidates: days=", len(days_filtered), "assets=", len(assets))
    print("------------------\n")

    rows = []
    for a in assets:
        res = rank_day_for_asset(model, tf, rel, a, days_filtered, target_day)
        if res:
            rows.append(res)

    if not rows:
        print("⚠️ Nothing ranked. Check relation/labels."); return

    # sort by percentile desc, then z desc
    rows.sort(key=lambda r: (r["percentile"], r["z"]), reverse=True)

    # optional: calibrated probability for (a, rel, target_day)
    probs = None
    if cal is not None:
        with torch.no_grad():
            e2i, r2i = tf.entity_to_id, tf.relation_to_id
            r_id = r2i[rel]; t_id = e2i[target_day]
            h_ids = torch.tensor([e2i[r["asset"]] for r in rows], dtype=torch.long)
            r_ids = torch.full((len(rows),), r_id, dtype=torch.long)
            t_ids = torch.full((len(rows),), t_id, dtype=torch.long)
            hrt = torch.stack([h_ids, r_ids, t_ids], dim=1)
            s = model.score_hrt(hrt).view(-1).cpu().numpy()
        probs = cal.predict_proba(s.reshape(-1,1))[:,1]

    # --- print top-k ---
    print(f"📈 Top {min(top_k, len(rows))} Assets – Rang von {str(target_day).replace('ex:d_','')} in eigener Historie")
    for i, r in enumerate(rows[:top_k], start=1):
        ptxt = f"  p(up)≈{probs[i-1]*100:4.1f}%" if probs is not None else ""
        print(f"{i:2d}. {r['asset'].replace('ex:',''):12}  "
              f"rank={r['rank']:>4}/{r['n']:<4}  "
              f"perc={r['percentile']*100:6.2f}%  "
              f"z={r['z']:+5.2f}  score={r['score']:+7.3f}{ptxt}")

if __name__ == "__main__":
    main()
