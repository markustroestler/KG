# embeddings/score_assets_at_day.py
from pathlib import Path
from datetime import datetime
import re, yaml, torch, numpy as np, joblib, os, random
from pykeen.triples import TriplesFactory
from datetime import date as _date

# ---------- utils ----------
def _set_seeds(s):
    os.environ["PYTHONHASHSEED"]=str(s); random.seed(s); np.random.seed(s); torch.manual_seed(s)

def _cfg():
    return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}

def _slug(s: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").upper()

def _to_ex_asset(sym: str) -> str:
    return f"ex:{_slug(sym)}"

def _to_ex_day(d: str) -> str:
    return f"ex:d_{d}"  # erwartet YYYY-MM-DD

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

# ---------- core: cross-asset ranking at fixed day ----------
def main():
    """
    Cross-Asset-Ranking für Relation ex:<target_relation> am Tag aus config.embeddings.split_anchor
    (Fallback: letzter verfügbarer Day aus dem TF). Nutzt stocks aus config.stocks.
    Druckt Top-K und gibt Liste von Dicts zurück.
    """
    cfg = _cfg(); e = cfg.get("embeddings", {}) or {}
    _set_seeds(int(e.get("seed", 42)))
    out = Path(e.get("out_dir", "data/kge")); model_dir = out / "model"

    # load
    model, tf = _load_model_and_tf(model_dir)
    cal = _load_calibrator(model_dir)

    # relation
    rel = f"ex:{e.get('target_relation', 'risesWithin_14d')}"
    if rel not in tf.relation_to_id:
        print(f"⚠️ Relation fehlt im Mapping: {rel}"); return []

    # target day
    # statt: target_date = (e.get("split_anchor") or "").strip()
    target_date = ( _date.today().isoformat()).strip()  # YYYY-MM-DD
    t_lbl = _to_ex_day(target_date)
    if t_lbl not in tf.entity_to_id:
        # fallback: nächstkleinerer Day im Mapping
        all_days = sorted([lbl for lbl in tf.entity_to_id if str(lbl).startswith("ex:d_")], key=_parse_day_label)
        d_star = datetime.strptime(target_date, "%Y-%m-%d").date()
        le = [d for d in all_days if _parse_day_label(d) and _parse_day_label(d) <= d_star]
        if not le:
            print("⚠️ Target day nicht gefunden und kein früherer verfügbar."); return []
        t_lbl = le[-1]
        print(f"⚠️ Using nearest available ≤ target: {str(t_lbl).replace('ex:d_','')}")
    else:
        # letzter verfügbarer Day
        days = [lbl for lbl in tf.entity_to_id if str(lbl).startswith("ex:d_")]
        if not days:
            print("⚠️ Keine Day-Entities im Modell."); return []
        t_lbl = sorted(days, key=_parse_day_label)[-1]
        print(f"⚠️ Using latest available training day: {str(t_lbl).replace('ex:d_','')}")

    e2i, r2i = tf.entity_to_id, tf.relation_to_id
    r_id, t_id = r2i[rel], e2i[t_lbl]

    # assets aus config
    assets_all = cfg.get("stocks") or []
    assets = [a for a in map(_to_ex_asset, assets_all) if a in e2i]
    if not assets:
        print("⚠️ Keine passenden Assets aus config.stocks im Entity-Mapping."); return []

    # batch scoring
    h_ids = torch.tensor([e2i[a] for a in assets], dtype=torch.long)
    r_ids = torch.full((len(assets),), r_id, dtype=torch.long)
    t_ids = torch.full((len(assets),), t_id, dtype=torch.long)
    hrt = torch.stack([h_ids, r_ids, t_ids], dim=1)

    with torch.no_grad():
        scores = model.score_hrt(hrt).view(-1).cpu().numpy()

    # optional: Kalibrierung
    probs = None
    if cal is not None:
        try:
            probs = cal.predict_proba(scores.reshape(-1,1))[:,1]
        except Exception as ex:
            print(f"⚠️ Kalibrierung fehlgeschlagen: {ex}")

    # sortieren
    order = np.argsort(-scores)
    assets_sorted = [assets[i] for i in order]
    scores_sorted = scores[order]
    probs_sorted = probs[order] if probs is not None else None

    top_k = int(e.get("top_k", 20))
    n = min(top_k, len(assets_sorted))
    print(f"📊 Cross-Asset-Ranking für '{rel}' am {str(t_lbl).replace('ex:d_','')}")
    for i in range(n):
        sym = assets_sorted[i].replace("ex:","")
        if probs_sorted is None:
            print(f"{i+1:2d}. {sym:12}  score={scores_sorted[i]:+8.4f}")
        else:
            print(f"{i+1:2d}. {sym:12}  score={scores_sorted[i]:+8.4f}  p(up)≈{probs_sorted[i]*100:5.1f}%")

    return [
        {"rank": i+1, "asset": assets_sorted[i], "score": float(scores_sorted[i]),
         "prob": (None if probs_sorted is None else float(probs_sorted[i])),
         "relation": rel, "day": str(t_lbl)}
        for i in range(len(assets_sorted))
    ]

