from pathlib import Path
from datetime import date
import re, yaml, torch, numpy as np
import joblib
from pykeen.triples import TriplesFactory
import os, random, numpy as np, torch

def _set_seeds(s):
    os.environ["PYTHONHASHSEED"]=str(s); random.seed(s); np.random.seed(s); torch.manual_seed(s)

def _cfg():
    return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}

def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def load_saved(model_dir="data/kge/model"):
    md = Path(model_dir)
    model = torch.load(md / "trained_model.pkl", map_location="cpu", weights_only=False)
    model.eval()
    tf = TriplesFactory.from_path_binary(md / "training_triples")
    return model, tf

def _rank_heads_restricted(model, tf, relation_label, tail_label, candidate_head_labels, top_k=20, remove_known=True):
    cand_ids = [tf.entity_to_id[x] for x in candidate_head_labels if x in tf.entity_to_id]
    if not cand_ids or relation_label not in tf.relation_to_id or tail_label not in tf.entity_to_id:
        return [], []

    r_id = tf.relation_to_id[relation_label]
    t_id = tf.entity_to_id[tail_label]
    h_ids = torch.tensor(cand_ids, dtype=torch.long)
    r_ids = torch.full((len(cand_ids),), r_id, dtype=torch.long)
    t_ids = torch.full((len(cand_ids),), t_id, dtype=torch.long)
    hrt = torch.stack([h_ids, r_ids, t_ids], dim=1)

    with torch.no_grad():
        scores = model.score_hrt(hrt).view(-1).numpy()

    known_h = set()
    if remove_known:
        mt = tf.mapped_triples
        mask = (mt[:, 1] == r_id) & (mt[:, 2] == t_id)
        if mask.any():
            known_h = set(mt[mask][:, 0].tolist())

    id2ent = {v: k for k, v in tf.entity_to_id.items()}
    order = np.argsort(scores)[::-1]
    ranked = []
    ranked_scores = []
    for pos in order:
        hid = cand_ids[pos]
        if hid in known_h:
            continue
        ranked.append((id2ent[hid], float(scores[pos])))
        ranked_scores.append(float(scores[pos]))
        if len(ranked) >= top_k:
            break
    return ranked, np.array(ranked_scores)

def _fallback_filter_labels(tf):
    labels = []
    for lbl in tf.entity_to_id.keys():
        s = str(lbl)
        if s.startswith("ex:d_"):
            continue
        if "news_" in s:
            continue
        if s.endswith("_fund_") or "_earn_" in s or "_div_" in s or "_split_" in s or "_cons_" in s:
            continue
        labels.append(s)
    return labels

def _softmax(x: np.ndarray, T: float = 1.0) -> np.ndarray:
    # T > 1 = flacher, T < 1 = spitzer
    x = (x - np.max(x)) / max(T, 1e-9)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

def load_calibrator(model_dir="data/kge/model"):
    p = Path(model_dir) / "calibrator.joblib"
    return joblib.load(p) if p.exists() else None

def main(target_date: str | None = None, top_k: int = 20):
    cfg = _cfg()
    e = cfg.get("embeddings", {}) or {}
    _set_seeds(int(e.get("seed",42)))
    model_dir = Path(e.get("out_dir", "data/kge")) / "model"

    model, tf = load_saved(model_dir)

    if not target_date:
        target_date = (_cfg().get("embeddings", {}) or {}).get("split_anchor")
    date_entity = f"ex:d_{target_date}"
    relation    = f"ex:{e.get('target_relation', 'risesWithin_14d')}"

    if date_entity not in tf.entity_to_id:
        day_labels = sorted([lbl for lbl in tf.entity_to_id.keys() if str(lbl).startswith("ex:d_")])
        if not day_labels:
            print("⚠️ No day entities found in training set.")
            return
        date_entity = day_labels[-1]
        print(f"⚠️ Using latest available training day: {str(date_entity).replace('ex:d_','')}")

    tickers = [str(t).upper() for t in (_cfg().get("stocks") or [])]
    candidate_head_labels = [f"ex:{slug(t)}" for t in tickers]

    have_ids = any(lbl in tf.entity_to_id for lbl in candidate_head_labels)
    # if not have_ids:
    #     candidate_head_labels = _fallback_filter_labels(tf)
    if not any(lbl in tf.entity_to_id for lbl in candidate_head_labels):
        print("❌ Keine Kandidaten im Modell (prüfe stocks & Export)."); return

    print("\n--- Debug Info ---")
    print("Relations in model:", list(tf.relation_to_id.keys()))
    print("Entities sample:", list(list(tf.entity_to_id.keys())[:10]))
    print("Configured tickers:", cfg.get("stocks"))
    print("------------------\n")

    ranked, raw_scores = _rank_heads_restricted(
        model, tf,
        relation_label=relation,
        tail_label=date_entity,
        candidate_head_labels=candidate_head_labels,
        top_k=top_k,
        remove_known=False,  # for scoring even known triples
    )
    if not ranked:
        print("⚠️ No candidates to rank (check triples / tickers / training window).")
        return

    # Softmax for display
    T = float((_cfg().get("embeddings", {}) or {}).get("softmax_temp", 15.0))
    sm = _softmax(raw_scores, T=T)
    print(f"📈 Top {len(ranked)} predicted up-movers on "
      f"{str(date_entity).replace('ex:d_','')} (softmax T={T}):")
    for i, ((label, score), p) in enumerate(zip(ranked, sm), start=1):
        print(f"{i:2d}. {str(label).replace('ex:',''):12}  "
          f"score={score:+7.3f}  softmax≈{p*100:5.1f}%")

    # Calibrated probabilities if available
    cal = load_calibrator(model_dir)
    if cal is not None:
        probs = cal.predict_proba(raw_scores.reshape(-1,1))[:,1]
        print("\n🎯 Calibrated probabilities (Platt scaling):")
        for (label, score), p in zip(ranked, probs):
            print(f"{str(label).replace('ex:',''):12}  p(up)≈{p*100:5.1f}%  (score={score:+.3f})")
    else:
        print("\nℹ️ No calibrator found – softmax values above are for display only.")

if __name__ == "__main__":
    main()
