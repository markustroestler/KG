# embeddings/calibrate_scores.py
# pip install scikit-learn joblib
from pathlib import Path
import yaml, torch, numpy as np, joblib
from pykeen.triples import TriplesFactory
from sklearn.linear_model import LogisticRegression
import os, random, numpy as np

def _set_seeds(s):
    os.environ["PYTHONHASHSEED"]=str(s); random.seed(s); np.random.seed(s)

def _cfg():
    return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}

def load_saved(model_dir: Path):
    model = torch.load(model_dir / "trained_model.pkl", map_location="cpu", weights_only=False)
    model.eval()
    tf = TriplesFactory.from_path_binary(model_dir / "training_triples")
    return model, tf

def main():
    cfg = _cfg(); e = cfg.get("embeddings", {}) or {}
    seed=int(e.get("seed",42)); _set_seeds(seed)
    out = Path(e.get("out_dir", "data/kge"))
    model_dir = out / "model"
    relation = f"ex:{e.get('target_relation','risesWithin_14d')}"

    # 1) Modell & TF laden
    model, tf = load_saved(model_dir)
    e2i, r_id = tf.entity_to_id, tf.relation_to_id.get(relation)
    if r_id is None:
        print(f"❌ Relation {relation} nicht im Modell – Abbruch.")
        return

    # 2) Ground-Truth aus valid.tsv
    val_path = out / "valid.tsv"
    if not val_path.exists() or val_path.stat().st_size == 0:
        print("⚠️ valid.tsv fehlt/leer – keine Kalibrierung möglich.")
        return

    pos = set()      # {(h,t)} für positives Label
    days = set()
    assets_from_valid = set()
    for line in val_path.read_text().splitlines():
        h, r, t = line.strip().split("\t")
        assets_from_valid.add(h)
        days.add(t)
        if r == relation:
            pos.add((h, t))

    # 3) Kandidaten bestimmen
    # a) aus config.stocks (falls vorhanden), in ex:-Form bringen
    def to_ex(a: str) -> str:
        a = str(a).upper().replace("-", "_").replace(".", "_")
        return f"ex:{a}"
    cfg_assets = [to_ex(a) for a in (cfg.get("stocks") or [])]
    # b) Fallback: alle Assets, die in valid vorkommen
    assets = [a for a in (cfg_assets or sorted(assets_from_valid)) if a in e2i]
    if not assets:
        print("⚠️ Keine Assets in valid.tsv/Config gefunden – Kalibrierung abgebrochen.")
        return

    # 4) Scores X und Labels y über alle (asset, day) in valid erstellen
    X, y = [], []
    for t in sorted(days):
        if t not in e2i:    # sollte existieren durch Day-Typing im Training
            continue
        t_id = e2i[t]
        cand = [a for a in assets if a in e2i]
        if not cand: continue

        h_ids = torch.tensor([e2i[a] for a in cand], dtype=torch.long)
        r_ids = torch.full((len(cand),), r_id, dtype=torch.long)
        t_ids = torch.full((len(cand),), t_id, dtype=torch.long)
        hrt = torch.stack([h_ids, r_ids, t_ids], dim=1)

        with torch.no_grad():
            s = model.score_hrt(hrt).view(-1).numpy()

        X.append(s.reshape(-1,1))
        y.append(np.array([1 if (a, t) in pos else 0 for a in cand], dtype=int))

    if not X:
        print("⚠️ Keine valid-Mappings erzeugt – prüfe Splits/Export.")
        return

    X = np.vstack(X); y = np.concatenate(y)

    # 5) Sanity-Checks
    n_pos = int(y.sum()); n = len(y)
    print(f"ℹ️ Calibration dataset: n={n}, positives={n_pos}, negatives={n-n_pos}")
    if n_pos == 0 or n_pos == n:
        print("⚠️ Nur eine Klasse in valid – Kalibrierung nicht sinnvoll. Abbruch.")
        return

    # 6) Platt Scaling fitten
    lr = LogisticRegression(max_iter=1000, random_state=seed).fit(X, y)
    joblib.dump(lr, model_dir / "calibrator.joblib")
    print(f"✅ Calibrator gespeichert → {model_dir/'calibrator.joblib'}")

if __name__ == "__main__":
    main()
