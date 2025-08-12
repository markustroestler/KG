# src/news/process_news.py
import json, glob, hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RAW_DIR = Path("data/news/raw")
OUT_DIR = Path("data/news/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
labels = ["negative", "neutral", "positive"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def load_processed_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                rid = rec.get("id") or sha1(rec.get("url") or "")
                if rid:
                    ids.add(rid)
            except Exception:
                continue
    return ids

def finbert_batch(texts: list[str]) -> list[tuple[str, float]]:
    if not texts:
        return []
    enc = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        idxs = probs.argmax(dim=-1).tolist()
    out = []
    for i, idx in enumerate(idxs):
        out.append((labels[idx], float(probs[i, idx].item())))
    return out

def main():
    files = sorted(glob.glob(str(RAW_DIR / "*.jsonl")))
    if not files:
        print("Keine RAW-News gefunden in data/news/raw/*.jsonl")
        return

    for fp in files:
        ticker = Path(fp).stem.upper()
        out_fp = OUT_DIR / f"{ticker.lower()}.jsonl"

        existing_ids = load_processed_ids(out_fp)

        # sammle neue Items
        to_analyze = []  # [(rid, rec, text)]
        seen_ids = set() # doppelte im selben RAW-File vermeiden
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rid = rec.get("id") or sha1(rec.get("url") or "")
                if not rid or rid in existing_ids or rid in seen_ids:
                    continue
                headline = rec.get("title") or rec.get("title_extracted")
                summary  = rec.get("summary")
                text = " ".join([s for s in [headline, summary] if s])
                to_analyze.append((rid, rec, text))
                seen_ids.add(rid)

        if not to_analyze:
            continue

        # batched inference
        results = []
        for i in range(0, len(to_analyze), BATCH_SIZE):
            batch = to_analyze[i:i+BATCH_SIZE]
            texts = [t[2] if (t[2] and t[2].strip()) else "" for t in batch]
            # leere Texte: markiere neutral 0.0 ohne Modell
            mask = [bool(x.strip()) for x in texts]
            preds = [("neutral", 0.0)] * len(batch)
            if any(mask):
                # nur nicht-leere durch FinBERT jagen
                idx_map = [j for j, m in enumerate(mask) if m]
                non_empty_texts = [texts[j] for j in idx_map]
                fe = finbert_batch(non_empty_texts)
                for k, (lab, prob) in zip(idx_map, fe):
                    preds[k] = (lab, prob)
            results.extend(preds)

        # anhängen
        new_cnt = 0
        with out_fp.open("a", encoding="utf-8") as w:
            for (rid, rec, _), (sent, conf) in zip(to_analyze, results):
                headline = rec.get("title") or rec.get("title_extracted")
                summary  = rec.get("summary")
                processed = {
                    "id": rid,
                    "ticker": ticker,
                    "url": rec.get("url"),
                    "title": headline,
                    "summary": summary,
                    "published_at": rec.get("published_at"),
                    "sentiment": sent,
                    "confidence": round(conf, 4),
                }
                w.write(json.dumps(processed, ensure_ascii=False) + "\n")
                new_cnt += 1


if __name__ == "__main__":
    main()
