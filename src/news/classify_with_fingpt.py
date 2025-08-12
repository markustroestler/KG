import json, glob
from pathlib import Path
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "fingpt-mt"

IN_DIR = Path("data/news/processed")   # your FinBERT step could feed this
OUT_DIR = Path("data/news/fingpt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def call_ollama(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    out = r.json()["response"].strip()
    # model should return JSON text; be defensive:
    try:
        return json.loads(out)
    except Exception:
        return {"sentiment":"neutral","impact":5,"rationale":out[:300]}

def build_prompt(ticker:str, title:str, summary:str, text:str|None) -> str:
    # keep it small; you can add a 1–2k char text snippet if you want
    body = f"Ticker: {ticker}\nHeadline: {title or ''}\nSummary: {summary or ''}"
    if text:
        snippet = text.strip().replace("\n"," ")
        snippet = snippet[:1500]  # keep context short-ish
        body += f"\nText: {snippet}"
    task = ("Classify sentiment and estimate likely short-term market impact (0–10). "
            "Be concise and objective.")
    return f"{task}\n\n{body}"

def main():
    files = sorted(glob.glob(str(IN_DIR / "*.jsonl")))
    if not files:
        print("No processed news found.")
        return

    for fp in files:
        ticker = Path(fp).stem.upper()
        out_fp = OUT_DIR / f"{ticker.lower()}.jsonl"
        out = out_fp.open("w", encoding="utf-8")
        cnt=0
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                title = rec.get("headline") or rec.get("title")
                summary = rec.get("summary")
                # If you still have full text elsewhere, stitch it in here if needed
                text = rec.get("text")  # or None

                prompt = build_prompt(ticker, title, summary, text)
                res = call_ollama(prompt)

                merged = {
                    "id": rec.get("id"),
                    "ticker": rec.get("ticker", ticker),
                    "url": rec.get("url"),
                    "finbert_sentiment": rec.get("sentiment"),
                    "finbert_confidence": rec.get("confidence"),
                    "finbert_impact": rec.get("impact"),
                    "fingpt_sentiment": res.get("sentiment"),
                    "fingpt_impact": res.get("impact"),
                    "fingpt_rationale": res.get("rationale"),
                }
                out.write(json.dumps(merged, ensure_ascii=False) + "\n")
                cnt += 1
        out.close()
        print(f"✅ {ticker}: {cnt} articles classified with FinGPT → {out_fp}")

if __name__ == "__main__":
    main()
