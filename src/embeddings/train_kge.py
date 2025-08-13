from pathlib import Path
import yaml, os, random, numpy as np, torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory
import pandas as pd

def _cfg():
    return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}

def _set_seeds(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def _strip_instanceof(path: str) -> str:
    df = pd.read_csv(path, sep="\t", header=None, names=["h","r","t"])
    df = df[df["r"] != "ex:instanceOf"]            # <- raus aus Training
    out = Path(path).with_name("train.no_types.tsv")
    df.to_csv(out, sep="\t", header=False, index=False)
    return str(out)

def main():
    cfg = _cfg(); e = cfg.get("embeddings", {}) or {}
    _set_seeds(int(e.get("seed", 42)))
    out = Path(e.get("out_dir", "data/kge"))

    train_p = str(out/"train.tsv")
    valid_p = str(out/"valid.tsv") if (out/"valid.tsv").exists() else None
    test_p  = str(out/"test.tsv")  if (out/"test.tsv").exists()  else None

    # 1) instanceOf aus Train entfernen
    # train_p = _strip_instanceof(train_p)

    result = pipeline(
        training=train_p,
        validation=valid_p,
        testing=test_p,
        model="RotatE",
        loss="softplus",
        model_kwargs=dict(embedding_dim=int(e.get("dim", 512))),
        optimizer_kwargs=dict(lr=float(e.get("lr", 1e-3))),
        training_kwargs=dict(num_epochs=int(e.get("epochs", 300)), use_tqdm_batch=False),
        evaluator="RankBasedEvaluator",
        # 2) härtere Negatives
        negative_sampler="basic",
    )
    # 1) Triples aus Pfaden laden
    tf_train = TriplesFactory.from_path(out / "train.tsv")
    tf_valid = TriplesFactory.from_path(out / "valid.tsv")
    tf_test  = TriplesFactory.from_path(out / "test.tsv")

    evaluator = RankBasedEvaluator()

    # 2) Gefilterte Evaluation
    metrics = evaluator.evaluate(
        model=result.model,
        mapped_triples=tf_test.mapped_triples,
        additional_filter_triples=[tf_train.mapped_triples, tf_valid.mapped_triples, tf_test.mapped_triples],
        # optional: batch_size=2048,
    )

    # 3) Metriken ausgeben
    print("Hits@1 :", float(metrics.get_metric("hits@1")))
    print("Hits@3 :", float(metrics.get_metric("hits@3")))
    print("Hits@5 :", float(metrics.get_metric("hits@5")))
    print("Hits@10:", float(metrics.get_metric("hits@10")))
    print("MRR    :", float(metrics.get_metric("mean_reciprocal_rank")))

if __name__ == "__main__":
    main()
