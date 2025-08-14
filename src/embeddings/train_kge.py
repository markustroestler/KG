from pathlib import Path
import yaml, os, random, numpy as np, torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
import pandas as pd

def _cfg():
    return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}

def _set_seeds(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def _strip_instanceof(path: str) -> str:
    df = pd.read_csv(path, sep="\t", header=None, names=["h","r","t"])
    df = df[df["r"] != "ex:instanceOf"]  # optional: aus Train entfernen
    out = Path(path).with_name("train.no_types.tsv")
    df.to_csv(out, sep="\t", header=False, index=False)
    return str(out)

def main():
    cfg = _cfg(); e = cfg.get("embeddings", {}) or {}
    _set_seeds(int(e.get("seed", 42)))
    out = Path(e.get("out_dir", "data/kge"))
    model_dir = out / "model"

    train_p = str(out / "train.tsv")
    valid_p = str(out / "valid.tsv") if (out / "valid.tsv").exists() else None
    test_p  = str(out / "test.tsv")  if (out / "test.tsv").exists()  else None

    # Optional: instanceOf im Training filtern
    # train_p = _strip_instanceof(train_p)

    # -- Inverse nur im Training erzeugen
    # --- Factories
    train_tf = TriplesFactory.from_path(train_p, create_inverse_triples=True)

    val_tf = TriplesFactory.from_path(
        valid_p, entity_to_id=train_tf.entity_to_id, relation_to_id=train_tf.relation_to_id
    ) if valid_p else None

    test_tf = TriplesFactory.from_path(
        test_p, entity_to_id=train_tf.entity_to_id, relation_to_id=train_tf.relation_to_id
    ) if test_p else None

    # --- Pipeline (keine weitere create_inverse_triples-Option!)
    result = pipeline(
        training=train_tf,
        validation=val_tf,
        testing=test_tf,
        model="RotatE",
        loss="SoftplusLoss",
        model_kwargs=dict(embedding_dim=int(e.get("dim", 512))),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=float(e.get("lr", 5e-4))),
        training_kwargs=dict(num_epochs=int(e.get("epochs", 300)), use_tqdm_batch=False),
        negative_sampler="bernoulli",
        negative_sampler_kwargs={"num_negs_per_pos": 128},
        evaluator="RankBasedEvaluator",
    )

    torch.save(result.model, model_dir / "trained_model.pkl")

    # 2) Mapping/Triples (für TriplesFactory.from_path_binary(...))
    train_tf.to_path_binary(model_dir / "training_triples")

    # --- Gefilterte Eval NUR mit Roh-Tripeln (ohne Inversen) als Filter
    raw_train_tf = TriplesFactory.from_path(
        train_p, create_inverse_triples=False,
        entity_to_id=train_tf.entity_to_id, relation_to_id=train_tf.relation_to_id
    )
    raw_val_tf = val_tf  # val/test haben ja keine Inversen
    raw_test_tf = test_tf

    evaluator = RankBasedEvaluator()
    metrics = evaluator.evaluate(
        model=result.model,
        mapped_triples=raw_test_tf.mapped_triples,
        additional_filter_triples=[
            raw_train_tf.mapped_triples,
            *( [raw_val_tf.mapped_triples] if raw_val_tf else [] ),
            raw_test_tf.mapped_triples,
        ],
    )
    print("MRR:", float(metrics.get_metric("mean_reciprocal_rank")))

    # Optionale, explizite gefilterte Evaluation (pipeline deckt das bereits ab)
    if test_tf is not None:
        evaluator = RankBasedEvaluator()
        filter_triples = [tf.mapped_triples for tf in (train_tf, val_tf, test_tf) if tf is not None]
        metrics = evaluator.evaluate(
            model=result.model,
            mapped_triples=test_tf.mapped_triples,
            additional_filter_triples=filter_triples,
        )
        print("Hits@1 :", float(metrics.get_metric("hits@1")))
        print("Hits@3 :", float(metrics.get_metric("hits@3")))
        print("Hits@5 :", float(metrics.get_metric("hits@5")))
        print("Hits@10:", float(metrics.get_metric("hits@10")))
        print("MRR    :", float(metrics.get_metric("mean_reciprocal_rank")))

if __name__ == "__main__":
    main()
