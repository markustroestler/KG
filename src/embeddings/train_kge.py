from pathlib import Path
import yaml
import os, random, numpy as np, torch

def _cfg():
    from pathlib import Path
    return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8")) or {}

def _set_seeds(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.use_deterministic_algorithms(False)

def main():
    # pip install pykeen
    from pykeen.pipeline import pipeline
    cfg = _cfg()
    e = cfg.get("embeddings", {}) or {}
    seed = int(e.get("seed", 42))
    _set_seeds(seed)
    out = Path(e.get("out_dir", "data/kge"))

    # neue: LR + Early Stopping Defaults
    lr = float(e.get("lr", 1e-3))
    use_es = bool(e.get("early_stopping", True))
    patience = int(e.get("patience", 10))
    rel_delta = float(e.get("relative_delta", 0.001))

    triples = {
        "training":   str(out / "train.tsv"),
        "validation": str(out / "valid.tsv"),
        "testing":    str(out / "test.tsv"),
    }

    val_path = out / "valid.tsv"
    test_path = out / "test.tsv"
    val = str(val_path) if val_path.exists() and val_path.stat().st_size > 0 else None
    tst = str(test_path) if test_path.exists() and test_path.stat().st_size > 0 else None

    result = pipeline(
    training=str(out / "train.tsv"),
    validation=val,
    testing=tst,
    model=e.get("model","RotatE"),
    model_kwargs={"embedding_dim": int(e.get("dim",256))},
    optimizer="Adam",
    optimizer_kwargs={"lr": float(e.get("lr", 3e-4))},
    loss=e.get("loss", "SoftplusLoss"),
    training_kwargs={
        "num_epochs": int(e.get("epochs",200)),
        "batch_size": int(e.get("batch_size",1024)),
        "num_workers": 0, "drop_last": False, "pin_memory": False,
    },
    negative_sampler="basic",
    negative_sampler_kwargs={"num_negs_per_pos": int(e.get("negs",50))},
    regularizer="LpRegularizer",
    regularizer_kwargs={"weight": float(e.get("reg",1e-5))},
    random_seed=seed,
    evaluator="RankBasedEvaluator",
    stopper="early" if (use_es and val) else None,
    stopper_kwargs={"frequency":1, "patience": int(e.get("patience",20)),
                    "relative_delta": float(e.get("relative_delta",0.0005)),
                    "metric":"mean_reciprocal_rank"} if (use_es and val) else None,
    )

    result.save_to_directory(str(out / "model"))
    print(f"✅ KGE model saved → {out/'model'}")

    if val:
        valid_res = result.metric_results.to_dict()
        print("📊 VALID:", {k: round(v, 4) for k, v in valid_res.items()
                        if k in ("mrr", "hits_at_1", "hits_at_3", "hits_at_10")})

    if tst:
        test_res = result.metric_results.to_dict()
        print("📊 TEST :", {k: round(v, 4) for k, v in test_res.items()
                        if k in ("mrr", "hits_at_1", "hits_at_3", "hits_at_10")})
        
        best_epoch = getattr(getattr(result, "stopper", None), "best_epoch", None)
    print("⭐ Best epoch:", best_epoch)
    metrics = result.metric_results.to_dict()
    print("📊 TEST/VALID:", {k: round(v,4) for k,v in metrics.items() if k in ("mrr","hits_at_1","hits_at_3","hits_at_10")})
    m = result.metric_results
    m = result.metric_results
    print("MRR:", float(m.get_metric("mrr")))



if __name__ == "__main__":
    main()
