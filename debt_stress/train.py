"""Fine-tune FinBERT on the synthetic messages and evaluate honestly.

    python -m debt_stress.train --out models/finbert-stress

Evaluates on (1) templates never seen in training and (2) the hand-written
challenge set, and compares against a TF-IDF + logistic-regression baseline
trained on the same split. Writes metrics.json next to the model.
"""
import argparse
import json
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

from . import data

BASE_MODEL = "ProsusAI/finbert"


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def tfidf_baseline(train, test, challenge) -> dict:
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000))
    model.fit([s.text for s in train], [s.label for s in train])
    return {
        "held_out_templates": scores([s.label for s in test], model.predict([s.text for s in test])),
        "challenge": scores([y for _, y in challenge], model.predict([t for t, _ in challenge])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("models/finbert-stress"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=64)   # longest message is ~25 tokens
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    samples = data.generate(seed=args.seed)
    train, test = data.split_by_template(samples, seed=args.seed)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3,
        id2label=dict(enumerate(data.LABELS)), label2id={v: i for i, v in enumerate(data.LABELS)},
    )
    # FinBERT ships a 3-way *sentiment* head (positive/negative/neutral). The
    # shapes match, so from_pretrained would silently keep it and map
    # "negative" onto MEDIUM. Start the stress head from scratch instead.
    model.classifier.reset_parameters()

    def to_ds(rows):
        ds = Dataset.from_dict({"text": [s.text for s in rows], "labels": [s.label for s in rows]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=64,
            weight_decay=0.01,
            eval_strategy="no",       # no peeking at the held-out templates while training
            save_strategy="no",
            logging_steps=25,
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=to_ds(train),
        processing_class=tok,
    )

    t0 = time.time()
    trainer.train()
    train_minutes = (time.time() - t0) / 60

    def predict(texts):
        enc = tok(texts, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
        model.eval()
        with torch.inference_mode():
            return model(**enc).logits.argmax(-1).numpy()

    challenge = data.CHALLENGE
    metrics = {
        "protocol": {
            "train_samples": len(train), "test_samples": len(test),
            "train_unique_texts": len({s.text for s in train}),
            "held_out_templates_per_class": 3, "base_model": BASE_MODEL,
            "epochs": args.epochs, "lr": args.lr, "max_len": args.max_len,
            "train_minutes_cpu": round(train_minutes, 1), "torch_threads": torch.get_num_threads(),
        },
        "finbert": {
            "held_out_templates": scores([s.label for s in test], predict([s.text for s in test])),
            "challenge": scores([y for _, y in challenge], predict([t for t, _ in challenge])),
        },
        "tfidf_logreg": tfidf_baseline(train, test, challenge),
    }
    errors = [
        {"text": t, "true": data.LABELS[y], "pred": data.LABELS[p]}
        for (t, y), p in zip(challenge, predict([t for t, _ in challenge]), strict=True) if y != p
    ]
    metrics["finbert"]["challenge_errors"] = errors

    args.out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "finbert"} | {
        "finbert": {k: v for k, v in metrics["finbert"].items() if k != "challenge_errors"}}, indent=2))
    print(f"{len(errors)} challenge errors; model + metrics saved to {args.out}")


if __name__ == "__main__":
    main()
