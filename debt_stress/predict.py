"""Classify banking messages with a fine-tuned model.

    python -m debt_stress.predict "Your EMI is overdue by 20 days." "Salary credited."
"""
import argparse
from functools import lru_cache
from pathlib import Path

DEFAULT_MODEL = Path("models/finbert-stress")


@lru_cache(maxsize=1)
def _load(model_dir: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    return tok, model


def predict(texts: list[str], model_dir: str | Path = DEFAULT_MODEL) -> list[dict]:
    import torch

    tok, model = _load(str(model_dir))
    enc = tok(texts, truncation=True, max_length=64, padding=True, return_tensors="pt")
    with torch.inference_mode():
        probs = model(**enc).logits.softmax(-1)
    labels = model.config.id2label
    return [
        {"text": t, "label": labels[int(p.argmax())],
         "probs": {labels[i]: round(float(v), 4) for i, v in enumerate(p)}}
        for t, p in zip(texts, probs, strict=True)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    for r in predict(args.texts, args.model):
        print(f"{r['label']:<7} {max(r['probs'].values()):.2f}  {r['text']}")


if __name__ == "__main__":
    main()
