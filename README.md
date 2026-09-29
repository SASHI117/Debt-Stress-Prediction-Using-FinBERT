# Debt-Stress Classification with FinBERT

[![CI](https://github.com/SASHI117/Debt-Stress-Prediction-Using-FinBERT/actions/workflows/ci.yml/badge.svg)](https://github.com/SASHI117/Debt-Stress-Prediction-Using-FinBERT/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SASHI117/Debt-Stress-Prediction-Using-FinBERT/blob/main/DebtStressPrediction.ipynb)
![Transformers](https://img.shields.io/badge/HF%20Transformers-FinBERT-yellow)

This project fine-tunes [FinBERT](https://huggingface.co/ProsusAI/finbert)
to sort banking messages (EMI reminders, balance alerts, overdue notices)
into **LOW / MEDIUM / HIGH financial stress**. A lender or fintech app could
use this kind of signal to route customers towards hardship support before
they default.

The model is the easy part. This repository is mostly about **evaluating it
honestly**: the first version reported 100% accuracy, and that number turned
out to measure memorization.

## Results

| Evaluation | FinBERT (fine-tuned) | TF-IDF + logistic regression |
|---|---|---|
| Random 80/20 split (original protocol) | 100% | 100% |
| **Held-out templates** (653 messages, wording never seen in training) | **54.1%** acc · 0.552 macro-F1 | 40.9% · 0.358 |
| Hand-written challenge set (24 bank-style messages) | 54.2% · 0.537 | 54.2% · 0.510 |

Chance is 33%. The committed run is in [`reports/metrics.json`](reports/metrics.json).
It trained on CPU in 5.8 minutes: 3 epochs, lr 2e-5, max length 64.

### Why the first number was 100%

The training data is synthetic: 30 sentence templates (10 per class) with
random amounts and day counts filled in. 2,100 samples contain only **619
distinct sentences**. With a random split, **74% of test sentences appear
word for word in the training set**, so any model scores 100%, including a
bag-of-words baseline. The random split tested lookup, not understanding.

`debt_stress/data.py` instead holds out **3 whole templates per class**. The
model is then tested on phrasings it has never seen, which is the situation
a deployed classifier is always in.

### What the held-out results say

```
                 predicted LOW  MEDIUM  HIGH
true LOW                  145      88     0
true MEDIUM                 0      74   133
true HIGH                   0      79   134
```

- FinBERT's pretraining helps: +13 points over TF-IDF on unseen wording.
  It **never confuses LOW with HIGH**, so it has learned the direction of
  financial sentiment.
- It **can't locate the MEDIUM/HIGH boundary**. With seven phrasings per class,
  nothing tells it that "due this week" is medium but "overdue" is high.
- Errors on the challenge set show the gaps. It misses negation ("No overdue
  amount on your loan account" → HIGH), Indian SMS shorthand ("Avl bal",
  "MAB", "NACH"), and neutral credit messages. "Salary of Rs.45,000
  credited" comes out as MEDIUM at 0.50 confidence.
- On 24 hand-written messages FinBERT and TF-IDF tie on accuracy. At that
  sample size the difference isn't measurable.

**Conclusion:** the pipeline works, but the dataset is the bottleneck. The
next steps are real, labelled messages (even a few hundred), more
templates per class written by different people, and negation or
contrastive examples.

## Usage

```bash
pip install -r requirements.txt
python -m debt_stress.train --out models/finbert-stress        # ~6 min on CPU
python -m debt_stress.predict --model models/finbert-stress \
    "Your EMI is overdue by 20 days." "Low balance alert: Rs.210 remaining."
```

```text
HIGH    1.00  Your EMI is overdue by 20 days.
MEDIUM  0.99  Low balance alert: Rs.210 remaining.
```

## Layout

| Path | |
|---|---|
| `debt_stress/data.py` | templates, seeded generation, template-held-out and random splits, challenge set |
| `debt_stress/train.py` | fine-tuning plus evaluation against the TF-IDF baseline. Writes `metrics.json` |
| `debt_stress/predict.py` | batch inference with class probabilities |
| `tests/` | balance, determinism, no template or sentence overlap, and a test that documents the leakage of the random split |
| `DebtStressPrediction.ipynb` | the original Colab run, kept as a record, with a note on its result |

## Implementation notes

- FinBERT already has a 3-way head, for positive/negative/neutral
  sentiment. Because the shapes match, `from_pretrained(num_labels=3)`
  silently keeps it, and "negative" would start out mapped to MEDIUM. The
  classifier is re-initialized explicitly.
- There is no evaluation during training, so the held-out templates are
  never used for model selection.
- Max length is 64 tokens, since the longest message is about 25. The
  original notebook padded every message to 512 (FinBERT's maximum), so most
  of each batch was padding.

## Limitations

- All training data is synthetic and in English. Real banking SMS in India
  mixes languages and abbreviations.
- The challenge-set labels are my own judgement, and n = 24 is a smoke
  test, not a benchmark.
- "Stress" is inferred from a single message. A useful system would
  aggregate a customer's message history.

## License

MIT
