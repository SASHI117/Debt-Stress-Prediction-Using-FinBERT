# Debt-Stress Classification with FinBERT

[![CI](https://github.com/SASHI117/Debt-Stress-Prediction-Using-FinBERT/actions/workflows/ci.yml/badge.svg)](https://github.com/SASHI117/Debt-Stress-Prediction-Using-FinBERT/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SASHI117/Debt-Stress-Prediction-Using-FinBERT/blob/main/DebtStressPrediction.ipynb)
![Transformers](https://img.shields.io/badge/HF%20Transformers-FinBERT-yellow)

This project fine-tunes [FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model pre-trained on
financial text, to sort banking messages (EMI reminders, balance alerts, overdue notices) into
**LOW / MEDIUM / HIGH financial stress**. A lender or fintech app can use this signal to route
customers towards hardship support before they default.

## Highlights

- **Domain-adapted transformer.** FinBERT's financial pre-training gives it a head start on banking
  language. A fresh 3-way classification head is trained on top.
- **Leakage-free evaluation.** The training messages are generated from sentence templates, so
  whole templates are held out for testing. The model is scored only on phrasings it has never seen.
- **Measured against a baseline.** On unseen wording FinBERT beats a TF-IDF + logistic-regression
  baseline by **13 accuracy points** (54.1% vs 40.9%, macro-F1 0.552 vs 0.358). It keeps LOW and
  HIGH stress cleanly apart: no LOW message was ever predicted as HIGH, or the reverse.
- **Fast to train.** 3 epochs on a CPU in **under 6 minutes**, with max length 64.
- **Reproducible.** A seeded data generator, a single training command, and metrics written to
  [`reports/metrics.json`](reports/metrics.json).

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

## How it works

```
templates (10 per class) ──► seeded generator ──► 2,100 messages
                                   │
             split by template: 7 train / 3 held out per class
                                   │
       ProsusAI/finbert + new 3-way head ──► fine-tune (3 epochs, lr 2e-5)
                                   │
    evaluate on held-out templates + hand-written bank-style messages
```

| Setting | Value |
|---|---|
| Base model | `ProsusAI/finbert` (BERT-base) |
| Head | fresh 3-way classifier (FinBERT's sentiment head is re-initialized) |
| Optimizer | AdamW, lr 2e-5, weight decay 0.01, 3 epochs, batch 16 |
| Max length | 64 tokens (the longest message is about 25) |
| Evaluation | held-out templates, plus 24 hand-written messages in bank SMS style |

## Layout

| Path | |
|---|---|
| `debt_stress/data.py` | templates, seeded generation, template-level split, hand-written evaluation messages |
| `debt_stress/train.py` | fine-tuning and evaluation against the TF-IDF baseline. Writes `metrics.json` |
| `debt_stress/predict.py` | batch inference with class probabilities |
| `tests/` | class balance, determinism, and train/test separation checks |
| `DebtStressPrediction.ipynb` | the original Colab experiment |

## Implementation notes

- FinBERT ships with a 3-way *sentiment* head (positive/negative/neutral). Because the shapes match
  a 3-way stress head, it is re-initialized explicitly, so training starts from a neutral classifier.
- There is no evaluation during training, so the held-out templates are never used for model selection.
- Padding to 64 tokens instead of 512 keeps every batch compact, which is what makes CPU training practical.

## Roadmap

- Fine-tune on real, labelled banking messages alongside the synthetic set.
- Add multilingual and code-mixed SMS: Indian bank messages mix English, Hindi and abbreviations.
- Aggregate a customer's message history into a single stress score.

## License

MIT
