"""Synthetic financial-message dataset with template-aware splitting.

Every sample is one of 30 templates with a random amount or day count
filled in. Evaluation splits by template, so the test set contains only
phrasings the model has never seen: it measures whether the model recognises
financial stress in new wording.
"""
import random
from dataclasses import dataclass

LABELS = ["LOW", "MEDIUM", "HIGH"]

# {amt} = currency amount, {days} = day count
TEMPLATES = {
    0: [
        "Your payment of ${amt} was received successfully.",
        "Thank you! No dues remaining on your account.",
        "Your credit score has improved this month.",
        "Your loan has been closed successfully.",
        "Your account balance is healthy with ${amt}.",
        "Your EMI for this month has been paid on time.",
        "Good news! Your credit limit has been increased.",
        "Your savings interest of ${amt} has been credited.",
        "Your repayment history is excellent.",
        "Your account is operating normally with no alerts.",
    ],
    1: [
        "Your account balance is below the minimum requirement.",
        "Your upcoming EMI of ${amt} is due in {days} days.",
        "A reminder: Your credit card bill of ${amt} is due soon.",
        "Your recent transaction of ${amt} is pending review.",
        "Your auto-debit is scheduled for tomorrow.",
        "You have a low balance warning on your account.",
        "Your credit score has dropped slightly.",
        "Your EMI payment window closes soon.",
        "Your loan repayment is due this week.",
        "Your account has unusual spending activity.",
    ],
    2: [
        "Your EMI payment is overdue by {days} days.",
        "Your account has insufficient funds for the recent transaction of ${amt}.",
        "Your loan is marked as 'high risk' due to non-payment.",
        "Your credit card bill of ${amt} is long overdue.",
        "Urgent: Your account is below the critical balance limit.",
        "Your payment has failed due to insufficient funds.",
        "Your credit score has dropped significantly this month.",
        "Your loan recovery process has been initiated.",
        "Your repayment is overdue and late fees have been applied.",
        "Your account has been flagged for financial risk assessment.",
    ],
}


@dataclass(frozen=True)
class Sample:
    text: str
    label: int
    template_id: str   # e.g. "2-00": label 2, template 0


def fill(template: str, rng: random.Random) -> str:
    return template.format(amt=rng.randint(50, 5000), days=rng.randint(1, 90))


def generate(per_class: int = 700, seed: int = 42) -> list[Sample]:
    rng = random.Random(seed)
    samples = []
    for label, templates in TEMPLATES.items():
        for _ in range(per_class):
            idx = rng.randrange(len(templates))
            samples.append(Sample(fill(templates[idx], rng), label, f"{label}-{idx:02d}"))
    rng.shuffle(samples)
    return samples


def split_by_template(samples: list[Sample], test_templates_per_class: int = 3,
                      seed: int = 42) -> tuple[list[Sample], list[Sample]]:
    """Hold out whole templates, the same number per class, for the test set."""
    rng = random.Random(seed)
    held_out = set()
    for label, templates in TEMPLATES.items():
        for idx in rng.sample(range(len(templates)), test_templates_per_class):
            held_out.add(f"{label}-{idx:02d}")
    train = [s for s in samples if s.template_id not in held_out]
    test = [s for s in samples if s.template_id in held_out]
    return train, test


def split_random(samples: list[Sample], test_frac: float = 0.2,
                 seed: int = 42) -> tuple[list[Sample], list[Sample]]:
    """Sample-level random split, kept for comparison with the template split."""
    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    n_test = int(len(shuffled) * test_frac)
    return shuffled[n_test:], shuffled[:n_test]


# Hand-written messages in styles absent from the templates: SMS shorthand,
# negation, mixed signals, bank-style phrasing. Labels are my judgement,
# the set is small (24), and it is meant as a smoke test for robustness,
# not a benchmark.
CHALLENGE = [
    ("Rs.12,450 credited to a/c XX4471 on 03-Mar. Avl bal Rs.58,210.", 0),
    ("Your FD of Rs.1,00,000 has matured and been renewed.", 0),
    ("Congrats! You are pre-approved for a personal loan.", 0),
    ("Payment received. Thank you for banking with us.", 0),
    ("Your card statement is ready. Total due: Rs.0.", 0),
    ("No overdue amount on your loan account.", 0),
    ("Your insurance premium auto-pay is set for 5th.", 0),
    ("Minimum due of Rs.2,300 on your card payable by 18th.", 1),
    ("Avl bal Rs.340 in a/c XX4471. Maintain MAB to avoid charges.", 1),
    ("Reminder: loan instalment due day after tomorrow.", 1),
    ("Your card is nearing its credit limit (92% used).", 1),
    ("We noticed a missed SIP instalment this month.", 1),
    ("Your standing instruction could not be verified; please update mandate.", 1),
    ("Utility bill of Rs.4,120 due on 21st.", 1),
    ("NACH mandate bounced. Penal charges of Rs.590 levied.", 2),
    ("Legal notice: loan a/c classified as NPA. Contact branch immediately.", 2),
    ("Cheque no. 004512 returned unpaid - funds insufficient.", 2),
    ("Your card has been blocked due to non-payment of dues.", 2),
    ("3 EMIs pending. Recovery agent visit scheduled.", 2),
    ("Final reminder before your account is sent to collections.", 2),
    ("Your overdraft limit has been exceeded and interest is accruing daily.", 2),
    ("Settlement offer: pay 60% of outstanding to close the account.", 2),
    ("Your CIBIL score fell by 85 points after the default was reported.", 2),
    ("Salary not credited this month; EMI of Rs.18,000 due tomorrow.", 2),
]
