import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debt_stress import data  # noqa: E402


def test_generate_is_balanced_and_deterministic():
    a, b = data.generate(seed=1), data.generate(seed=1)
    assert a == b
    counts = {lbl: sum(s.label == lbl for s in a) for lbl in (0, 1, 2)}
    assert counts == {0: 700, 1: 700, 2: 700}


def test_placeholders_are_filled_with_sane_ranges():
    for s in data.generate(per_class=200):
        assert "{" not in s.text
        m = re.search(r"overdue by (\d+) days", s.text)
        if m:
            assert 1 <= int(m.group(1)) <= 90


def test_template_split_has_no_template_or_sentence_overlap():
    train, test = data.split_by_template(data.generate())
    assert {s.template_id for s in train}.isdisjoint({s.template_id for s in test})
    # sentences can only coincide if two templates render identically; they don't
    assert {s.text for s in train}.isdisjoint({s.text for s in test})
    for label in (0, 1, 2):
        assert len({s.template_id for s in test if s.label == label}) == 3


def test_random_split_shares_sentences_across_sets():
    """Why evaluation uses the template split: a sample-level split repeats sentences."""
    train, test = data.split_random(data.generate())
    seen = {s.text for s in train}
    assert sum(s.text in seen for s in test) / len(test) > 0.6


def test_challenge_set_is_labelled_and_covers_all_classes():
    labels = [y for _, y in data.CHALLENGE]
    assert set(labels) == {0, 1, 2}
    assert len({t for t, _ in data.CHALLENGE}) == len(data.CHALLENGE)
