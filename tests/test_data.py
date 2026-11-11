
import numpy as np

from data import stratified_split


def test_split_sizes():
    labels = np.repeat(np.arange(100), 600)

    train, val, test = stratified_split(labels)

    assert len(train) == 48000
    assert len(val) == 6000
    assert len(test) == 6000


def test_no_overlap():
    labels = np.repeat(np.arange(100), 600)

    train, val, test = stratified_split(labels)

    assert not np.intersect1d(train, val).size
    assert not np.intersect1d(train, test).size
    assert not np.intersect1d(val, test).size

    combined = np.concatenate([train, val, test])
    assert len(np.unique(combined)) == 60000


def test_reproducibility():
    labels = np.repeat(np.arange(100), 600)

    split_a = stratified_split(labels, seed=42)
    split_b = stratified_split(labels, seed=42)

    for a, b in zip(split_a, split_b):
        assert np.array_equal(a, b)


def test_class_balance():
    labels = np.repeat(np.arange(100), 600)

    train, val, test = stratified_split(labels)

    for class_id in range(100):
        assert np.sum(labels[train] == class_id) == 480
        assert np.sum(labels[val] == class_id) == 60
        assert np.sum(labels[test] == class_id) == 60
