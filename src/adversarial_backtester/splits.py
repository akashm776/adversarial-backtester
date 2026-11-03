"""Expanding training sets, conservative label purging, and a pre-test gap."""

import numpy as np
from numpy.typing import NDArray

from ._validation import integer

Split = tuple[NDArray[np.int64], NDArray[np.int64]]


def purged_walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    embargo: int = 5,
    *,
    label_horizon: int = 1,
    min_train_size: int | None = None,
) -> list[Split]:
    """Require max(train) + label_horizon + embargo < min(test).

    `min_train_size` counts actual training rows in the first fold. Tests tile
    the remaining suffix, including any remainder. The embargo is a pre-test
    gap, not the post-test embargo used in combinatorial cross-validation.
    """
    integer(n_samples, "n_samples")
    integer(n_splits, "n_splits")
    integer(embargo, "embargo", 0)
    integer(label_horizon, "label_horizon")
    if min_train_size is None:
        min_train_size = max(2, n_samples // (n_splits + 1))
    integer(min_train_size, "min_train_size", 2)
    gap = label_horizon + embargo
    first_test = min_train_size + gap
    if n_samples - first_test < n_splits:
        raise ValueError("not enough observations for training, purge, embargo and test folds")
    return [
        (np.arange(int(test[0]) - gap, dtype=np.int64), test)
        for test in np.array_split(np.arange(first_test, n_samples, dtype=np.int64), n_splits)
    ]
