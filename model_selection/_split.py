# Authors: Leandro Hermida <hermidal@cs.umd.edu>
#
# License: BSD 3 clause

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits
from sklearn.utils.validation import check_array, check_random_state, indexable


class StratifiedGroupKFold(_BaseKFold):
    """Stratified K-Folds iterator variant with non-overlapping groups.

    This cross-validation object is a variation of StratifiedKFold that returns
    stratified folds with non-overlapping groups. The folds are made by
    preserving the percentage of samples for each class.

    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).

    The difference between GroupKFold and StratifiedGroupKFold is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    StratifiedGroupKFold attempts to create folds which preserve the
    percentage of samples for each class.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> cv = StratifiedGroupKFold(n_splits=3)
    >>> for train_idxs, test_idxs in cv.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idxs])
    ...     print("      ", y[train_idxs])
    ...     print(" TEST:", groups[test_idxs])
    ...     print("      ", y[test_idxs])
    TRAIN: [2 2 4 5 5 5 5 6 6 7]
           [1 1 1 0 0 0 0 0 0 0]
     TEST: [1 1 3 3 3 8 8]
           [0 0 1 1 1 0 0]
    TRAIN: [1 1 3 3 3 4 5 5 5 5 8 8]
           [0 0 1 1 1 1 0 0 0 0 0 0]
     TEST: [2 2 6 6 7]
           [1 1 0 0 0]
    TRAIN: [1 1 2 2 3 3 3 6 6 7 8 8]
           [0 0 1 1 1 1 1 0 0 0 0 0]
     TEST: [4 5 5 5 5]
           [1 0 0 0 0]

    See also
    --------
    StratifiedKFold: Takes class information into account to build folds which
        retain class distributions (for binary or multiclass classification
        tasks).

    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    # Implementation based on this kaggle kernel:
    # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def _iter_test_indices(self, X, y, groups):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, group in zip(y, groups):
            y_counts_per_group[group][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        groups_and_y_counts = list(y_counts_per_group.items())
        rng = check_random_state(self.random_state)
        if self.shuffle:
            rng.shuffle(groups_and_y_counts)

        for group, y_counts in sorted(groups_and_y_counts,
                                      key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                y_counts_per_fold[i] += y_counts
                std_per_label = []
                for label in range(labels_num):
                    std_per_label.append(np.std(
                        [y_counts_per_fold[j][label] / y_distr[label]
                         for j in range(self.n_splits)]))
                y_counts_per_fold[i] -= y_counts
                fold_eval = np.mean(std_per_label)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(group)

        for i in range(self.n_splits):
            test_indices = [idx for idx, group in enumerate(groups)
                            if group in groups_per_fold[i]]
            yield test_indices


class RepeatedStratifiedGroupKFold(_RepeatedSplits):
    """Repeated Stratified Group K-Fold cross validator.

    Repeats Stratified Group K-Fold with non-overlapping groups n times with
    different randomization in each repetition.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.

    random_state : int or RandomState instance, default=None
        Controls the generation of the random states for each repetition.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedStratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> cv = RepeatedStratifiedGroupKFold(n_splits=2, n_repeats=2,
    ...                                   random_state=36851234)
    >>> for train_index, test_index in cv.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idxs])
    ...     print("      ", y[train_idxs])
    ...     print(" TEST:", groups[test_idxs])
    ...     print("      ", y[test_idxs])
    TRAIN: [2 2 4 5 5 5 5 8 8]
           [1 1 1 0 0 0 0 0 0]
     TEST: [1 1 3 3 3 6 6 7]
           [0 0 1 1 1 0 0 0]
    TRAIN: [1 1 3 3 3 6 6 7]
           [0 0 1 1 1 0 0 0]
     TEST: [2 2 4 5 5 5 5 8 8]
           [1 1 1 0 0 0 0 0 0]
    TRAIN: [3 3 3 4 7 8 8]
           [1 1 1 1 0 0 0]
     TEST: [1 1 2 2 5 5 5 5 6 6]
           [0 0 1 1 0 0 0 0 0 0]
    TRAIN: [1 1 2 2 5 5 5 5 6 6]
           [0 0 1 1 0 0 0 0 0 0]
     TEST: [3 3 3 4 7 8 8]
           [1 1 1 1 0 0 0]

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.

    See also
    --------
    RepeatedStratifiedKFold: Repeats Stratified K-Fold n times.
    """

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(StratifiedGroupKFold, n_splits=n_splits,
                         n_repeats=n_repeats, random_state=random_state)


class StratifiedSampleFromGroupKFold(StratifiedKFold):

    def split(self, X, y, groups):
        X, y, groups = indexable(X, y, groups)
        indices = pd.DataFrame(groups).groupby(0).apply(
            pd.DataFrame.sample, n=1, random_state=self.random_state
            ).reset_index(level=0, drop=True).sort_index().index.values
        X = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        y = y[indices]
        for test_mask in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_mask)]
            test_index = indices[test_mask]
            yield train_index, test_index


class RepeatedStratifiedSampleFromGroupKFold(_RepeatedSplits):

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(StratifiedSampleFromGroupKFold, n_splits=n_splits,
                         n_repeats=n_repeats, random_state=random_state)


class StratifiedGroupShuffleSplit(StratifiedShuffleSplit):
    """Stratified GroupShuffleSplit cross-validator

    Provides randomized train/test indices to split data according to a
    third-party provided group. This group information can be used to encode
    arbitrary domain specific stratifications of the samples as integers.

    This cross-validation object is a merge of GroupShuffleSplit and
    StratifiedShuffleSplit, which returns randomized folds stratified by group
    class. The folds are made by preserving the percentage of groups for each
    class.

    Note: like the StratifiedShuffleSplit strategy, stratified random group
    splits do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of re-shuffling & splitting iterations.

    test_size : float, int, None, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the test split (rounded up). If int,
        represents the absolute number of test groups. If None, the value is
        set to the complement of the train size. By default, the value is set
        to 0.1.

    train_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the groups to include in the train split. If
        int, represents the absolute number of train groups. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupShuffleSplit
    >>> X = np.ones(shape=(15, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6])
    >>> print(groups.shape)
    (15,)
    >>> sgss = StratifiedGroupShuffleSplit(n_splits=3, train_size=.7,
    ...                                    random_state=43)
    >>> sgss.get_n_splits()
    3
    >>> for train_idx, test_idx in sgss.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idx])
    ...     print("      ", y[train_idx])
    ...     print(" TEST:", groups[test_idx])
    ...     print("      ", y[test_idx])
    TRAIN: [2 2 2 4 5 5 5 5 6 6]
           [1 1 1 0 1 1 1 1 0 0]
     TEST: [1 1 3 3 3]
           [0 0 1 1 1]
    TRAIN: [1 1 2 2 2 3 3 3 4]
           [0 0 1 1 1 1 1 1 0]
     TEST: [5 5 5 5 6 6]
           [1 1 1 1 0 0]
    TRAIN: [1 1 2 2 2 3 3 3 6 6]
           [0 0 1 1 1 1 1 1 0 0]
     TEST: [4 5 5 5 5]
           [0 1 1 1 1]

    See also
    --------
    GroupShuffleSplit: Shuffle-Group(s)-Out iterator.

    StratifiedShuffleSplit: Stratified ShuffleSplit iterator.
    """

    def __init__(self, n_splits=5, test_size=None, train_size=None,
                 random_state=None):
        super().__init__(n_splits=n_splits, test_size=test_size,
                         train_size=train_size, random_state=random_state)
        self._default_test_size = 0.1

    def _iter_indices(self, X, y, groups):
        y = check_array(y, ensure_2d=False, dtype=None)
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)
        (unique_groups, unique_groups_y), group_indices = np.unique(
            np.stack((groups, y)), axis=1, return_inverse=True)
        if unique_groups.shape[0] != np.unique(groups).shape[0]:
            raise ValueError("Members of each group must all be of the same "
                             "class.")
        for group_train, group_test in super()._iter_indices(
                X=unique_groups, y=unique_groups_y):
            # these are the indices of unique_groups in the partition invert
            # them into data indices
            train = np.flatnonzero(np.in1d(group_indices, group_train))
            test = np.flatnonzero(np.in1d(group_indices, group_test))
            yield train, test
