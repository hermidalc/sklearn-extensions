import os
import warnings
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.rinterface import RRuntimeWarning
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_memory
from .base import SelectorMixin

numpy2ri.deactivate()
pandas2ri.deactivate()
numpy2ri.activate()
pandas2ri.activate()

if 'cfs_feature_idxs' not in robjects.globalenv:
    r_base = importr('base')
    r_base.source(os.path.dirname(__file__) + '/multivariate.R')
r_cfs_feature_idxs = robjects.globalenv['cfs_feature_idxs']
r_fcbf_feature_idxs = robjects.globalenv['fcbf_feature_idxs']
r_relieff_feature_score = robjects.globalenv['relieff_feature_score']


def fcbf_feature_idxs(X, y, threshold):
    idxs, scores = r_fcbf_feature_idxs(X, y, threshold=threshold)
    return np.array(idxs, dtype=int), np.array(scores, dtype=float)


def relieff_feature_score(X, y, num_neighbors, sample_size):
    return np.array(r_relieff_feature_score(
        X, y, num_neighbors=num_neighbors, sample_size=sample_size),
                    dtype=float)


class CFS(BaseEstimator, SelectorMixin):
    """Feature selector using Correlation Feature Selection (CFS) algorithm

    Attributes
    ----------
    selected_idxs_ : array-like, 1d
        CFS selected feature indexes
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training input data matrix.

        y : array-like, shape = (n_samples)
            Target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=None)
        self._n_features = X.shape[1]
        warnings.filterwarnings('ignore', category=RRuntimeWarning,
                                message='^Rjava\.init\.warning')
        self.selected_idxs_ = np.array(r_cfs_feature_idxs(X, y), dtype=int)
        warnings.filterwarnings('always', category=RRuntimeWarning)
        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'selected_idxs_')
        mask = np.zeros(self._n_features, dtype=bool)
        mask[self.selected_idxs_] = True
        return mask


class FCBF(BaseEstimator, SelectorMixin):
    """Feature selector using Fast Correlation-Based Filter (FCBF) algorithm

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. The "all" option bypasses selection,
        for use in a parameter search. If k is specified threshold is ignored.

    threshold : float (default = 0)
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    selected_idxs_ : array-like, 1d
        FCBF selected feature indexes
    """

    def __init__(self, k='all', threshold=0, memory=None):
        self.k = k
        self.threshold = threshold
        self.memory = memory
        self.selected_idxs_ = np.array([], dtype=int)
        self.scores_ = np.array([], dtype=float)

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training input data matrix.

        y : array-like, shape = (n_samples)
            Target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=None)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        self._n_features = X.shape[1]
        if self.k == 'all' or self.k > 0:
            warnings.filterwarnings('ignore', category=RRuntimeWarning,
                                    message='^Rjava\.init\.warning')
            feature_idxs, scores = memory.cache(fcbf_feature_idxs)(
                X, y, threshold=self.threshold)
            warnings.filterwarnings('always', category=RRuntimeWarning)
            if self.k != 'all':
                feature_idxs = feature_idxs[
                    np.argsort(scores, kind='mergesort')[-self.k:]]
                scores = np.sort(scores, kind='mergesort')[-self.k:]
            self.selected_idxs_ = np.sort(feature_idxs, kind='mergesort')
            self.scores_ = scores[np.argsort(feature_idxs, kind='mergesort')]
        return self

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features."
                % self.k)

    def _get_support_mask(self):
        check_is_fitted(self, 'selected_idxs_')
        mask = np.zeros(self._n_features, dtype=bool)
        if self.k == 'all' or self.k > 0:
            mask[self.selected_idxs_] = True
        return mask


class ReliefF(BaseEstimator, SelectorMixin):
    """Feature selector using ReliefF algorithm

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. The "all" option bypasses selection,
        for use in a parameter search.

    n_neighbors : int (default = 10)
        Number of neighbors for ReliefF algorithm

    sample_size : int (default = 5)
        Sample size for ReliefF algorithm

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    scores_ : array-like, shape=(n_features,)
        Feature scores
    """

    def __init__(self, k='all', n_neighbors=10, sample_size=5, memory=None):
        self.k = k
        self.n_neighbors = n_neighbors
        self.sample_size = sample_size
        self.memory = memory

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training input data matrix.

        y : array-like, shape = (n_samples)
            Target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=None)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        warnings.filterwarnings('ignore', category=RRuntimeWarning,
                                message='^Rjava\.init\.warning')
        self.scores_ = memory.cache(relieff_feature_score)(X, y)
        warnings.filterwarnings('always', category=RRuntimeWarning)
        return self

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features."
                % self.k)

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.k == 'all':
            mask = np.ones_like(self.scores_, dtype=bool)
        elif self.k > 0:
            mask[np.argsort(self.scores_, kind='mergesort')[-self.k:]] = True
        return mask
