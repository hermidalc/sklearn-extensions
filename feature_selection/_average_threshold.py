from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from ._base import ExtendedSelectorMixin
from ..utils.validation import check_is_fitted


class MeanThreshold(ExtendedSelectorMixin, BaseEstimator):
    """Mean threshold feature selector

    Parameters
    ----------
    threshold : float (default = 0)
        Features with a mean lower than this threshold will be
        removed.

    Attributes
    ----------
    means_ : array, shape (n_features,)
        Feature means.
    """

    def __init__(self, threshold=0.):
        self.threshold = threshold

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        feature_meta : pandas.DataFrame, pandas.Series \
            shape = (n_features, n_metadata)
            Feature metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=None)
        self.means_ = X.mean(axis=0)
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.means_ >= self.threshold


class MedianThreshold(ExtendedSelectorMixin, BaseEstimator):
    """Median threshold feature selector

    Parameters
    ----------
    threshold : float (default = 0)
        Features with a median lower than this threshold will be
        removed.

    Attributes
    ----------
    medians_ : array, shape (n_features,)
        Feature medians.
    """

    def __init__(self, threshold=0.):
        self.threshold = threshold

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        feature_meta : pandas.DataFrame, pandas.Series \
            shape = (n_features, n_metadata)
            Feature metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=None)
        self.medians_ = X.median(axis=0)
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.medians_ >= self.threshold
