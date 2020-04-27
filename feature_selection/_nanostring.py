import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from ._base import ExtendedSelectorMixin
from ..utils.validation import check_is_fitted


class NanoStringEndogenousSelector(ExtendedSelectorMixin, BaseEstimator):
    """NanoString Endogenous feature selector.

    Parameters
    ----------
    filter_empty : bool (default = True)
        Whether to also filter endogenous features with all zero counts.

    meta_col : str (default = "Code.Class")
        Feature metadata column name holding Code Class information.
    """

    def __init__(self, filter_empty=True, meta_col='Code.Class'):
        self.filter_empty = filter_empty
        self.meta_col = meta_col

    def fit(self, X, y, feature_meta):
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
        self._check_params(X, y, feature_meta)
        mask = feature_meta[self.meta_col].isin(['Endogenous']).to_numpy()
        if self.filter_empty:
            mask &= np.any(X, axis=0)
        self.mask_ = mask
        return self

    def transform(self, X, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input data matrix.

        feature_meta : Ignored.

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            Input data matrix with only endogenous features.
        """
        check_is_fitted(self)
        return super().transform(X)

    def inverse_transform(self, X, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        feature_meta : Ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        check_is_fitted(self)
        return super().inverse_transform(X)

    def _check_params(self, X, y, feature_meta):
        if X.shape[1] != feature_meta.shape[0]:
            raise ValueError('X ({:d}) and feature_meta ({:d}) have '
                             'different feature dimensions'
                             .format(X.shape[1], feature_meta.shape[0]))
        if self.meta_col not in feature_meta.columns:
            raise ValueError('{} feature_meta column does not exist.'
                             .format(self.meta_col))
        if not feature_meta[self.meta_col].isin(['Endogenous']).any():
            raise ValueError('{} feature_meta column does not have any '
                             'Endogenous features'.format(self.meta_col))

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.mask_
