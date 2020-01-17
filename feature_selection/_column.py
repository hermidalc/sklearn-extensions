import warnings
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from ._base import ExtendedSelectorMixin


class ColumnSelectorWarning(UserWarning):
    """Warning used to notify when column name does not exist
    """


class ColumnSelector(ExtendedSelectorMixin, BaseEstimator):
    """Column feature selector

    Parameters
    ----------
    cols : array-like (default = None)
        A list specifying the feature indices to be selected. For example,
        [1, 4, 5] to select the 2nd, 5th, and 6th feature columns, and
        ['A','C','D'] to select the name of feature columns A, C and D.
        If None, returns all columns in the array.

    meta_col : str (default = None)
        Feature metadata column name to use instead of feature names when cols
        is a list of names.
    """

    def __init__(self, cols=None, meta_col=None):
        self.cols = cols
        self.meta_col = meta_col

    def fit(self, X, y, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training input data matrix.

        y : array-like, shape = (n_samples)
            Target values (class labels in classification, real numbers in
            regression).

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata.

        Returns
        ---------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=None)
        self._check_params(X, y, feature_meta)
        if self.cols is None:
            mask = np.ones(X.shape[1], dtype=bool)
        elif isinstance(self.cols[0], str):
            if self.meta_col:
                mask = feature_meta.isin(
                    {self.meta_col: self.cols})[self.meta_col].to_numpy()
            else:
                mask = feature_meta.index.isin(self.cols)
        else:
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[list(self.cols)] = True
        self._mask = mask
        return self

    def transform(self, X, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input data matrix.

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata.

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            edgeR filterByExpr counts data matrix with only the selected
            features.
        """
        check_is_fitted(self, '_mask')
        return super().transform(X, feature_meta)

    def inverse_transform(self, X, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        check_is_fitted(self, '_mask')
        return super().inverse_transform(X, feature_meta)

    def _check_params(self, X, y, feature_meta):
        if self.cols:
            types = {type(i) for i in self.cols}
            if len(types) > 1:
                raise ValueError('cols should be all names or indices.')
            if isinstance(self.cols[0], str):
                if feature_meta is None:
                    raise ValueError('feature_meta must be passed if cols are '
                                     'names')
                if self.meta_col is None:
                    for col in self.cols:
                        if col not in feature_meta.index:
                            warnings.warn('%s does not exist.' % col,
                                          ColumnSelectorWarning)
                elif self.meta_col not in feature_meta.columns:
                    raise ValueError('%s feature_meta column does not exist.'
                                     % self.meta_col)
            else:
                for col in self.cols:
                    if not 0 <= col <= X.shape[1]:
                        raise ValueError(
                            'cols should be 0 <= col <= n_features; got %r.'
                            'Use cols=None to return all features.' % col)

    def _get_support_mask(self):
        check_is_fitted(self, '_mask')
        return self._mask
