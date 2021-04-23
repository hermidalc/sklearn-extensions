from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from ._base import ExtendedSelectorMixin
from ..utils.validation import check_is_fitted


class ConfidenceThreshold(ExtendedSelectorMixin, BaseEstimator):
    """Confidence score threshold feature selector

    Parameters
    ----------
    threshold : float (default = 0.95)
        Features with a confidence score lower than this threshold will be
        removed.

    meta_col : str (default = "Confidence Score")
        Feature metadata column name with confidence scores.

    Attributes
    ----------
    confidence_scores_ : array, shape (n_features,)
        Feature confidence scores.
    """

    def __init__(self, threshold=0.95, meta_col='Confidence Score'):
        self.threshold = threshold
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
        self.confidence_scores_ = feature_meta[self.meta_col].to_numpy()
        return self

    def _check_params(self, X, y, feature_meta):
        if X.shape[1] != feature_meta.shape[0]:
            raise ValueError('X ({:d}) and feature_meta ({:d}) have '
                             'different feature dimensions'
                             .format(X.shape[1], feature_meta.shape[0]))
        if self.meta_col not in feature_meta.columns:
            raise ValueError('{} feature_meta column does not exist.'
                             .format(self.meta_col))

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.confidence_scores_ >= self.threshold
