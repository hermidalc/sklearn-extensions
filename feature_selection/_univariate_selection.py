"""Univariate features selection."""

# Authors: V. Michel, B. Thirion, G. Varoquaux, A. Gramfort, E. Duchesnay.
#          L. Buitinck, A. Joly, L. Hermida
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.feature_selection import chi2, f_classif, f_regression
from sklearn.feature_selection import (
    SelectFdr as SklearnSelectFdr,
    SelectFpr as SklearnSelectFpr,
    SelectFwe as SklearnSelectFwe,
    SelectKBest as SklearnSelectKBest,
    SelectPercentile as SklearnSelectPercentile,
    GenericUnivariateSelect as SklearnGenericUnivariateSelect,
)
from sklearn.feature_selection._univariate_selection import (
    _BaseFilter as _SklearnBaseFilter,
)

from ..feature_selection import ExtendedSelectorMixin


class BaseScorer(BaseEstimator):
    """Base univariate feature scorer."""


class ANOVAFScorerClassification(BaseScorer):
    """Compute the ANOVA F-value for the provided sample.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The set of regressors that will be tested sequentially.

    y : ndarray of shape (n_samples,)
        The target vector.

    Returns
    -------
    f_statistic : ndarray of shape (n_features,)
        F-statistic for each feature.

    p_values : ndarray of shape (n_features,)
        P-values associated with the F-statistic.
    """

    def fit(self, X, y):
        """Run feature scorer on (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} shape = [n_samples, n_features]
            The set of regressors that will be tested sequentially.

        y : array of shape (n_samples,)
            Target vector.

        Returns
        -------
        self : object
            Returns self.
        """
        self.scores_, self.pvalues_ = f_classif(X, y)
        return self


class ANOVAFScorerRegression(BaseScorer):
    """Univariate linear regression tests returning F-statistic and p-values.

    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors.

    This is done in 2 steps:

    1. The cross correlation between each regressor and the target is computed
       using :func:`r_regression` as::

           E[(X[:, i] - mean(X[:, i])) * (y - mean(y))] / (std(X[:, i]) * std(y))

    2. It is converted to an F score and then to a p-value.

    :func:`f_regression` is derived from :func:`r_regression` and will rank
    features in the same order if all the features are positively correlated
    with the target.

    Note however that contrary to :func:`f_regression`, :func:`r_regression`
    values lie in [-1, 1] and can thus be negative. :func:`f_regression` is
    therefore recommended as a feature selection criterion to identify
    potentially predictive feature for a downstream classifier, irrespective of
    the sign of the association with the target variable.

    Furthermore :func:`f_regression` returns p-values while
    :func:`r_regression` does not.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data matrix.

    y : array-like of shape (n_samples,)
        The target vector.

    center : bool, default=True
        Whether or not to center the data matrix `X` and the target vector `y`.
        By default, `X` and `y` will be centered.

    force_finite : bool, default=True
        Whether or not to force the F-statistics and associated p-values to
        be finite. There are two cases where the F-statistic is expected to not
        be finite:

        - when the target `y` or some features in `X` are constant. In this
          case, the Pearson's R correlation is not defined leading to obtain
          `np.nan` values in the F-statistic and p-value. When
          `force_finite=True`, the F-statistic is set to `0.0` and the
          associated p-value is set to `1.0`.
        - when the a feature in `X` is perfectly correlated (or
          anti-correlated) with the target `y`. In this case, the F-statistic
          is expected to be `np.inf`. When `force_finite=True`, the F-statistic
          is set to `np.finfo(dtype).max` and the associated p-value is set to
          `0.0`.

        .. versionadded:: 1.1
    """

    def __init__(self, center=True, force_finite=True):
        self.center = center
        self.force_finite = force_finite

    def fit(self, X, y):
        """Run feature scorer on (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix}  shape = (n_samples, n_features)
            The set of regressors that will be tested sequentially.

        y : array of shape (n_samples,).
            Target vector.

        Returns
        -------
        self : object
            Returns self.
        """
        self.scores_, self.pvalues_ = f_regression(
            X, y, center=self.center, force_finite=self.force_finite
        )
        return self


class Chi2Scorer(BaseScorer):
    """Compute chi-squared stats between each non-negative feature and class.

    This score can be used to select the n_features features with the
    highest values for the test chi-squared statistic from X, which must
    contain only non-negative features such as booleans or frequencies
    (e.g., term counts in document classification), relative to the classes.

    Recall that the chi-square test measures dependence between stochastic
    variables, so using this function "weeds out" the features that are the
    most likely to be independent of class and therefore irrelevant for
    classification.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Attributes
    ----------
    scores_ : array, shape = (n_features,)
        chi2 statistics of each feature.

    pvalues_ : array, shape = (n_features,)
        p-values of each feature.

    Notes
    -----
    Complexity of this algorithm is O(n_classes * n_features).

    See also
    --------
    ANOVAFScorerClassification: ANOVA F-value between label/feature for \
    classification tasks.
    ANOVAFScorerRegression: ANOVA F-value between label/feature for \
    regression tasks.
    MutualInfoScorerClassification: Mutual information for a discrete target.
    MutualInfoScorerRegression: Mutual information for a continuous target.
    """

    def fit(self, X, y):
        """Run feature scorer on (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features_in)
            Sample vectors.

        y : array-like, shape = (n_samples,)
            Target vector (class labels).

        Returns
        -------
        self : object
            Returns self.
        """
        self.scores_, self.pvalues_ = chi2(X, y)
        return self


class _BaseFilter(ExtendedSelectorMixin, _SklearnBaseFilter):
    def _check_params(self, X, y):
        if not callable(self.score_func) and not isinstance(
            self.score_func, BaseScorer
        ):
            raise TypeError(
                "The score function should be a callable or"
                "scorer object, %s (%s) was passed."
                % (self.score_func, type(self.score_func))
            )


class SelectPercentile(_BaseFilter, SklearnSelectPercentile):
    pass


class SelectKBest(_BaseFilter, SklearnSelectKBest):
    pass


class SelectFpr(_BaseFilter, SklearnSelectFpr):
    pass


class SelectFdr(_BaseFilter, SklearnSelectFdr):
    pass


class SelectFwe(_BaseFilter, SklearnSelectFwe):
    pass


class GenericUnivariateSelect(_BaseFilter, SklearnGenericUnivariateSelect):
    pass
