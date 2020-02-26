"""Univariate features selection."""

# Authors: V. Michel, B. Thirion, G. Varoquaux, A. Gramfort, E. Duchesnay.
#          L. Buitinck, A. Joly, L. Hermida
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import as_float_array, check_X_y
from sklearn.feature_selection import (
    chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression)
from ._base import ExtendedSelectorMixin
from ..utils.validation import check_is_fitted


def _clean_nans(scores):
    """
    Fixes Issue #1240: NaNs can't be properly compared, so change them to the
    smallest value of scores's dtype. -inf seems to be unreliable.
    """
    # XXX where should this function be called? fit? scoring functions
    # themselves?
    scores = as_float_array(scores, copy=True)
    scores[np.isnan(scores)] = np.finfo(scores.dtype).min
    return scores


######################################################################
# Base scorer class

class BaseScorer(BaseEstimator):
    """Base univariate feature scorer."""


######################################################################
# Specific scorer classes

class ANOVAFScorerClassification(BaseScorer):
    """Compute the ANOVA F-value for the provided sample.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Attributes
    ----------
    scores_ : array, shape = [n_features,]
        The set of F values.

    pvalues_ : array, shape = [n_features,]
        The set of p-values.

    See also
    --------
    ANOVAFScorerRegression: ANOVA F-value between label/feature for \
    regression tasks.
    Chi2Scorer: Chi-squared stats of non-negative features for classification \
    tasks.
    MutualInfoScorerClassification: Mutual information for a discrete target.
    MutualInfoScorerRegression: Mutual information for a continuous target.
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
    """Univariate linear regression tests.

    Linear model for testing the individual effect of each of many regressors.
    This is a scoring function to be used in a feature seletion procedure, not
    a free standing feature selection procedure.

    This is done in 2 steps:

    1. The correlation between each regressor and the target is computed,
       that is, ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) *
       std(y)).
    2. It is converted to an F score then to a p-value.

    For more on usage see the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    center : boolean, default=True
        If true, X and y will be centered.

    Attributes
    ----------
    scores_ : array, shape=(n_features,)
        The set of F values.

    pvalues_ : array, shape=(n_features,)
        The set of p-values.

    See also
    --------
    ANOVAFScorerClassification: ANOVA F-value between label/feature for \
    classification tasks.
    Chi2Scorer: Chi-squared stats of non-negative features for classification \
    tasks.
    MutualInfoScorerClassification: Mutual information for a discrete target.
    MutualInfoScorerRegression: Mutual information for a continuous target.
    """

    def __init__(self, center=True):
        self.center = center

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
        self.scores_, self.pvalues_ = f_regression(X, y, self.center)
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


class MutualInfoScorerClassification(BaseScorer):
    """Estimate mutual information for a discrete target variable.

    Mutual information (MI) [1]_ between two random variables is a non-negative
    value, which measures the dependency between the variables. It is equal
    to zero if and only if two random variables are independent, and higher
    values mean higher dependency.

    The function relies on nonparametric methods based on entropy estimation
    from k-nearest neighbors distances as described in [2]_ and [3]_. Both
    methods are based on the idea originally proposed in [4]_.

    It can be used for univariate features selection, read more in the
    :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    discrete_features : {'auto', bool, array_like}, default 'auto'
        If bool, then determines whether to consider all features discrete
        or continuous. If array, then it should be either a boolean mask
        with shape (n_features,) or array with indices of discrete features.
        If 'auto', it is assigned to False for dense `X` and to True for
        sparse `X`.

    n_neighbors : int, default 3
        Number of neighbors to use for MI estimation for continuous variables,
        see [2]_ and [3]_. Higher values reduce variance of the estimation, but
        could introduce a bias.

    copy : bool, default True
        Whether to make a copy of the given data. If set to False, the initial
        data will be overwritten.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator for adding small noise
        to continuous variables in order to remove repeated values.  If int,
        random_state is the seed used by the random number generator; If
        RandomState instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance used by
        `np.random`.

    Attributes
    ----------
    scores_ : ndarray, shape (n_features,)
        Estimated mutual information between each feature and the target.

    Notes
    -----
    1. The term "discrete features" is used instead of naming them
       "categorical", because it describes the essence more accurately.
       For example, pixel intensities of an image are discrete features
       (but hardly categorical) and you will get better results if mark them
       as such. Also note, that treating a continuous variable as discrete and
       vice versa will usually give incorrect results, so be attentive about
       that.
    2. True mutual information can't be negative. If its estimate turns out
       to be negative, it is replaced by zero.

    References
    ----------
    .. [1] `Mutual Information <https://en.wikipedia.org/wiki/Mutual_information>`_
           on Wikipedia.
    .. [2] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [3] B. C. Ross "Mutual Information between Discrete and Continuous
           Data Sets". PLoS ONE 9(2), 2014.
    .. [4] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
           of a Random Vector:, Probl. Peredachi Inf., 23:2 (1987), 9-16

    See also
    --------
    ANOVAFScorerClassification: ANOVA F-value between label/feature for \
    classification tasks.
    ANOVAFScorerRegression: ANOVA F-value between label/feature for \
    regression tasks.
    Chi2Scorer: Chi-squared stats of non-negative features for classification \
    tasks.
    MutualInfoScorerRegression: Mutual information for a continuous target.
    """

    def __init__(self, discrete_features='auto', n_neighbors=3,
                 copy=True, random_state=None):
        self.discrete_features = discrete_features
        self.n_neighbors = n_neighbors
        self.copy = copy
        self.random_state = random_state

    def fit(self, X, y):
        """Run scorer on (X, y).

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Feature matrix.

        y : array_like, shape (n_samples,)
            Target vector.

        Returns
        -------
        self : object
            Returns self.
        """
        self.scores_ = mutual_info_classif(X, y, self.discrete_features,
                                           self.n_neighbors, self.copy,
                                           self.random_state)
        return self


class MutualInfoScorerRegression(BaseScorer):
    """Estimate mutual information for a continuous target variable.

    Mutual information (MI) [1]_ between two random variables is a non-negative
    value, which measures the dependency between the variables. It is equal
    to zero if and only if two random variables are independent, and higher
    values mean higher dependency.

    The function relies on nonparametric methods based on entropy estimation
    from k-nearest neighbors distances as described in [2]_ and [3]_. Both
    methods are based on the idea originally proposed in [4]_.

    It can be used for univariate features selection, read more in the
    :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    discrete_features : {'auto', bool, array_like}, default 'auto'
        If bool, then determines whether to consider all features discrete
        or continuous. If array, then it should be either a boolean mask
        with shape (n_features,) or array with indices of discrete features.
        If 'auto', it is assigned to False for dense `X` and to True for
        sparse `X`.

    n_neighbors : int, default 3
        Number of neighbors to use for MI estimation for continuous variables,
        see [2]_ and [3]_. Higher values reduce variance of the estimation, but
        could introduce a bias.

    copy : bool, default True
        Whether to make a copy of the given data. If set to False, the initial
        data will be overwritten.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator for adding small noise
        to continuous variables in order to remove repeated values.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    scores_ : ndarray, shape (n_features,)
        Estimated mutual information between each feature and the target.

    Notes
    -----
    1. The term "discrete features" is used instead of naming them
       "categorical", because it describes the essence more accurately.
       For example, pixel intensities of an image are discrete features
       (but hardly categorical) and you will get better results if mark them
       as such. Also note, that treating a continuous variable as discrete and
       vice versa will usually give incorrect results, so be attentive about
       that.
    2. True mutual information can't be negative. If its estimate turns out
       to be negative, it is replaced by zero.

    References
    ----------
    .. [1] `Mutual Information <https://en.wikipedia.org/wiki/Mutual_information>`_
           on Wikipedia.
    .. [2] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [3] B. C. Ross "Mutual Information between Discrete and Continuous
           Data Sets". PLoS ONE 9(2), 2014.
    .. [4] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
           of a Random Vector", Probl. Peredachi Inf., 23:2 (1987), 9-16

    See also
    --------
    ANOVAFScorerClassification: ANOVA F-value between label/feature for \
    classification tasks.
    ANOVAFScorerRegression: ANOVA F-value between label/feature for \
    regression tasks.
    Chi2Scorer: Chi-squared stats of non-negative features for classification \
    tasks.
    MutualInfoScorerClassification: Mutual information for a discrete target.
    """

    def __init__(self, discrete_features='auto', n_neighbors=3,
                 copy=True, random_state=None):
        self.discrete_features = discrete_features
        self.n_neighbors = n_neighbors
        self.copy = copy
        self.random_state = random_state

    def fit(self, X, y):
        """Run scorer on (X, y).

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Feature matrix.

        y : array_like, shape (n_samples,)
            Target vector.

        Returns
        -------
        self : object
            Returns self.
        """
        self.scores_ = mutual_info_regression(X, y, self.discrete_features,
                                              self.n_neighbors, self.copy,
                                              self.random_state)
        return self


######################################################################
# Base filter class

class _BaseFilter(ExtendedSelectorMixin, BaseEstimator):
    """Initialize the univariate feature selection.

    Parameters
    ----------
    score_func : callable or object
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores or a scorer object.
    """

    def __init__(self, score_func):
        self.score_func = score_func

    def fit(self, X, y):
        """Run score function on (X, y) and get the appropriate features.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, ['csr', 'csc'], multi_output=True)
        self._check_params(X, y)
        if callable(self.score_func):
            score_func_ret = self.score_func(X, y)
            if isinstance(score_func_ret, (list, tuple)):
                self.scores_, self.pvalues_ = score_func_ret
                self.pvalues_ = np.asarray(self.pvalues_)
            else:
                self.scores_ = score_func_ret
                self.pvalues_ = None
            self.scores_ = np.asarray(self.scores_)
        else:
            self.score_func.fit(X, y)
            self.scores_ = np.asarray(self.score_func.scores_)
            if hasattr(self.score_func, 'pvalues_'):
                self.pvalues_ = np.asarray(self.score_func.pvalues_)
            else:
                self.pvalues_ = None

        return self

    def _check_params(self, X, y):
        if (not callable(self.score_func)
                and not isinstance(self.score_func, BaseScorer)):
            raise TypeError("The score function should be a callable or"
                            "scorer object, %s (%s) was passed."
                            % (self.score_func, type(self.score_func)))


######################################################################
# Specific filters
######################################################################
class SelectPercentile(_BaseFilter):
    """Select features according to a percentile of the highest scores.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable or object
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores or a scorer object.
        Default is f_classif (see below "See also"). The default function only
        works with classification tasks.

    percentile : int, optional, default=10
        Percent of features to keep.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores, None if `score_func` returned only scores.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.feature_selection import SelectPercentile, chi2
    >>> X, y = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
    >>> X_new.shape
    (1797, 7)

    Notes
    -----
    Ties between features with equal scores will be broken in an unspecified
    way.

    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    mutual_info_regression: Mutual information for a continuous target.
    ANOVAFScorerClassification: F-value between label/feature for \
    classification tasks.
    ANOVAFScorerRegression: F-value between label/feature for regression tasks.
    Chi2Scorer: Chi-squared stats of non-negative features for classification \
    tasks.
    MutualInfoScorerClassification: Mutual information for a discrete target.
    MutualInfoScorerRegression: Mutual information for a continuous target.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    GenericUnivariateSelect: Univariate feature selector with configurable \
    mode.
    """

    def __init__(self, score_func=f_classif, percentile=10):
        super().__init__(score_func)
        self.percentile = percentile

    def _check_params(self, X, y):
        super()._check_params(X, y)
        if not 0 <= self.percentile <= 100:
            raise ValueError("percentile should be >=0, <=100; got %r"
                             % self.percentile)

    def _get_support_mask(self):
        check_is_fitted(self)
        # Cater for NaNs
        if self.percentile == 100:
            return np.ones(len(self.scores_), dtype=np.bool)
        elif self.percentile == 0:
            return np.zeros(len(self.scores_), dtype=np.bool)

        scores = _clean_nans(self.scores_)
        threshold = np.percentile(scores, 100 - self.percentile)
        mask = scores > threshold
        ties = np.where(scores == threshold)[0]
        if len(ties):
            max_feats = int(len(scores) * self.percentile / 100)
            kept_ties = ties[:max_feats - mask.sum()]
            mask[kept_ties] = True
        return mask


class SelectKBest(_BaseFilter):
    """Select features according to the k highest scores.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable or object
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores or a scorer object.
        Default is f_classif (see below "See also"). The default function only
        works with classification tasks.

    k : int or "all", optional, default=10
        Number of top features to select.
        The "all" option bypasses selection, for use in a parameter search.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores, None if `score_func` returned only scores.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.feature_selection import SelectKBest, chi2
    >>> X, y = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
    >>> X_new.shape
    (1797, 20)

    Notes
    -----
    Ties between features with equal scores will be broken in an unspecified
    way.

    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    mutual_info_regression: Mutual information for a continuous target.
    ANOVAFScorerClassification: F-value between label/feature for \
    classification tasks.
    ANOVAFScorerRegression: F-value between label/feature for regression tasks.
    Chi2Scorer: Chi-squared stats of non-negative features for classification \
    tasks.
    MutualInfoScorerClassification: Mutual information for a discrete target.
    MutualInfoScorerRegression: Mutual information for a continuous target.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    GenericUnivariateSelect: Univariate feature selector with configurable \
    mode.
    """

    def __init__(self, score_func=f_classif, k=10):
        super().__init__(score_func)
        self.k = k

    def _check_params(self, X, y):
        super()._check_params(X, y)
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError("k should be >=0, <= n_features = %d; got %r. "
                             "Use k='all' to return all features."
                             % (X.shape[1], self.k))

    def _get_support_mask(self):
        check_is_fitted(self)
        if self.k == 'all':
            return np.ones(self.scores_.shape, dtype=bool)
        elif self.k == 0:
            return np.zeros(self.scores_.shape, dtype=bool)
        else:
            scores = _clean_nans(self.scores_)
            mask = np.zeros(scores.shape, dtype=bool)

            # Request a stable sort. Mergesort takes more memory (~40MB per
            # megafeature on x86-64).
            mask[np.argsort(scores, kind="mergesort")[-self.k:]] = 1
            return mask


class SelectFpr(_BaseFilter):
    """Filter: Select the pvalues below alpha based on a FPR test.

    FPR test stands for False Positive Rate test. It controls the total
    amount of false detections.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable or object
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores or a scorer object.
        Default is f_classif (see below "See also"). The default function only
        works with classification tasks.

    alpha : float, optional
        The highest p-value for features to be kept.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import SelectFpr, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> X_new = SelectFpr(chi2, alpha=0.01).fit_transform(X, y)
    >>> X_new.shape
    (569, 16)

    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    mutual_info_regression: Mutual information for a continuous target.
    ANOVAFScorerClassification: F-value between label/feature for \
    classification tasks.
    ANOVAFScorerRegression: F-value between label/feature for regression tasks.
    Chi2Scorer: Chi-squared stats of non-negative features for classification \
    tasks.
    MutualInfoScorerClassification: Mutual information for a discrete target.
    MutualInfoScorerRegression: Mutual information for a continuous target.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    GenericUnivariateSelect: Univariate feature selector with configurable \
    mode.
    """

    def __init__(self, score_func=f_classif, alpha=5e-2):
        super().__init__(score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.pvalues_ < self.alpha


class SelectFdr(_BaseFilter):
    """Filter: Select the p-values for an estimated false discovery rate

    This uses the Benjamini-Hochberg procedure. ``alpha`` is an upper bound
    on the expected false discovery rate.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable or object
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores or a scorer object.
        Default is f_classif (see below "See also"). The default function only
        works with classification tasks.

    alpha : float, optional
        The highest uncorrected p-value for features to keep.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import SelectFdr, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> X_new = SelectFdr(chi2, alpha=0.01).fit_transform(X, y)
    >>> X_new.shape
    (569, 16)

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores.

    References
    ----------
    https://en.wikipedia.org/wiki/False_discovery_rate

    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    mutual_info_regression: Mutual information for a continuous target.
    ANOVAFScorerClassification: F-value between label/feature for \
    classification tasks.
    ANOVAFScorerRegression: F-value between label/feature for regression tasks.
    Chi2Scorer: Chi-squared stats of non-negative features for classification \
    tasks.
    MutualInfoScorerClassification: Mutual information for a discrete target.
    MutualInfoScorerRegression: Mutual information for a continuous target.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    GenericUnivariateSelect: Univariate feature selector with configurable \
    mode.
    """

    def __init__(self, score_func=f_classif, alpha=5e-2):
        super().__init__(score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        check_is_fitted(self)
        n_features = len(self.pvalues_)
        sv = np.sort(self.pvalues_)
        selected = sv[sv <= float(self.alpha) / n_features *
                      np.arange(1, n_features + 1)]
        if selected.size == 0:
            return np.zeros_like(self.pvalues_, dtype=bool)
        return self.pvalues_ <= selected.max()


class SelectFwe(_BaseFilter):
    """Filter: Select the p-values corresponding to Family-wise error rate

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable or object
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores or a scorer object.
        Default is f_classif (see below "See also"). The default function only
        works with classification tasks.

    alpha : float, optional
        The highest uncorrected p-value for features to keep.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import SelectFwe, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> X_new = SelectFwe(chi2, alpha=0.01).fit_transform(X, y)
    >>> X_new.shape
    (569, 15)

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores.

    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    mutual_info_regression: Mutual information for a continuous target.
    ANOVAFScorerClassification: F-value between label/feature for \
    classification tasks.
    ANOVAFScorerRegression: F-value between label/feature for regression tasks.
    Chi2Scorer: Chi-squared stats of non-negative features for classification \
    tasks.
    MutualInfoScorerClassification: Mutual information for a discrete target.
    MutualInfoScorerRegression: Mutual information for a continuous target.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    GenericUnivariateSelect: Univariate feature selector with configurable \
    mode.
    """

    def __init__(self, score_func=f_classif, alpha=5e-2):
        super().__init__(score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        check_is_fitted(self)
        return (self.pvalues_ < self.alpha / len(self.pvalues_))


######################################################################
# Generic filter
######################################################################

# TODO this class should fit on either p-values or scores,
# depending on the mode.
class GenericUnivariateSelect(_BaseFilter):
    """Univariate feature selector with configurable strategy.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable or object
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores or a scorer object.
        For modes 'percentile' or 'kbest' it can return a single array scores.

    mode : {'percentile', 'k_best', 'fpr', 'fdr', 'fwe'}
        Feature selection mode.

    param : float or int depending on the feature selection mode
        Parameter of the corresponding mode.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores, None if `score_func` returned scores only.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import GenericUnivariateSelect, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> transformer = GenericUnivariateSelect(chi2, 'k_best', param=20)
    >>> X_new = transformer.fit_transform(X, y)
    >>> X_new.shape
    (569, 20)

    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    mutual_info_regression: Mutual information for a continuous target.
    ANOVAFScorerClassification: F-value between label/feature for \
    classification tasks.
    ANOVAFScorerRegression: F-value between label/feature for regression tasks.
    Chi2Scorer: Chi-squared stats of non-negative features for classification \
    tasks.
    MutualInfoScorerClassification: Mutual information for a discrete target.
    MutualInfoScorerRegression: Mutual information for a continuous target.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    GenericUnivariateSelect: Univariate feature selector with configurable \
    mode.
    """

    _selection_modes = {'percentile': SelectPercentile,
                        'k_best': SelectKBest,
                        'fpr': SelectFpr,
                        'fdr': SelectFdr,
                        'fwe': SelectFwe}

    def __init__(self, score_func=f_classif, mode='percentile', param=1e-5):
        super().__init__(score_func)
        self.mode = mode
        self.param = param

    def _make_selector(self):
        selector = self._selection_modes[self.mode](score_func=self.score_func)
        # Now perform some acrobatics to set the right named parameter in
        # the selector
        possible_params = selector._get_param_names()
        possible_params.remove('score_func')
        selector.set_params(**{possible_params[0]: self.param})

        return selector

    def _check_params(self, X, y):
        super()._check_params(X, y)
        if self.mode not in self._selection_modes:
            raise ValueError("The mode passed should be one of %s, %r,"
                             " (type %s) was passed."
                             % (self._selection_modes.keys(), self.mode,
                                type(self.mode)))
        self._make_selector()._check_params(X, y)

    def _get_support_mask(self):
        check_is_fitted(self)
        selector = self._make_selector()
        selector.pvalues_ = self.pvalues_
        selector.scores_ = self.scores_
        return selector._get_support_mask()
