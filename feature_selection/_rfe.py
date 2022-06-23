# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Vincent Michel <vincent.michel@inria.fr>
#          Gilles Louppe <g.louppe@gmail.com>
#          Leandro Hermida <hermidal@cs.umd.edu>
#
# License: BSD 3 clause

"""Recursive feature elimination for feature ranking with advanced
functionalties and redesign of scikit-learn version to be more efficient
and better performance
"""
import numbers

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import clone, is_classifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import check_cv
from sklearn.utils import safe_sqr
from sklearn.utils.metaestimators import available_if, _safe_split
from sklearn.utils.validation import (
    indexable,
    check_is_fitted,
    _check_fit_params,
    check_memory,
)

from ._base import ExtendedSelectorMixin
from ..metrics import check_scoring
from ..model_selection._validation import _score
from ..utils.metaestimators import check_routing


def _rfe_fit(
    base_estimator, X, y, fit_params, steps, keep_features, verbose=0, step_score=None
):
    # step_score parameter controls the calculation of self.scores_.
    # step_score is not exposed to users and is only used when implementing
    # RFECV self.scores_ and will not be calculated when calling regular
    # fit() method

    supports = np.ones((len(steps) + 1, X.shape[1]), dtype=np.bool)
    rankings = np.ones((len(steps) + 1, X.shape[1]), dtype=np.int)

    if step_score:
        scores = []

    # Elimination
    remaining_features = np.setdiff1d(
        np.arange(X.shape[1]), keep_features, assume_unique=True
    )
    for step_num, step in enumerate(steps, start=1):
        # Rank the remaining features
        if verbose > 0:
            print(
                "Fitting estimator with {:d} features".format(remaining_features.size)
            )

        features = np.union1d(remaining_features, keep_features)
        estimator = clone(base_estimator)
        estimator.fit(X[:, features], y, **fit_params)

        # Get coefs
        if hasattr(estimator, "coef_"):
            coefs = estimator.coef_
        elif hasattr(estimator, "feature_importances_"):
            coefs = estimator.feature_importances_
        else:
            raise RuntimeError(
                'The classifier does not expose "coef_" or '
                '"feature_importances_" attributes.'
            )

        # Get ranks
        coef_idxs = np.where(np.isin(features, remaining_features, assume_unique=True))[
            0
        ]
        if coefs.ndim > 1:
            ranks = np.argsort(safe_sqr(coefs[:, coef_idxs]).sum(axis=0))
        else:
            ranks = np.argsort(safe_sqr(coefs[coef_idxs]))

        # for sparse case ranks is matrix
        ranks = np.ravel(ranks)

        # Compute step score on the previous selection iteration because
        # 'estimator' must use features that have not been eliminated yet
        if step_score:
            scores.append(step_score(estimator, features))

        # Eliminate worst features
        eliminate_features, remaining_features = np.split(
            remaining_features[ranks], [step]
        )
        remaining_features = np.sort(remaining_features)
        supports[step_num] = supports[step_num - 1]
        rankings[step_num] = rankings[step_num - 1]
        supports[step_num, eliminate_features] = False
        rankings[step_num, np.logical_not(supports[step_num])] += 1

    return supports, rankings


def _rfe_single_fit(
    rfe,
    estimator,
    X,
    y,
    train,
    test,
    scorer,
    fit_params,
    score_params=None,
    feature_params=None,
):
    """
    Return the score for a fit across one fold.
    """
    # Subset fit_params values for train indices
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)
    # fit_params = {**fit_params, **feature_params}
    # Subset score_params values for test indices
    score_params = score_params if score_params is not None else {}
    score_params = _check_fit_params(X, score_params, test)
    score_params = {**score_params, **feature_params}

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    rfe._fit(
        X_train,
        y_train,
        fit_params,
        feature_params,
        lambda estimator, features: _score(
            estimator, X_test[:, features], y_test, scorer, score_params
        ),
    )
    return rfe.scores_, rfe.n_remaining_feature_steps_


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the first fitted estimator if available, otherwise we
    check the unfitted estimator.
    """
    return lambda self: (
        hasattr(self.estimator_, attr)
        if hasattr(self, "estimator_")
        else hasattr(self.estimator, attr)
    )


class ExtendedRFE(ExtendedSelectorMixin, RFE):
    """Feature ranking with advanced recursive feature elimination.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained either through
    any specific attribute or callable.
    Then, the least important features are pruned from current set of features.
    That procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance
        (e.g. `coef_`, `feature_importances_`).

    n_features_to_select : int or float, default=None
        The number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of features to
        select.

        .. versionchanged:: 0.24
           Added float values for fractions.

    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    tune_step_at : int or float or None, optional (default=None)
        Number of remaining features reached when ``tuning_step`` is used
        rather than ``step``. May be specified as an (integer) number of
        remaining features or, if within (0.0, 1.0), a percentage (rounded
        down) of the original number of features. If original number of
        features and parameter settings would result in stepping past
        ``tune_step_at``, then the number of features removed in the iteration
        prior to stepping over will adjust to arrive at this value.

    tuning_step : int or float, optional (default=1)
        Step to use starting at ``tune_step_at`` number of remaining features.
        If greater than or equal to 1, then ``tuning_step`` corresponds to the
        (integer) number of features to remove at each iteration. If within
        (0.0, 1.0), then ``tuning_step`` corresponds to the percentage (rounded
        down) of features to remove at each iteration.

    reducing_step : boolean, optional (default=False)
        If true and ``step`` or ``tuning_step`` is a float, the number of
        features removed is calculated as a fraction of the remaining features
        in that iteration. If false, the number of features removed is constant
        across iterations and a fraction of the original number of features for
        ``step`` or fraction of the ``tune_step_at`` number of remaining
        features for ``tuning_step``.

    verbose : int, (default=0)
        Controls verbosity of output.

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a `coef_`
        or `feature_importances_` attributes of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance (implemented with `attrgetter`).
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should
        return importance for each feature.

        .. versionadded:: 0.24

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only available when `estimator` is a classifier.

    estimator_ : ``Estimator`` instance
        The fitted estimator used to select features.

    n_features_ : int
        The number of selected features.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    ranking_ : ndarray of shape (n_features,)
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    See Also
    --------
    RFECV : Recursive feature elimination with built-in cross-validated
        selection of the best number of features.
    SelectFromModel : Feature selection based on thresholds of importance
        weights.
    SequentialFeatureSelector : Sequential cross-validation based feature
        selection. Does not rely on importance weights.

    Notes
    -----
    Allows NaN/Inf in the input if the underlying estimator does as well.

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.

    Examples
    --------
    The following example shows how to retrieve the 5 most informative
    features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFE
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFE(estimator, n_features_to_select=5, step=1)
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
    """

    def __init__(
        self,
        estimator,
        n_features_to_select=None,
        step=1,
        tune_step_at=None,
        tuning_step=1,
        reducing_step=False,
        verbose=0,
        importance_getter="auto",
        memory=None,
        penalty_factor_meta_col=None,
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.tune_step_at = tune_step_at
        self.tuning_step = tuning_step
        self.reducing_step = reducing_step
        self.verbose = verbose
        self.importance_getter = importance_getter
        self.memory = memory
        self.penalty_factor_meta_col = penalty_factor_meta_col

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        """Classes labels available when `estimator` is a classifier.

        Returns
        -------
        ndarray of shape (n_classes,)
        """
        return self.estimator_.classes_

    def fit(self, X, y, feature_meta=None, **fit_params):
        """Fit the RFE model and then the underlying estimator on the selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata.

        **fit_params : dict
            Additional parameters passed to the `fit` method of the underlying
            estimator.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        tags = self._get_tags()
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
            multi_output=True,
        )
        error_msg = (
            "n_features_to_select must be either None, a "
            "positive integer representing the absolute "
            "number of features or a float in (0.0, 1.0] "
            "representing a percentage of features to "
            f"select. Got {self.n_features_to_select}"
        )

        self._check_params(X, y, feature_meta)
        memory = check_memory(self.memory)

        # Initialization
        if self.penalty_factor_meta_col is None:
            keep_features = np.array([], dtype=np.int)
            n_features = X.shape[1]
        else:
            penalty_factor = feature_meta[self.penalty_factor_meta_col].to_numpy(
                dtype=float
            )
            keep_features = np.where(penalty_factor == 0)[0]
            n_features = X.shape[1] - keep_features.size

        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        elif self.n_features_to_select < 0:
            raise ValueError(error_msg)
        elif isinstance(self.n_features_to_select, numbers.Integral):  # int
            n_features_to_select = self.n_features_to_select
        elif self.n_features_to_select > 1.0:  # float > 1
            raise ValueError(error_msg)
        else:  # float
            n_features_to_select = int(n_features * self.n_features_to_select)

        steps = self._get_steps(n_features, n_features_to_select)

        supports, rankings = memory.cache(_rfe_fit, ignore=["verbose"])(
            self.estimator, X, y, fit_params, steps, keep_features, verbose=self.verbose
        )

        for rfe_idx, (support, ranking) in enumerate(zip(supports, rankings)):
            n_remaining_features = np.count_nonzero(support) - keep_features.size
            if n_remaining_features <= n_features_to_select:
                if n_remaining_features == n_features_to_select:
                    features = np.where(support)[0]
                    remaining_features = np.setdiff1d(
                        features, keep_features, assume_unique=True
                    )
                else:
                    support = supports[rfe_idx - 1]
                    ranking = rankings[rfe_idx - 1]
                    features = np.where(support)[0]
                    remaining_features = np.setdiff1d(
                        features, keep_features, assume_unique=True
                    )
                    estimator = clone(self.estimator)
                    estimator.fit(X[:, features], y, **fit_params)
                    if hasattr(estimator, "coef_"):
                        coefs = estimator.coef_
                    elif hasattr(estimator, "feature_importances_"):
                        coefs = estimator.feature_importances_
                    coef_idxs = np.where(
                        np.isin(features, remaining_features, assume_unique=True)
                    )[0]
                    if coefs.ndim > 1:
                        ranks = np.argsort(safe_sqr(coefs[:, coef_idxs]).sum(axis=0))
                    else:
                        ranks = np.argsort(safe_sqr(coefs[coef_idxs]))
                    ranks = np.ravel(ranks)
                    step = remaining_features.size - n_features_to_select
                    eliminate_features, remaining_features = np.split(
                        remaining_features[ranks], [step]
                    )
                    remaining_features = np.sort(remaining_features)
                    support[eliminate_features] = False
                    ranking[np.logical_not(support)] += 1
                    features = np.union1d(remaining_features, keep_features)
                if self.verbose > 0:
                    print(
                        "Fitting estimator with {:d} features".format(
                            remaining_features.size
                        )
                    )
                estimator = clone(self.estimator)
                estimator.fit(X[:, features], y, **fit_params)
                # if step_score:
                #     scores.append(step_score(estimator, features))
                self.support_ = support
                self.ranking_ = ranking
                self.estimator_ = estimator
                break

        self.n_features_ = np.count_nonzero(self.support_)

        return self

    def _get_steps(self, n_features, n_features_to_select):

        if self.step >= 1.0:
            step = int(self.step)
        elif 0.0 < self.step < 1.0 and not self.reducing_step:
            step = int(max(1, self.step * n_features))
        elif self.step <= 0:
            raise ValueError("step must be > 0")

        if self.tune_step_at is not None:
            if self.tune_step_at >= 1.0:
                tune_step_at = int(self.tune_step_at)
            elif 0.0 < self.tune_step_at < 1.0:
                tune_step_at = int(max(1, self.tune_step_at * n_features))
            if not n_features_to_select < tune_step_at < n_features:
                raise ValueError(
                    "tune_step_at must be greater than "
                    "n_features_to_select and less than initial "
                    "number of features"
                )
            if self.tuning_step >= 1.0:
                tuning_step = int(self.tuning_step)
            elif 0.0 < self.tuning_step < 1.0 and not self.reducing_step:
                tuning_step = int(max(1, self.tuning_step * tune_step_at))
            elif self.tuning_step <= 0:
                raise ValueError("tuning_step must be > 0")

        steps = []
        n_remaining_features = n_features
        n_remaining_feature_steps = [n_remaining_features]
        while n_remaining_features > 1:
            if self.tune_step_at is not None:
                if n_remaining_features > tune_step_at:
                    if 0.0 < self.step < 1.0 and self.reducing_step:
                        step = int(
                            max(
                                1,
                                min(
                                    n_remaining_features - tune_step_at,
                                    self.step * n_remaining_features,
                                ),
                            )
                        )
                    else:
                        step = min(n_remaining_features - tune_step_at, step)
                elif 0.0 < self.tuning_step < 1.0 and self.reducing_step:
                    step = int(
                        max(
                            1,
                            min(
                                n_remaining_features - 1,
                                self.tuning_step * n_remaining_features,
                            ),
                        )
                    )
                else:
                    step = min(n_remaining_features - 1, tuning_step)
            elif 0.0 < self.step < 1.0 and self.reducing_step:
                step = int(
                    max(
                        1,
                        min(n_remaining_features - 1, self.step * n_remaining_features),
                    )
                )
            else:
                step = min(n_remaining_features - 1, step)
            n_remaining_features -= step
            n_remaining_feature_steps.append(n_remaining_features)
            steps.append(step)

        self.n_remaining_feature_steps_ = np.array(
            n_remaining_feature_steps, dtype=np.int
        )
        return steps

    def _check_params(self, X, y, feature_meta):
        if self.n_features_to_select is not None and self.n_features_to_select < 1:
            raise ValueError("n_features_to_select must be >= 1")
        if self.penalty_factor_meta_col is not None:
            if feature_meta is None:
                raise ValueError(
                    "penalty_factor_meta_col specified but " "feature_meta not passed."
                )
            if self.penalty_factor_meta_col not in feature_meta.columns:
                raise ValueError(
                    "%s feature_meta column does not exist."
                    % self.penalty_factor_meta_col
                )
            if X.shape[1] != feature_meta.shape[0]:
                raise ValueError(
                    "X ({:d}) and feature_meta ({:d}) have "
                    "different feature dimensions".format(
                        X.shape[1], feature_meta.shape[0]
                    )
                )

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Reduce X to the selected features and then predict using the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        check_is_fitted(self)
        return self.estimator_.predict(self.transform(X))

    @available_if(_estimator_has("score"))
    def score(self, X, y, **score_params):
        """Reduce X to the selected features and return the score of the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.

        **score_params : dict
            Parameters to pass to the `score` method of the underlying
            estimator.

            .. versionadded:: 1.0

        Returns
        -------
        score : float
            Score of the underlying base estimator computed with the selected
            features returned by `rfe.transform(X)` and `y`.
        """
        check_is_fitted(self)
        return self.estimator_.score(self.transform(X), y, **score_params)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like or sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : array, shape = [n_samples, n_classes] or [n_samples]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification produce an array of shape
            [n_samples].
        """
        check_is_fitted(self)
        return self.estimator_.decision_function(self.transform(X))

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like or sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.estimator_.predict_proba(self.transform(X))

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.estimator_.predict_log_proba(self.transform(X))

    def _more_tags(self):
        estimator_tags = self.estimator._get_tags()
        return {"poor_score": True, "allow_nan": estimator_tags.get("allow_nan", True)}


class ExtendedRFECV(ExtendedRFE):
    """Feature ranking with recursive feature elimination and cross-validated
    selection of the best number of features.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
        Note that the last iteration may remove fewer than ``step`` features in
        order to reach ``min_features_to_select``.

    tune_step_at : int or float or None, optional (default=None)
        Number of remaining features reached when ``tuning_step`` is used
        rather than ``step``. May be specified as an (integer) number of
        remaining features or, if within (0.0, 1.0), a percentage (rounded
        down) of the original number of features. If original number of
        features and parameter settings would result in stepping past
        ``tune_step_at``, then the number of features removed in the iteration
        prior to stepping over will adjust to arrive at this value.

    tuning_step : int or float, optional (default=1)
        Step to use starting at ``tune_step_at`` number of remaining features.
        If greater than or equal to 1, then ``tuning_step`` corresponds to the
        (integer) number of features to remove at each iteration. If within
        (0.0, 1.0), then ``tuning_step`` corresponds to the percentage (rounded
        down) of features to remove at each iteration.

    reducing_step : boolean, optional (default=False)
        If true and ``step`` or ``tuning_step`` is a float, the number of
        features removed is calculated as a fraction of the remaining features
        in that iteration. If false, the number of features removed is constant
        across iterations and a fraction of the original number of features for
        ``step`` or fraction of the ``tune_step_at`` number of remaining
        features for ``tuning_step``.

    min_features_to_select : int, default=1
        The minimum number of features to be selected. This number of features
        will always be scored, even if the difference between the original
        feature count and ``min_features_to_select`` isn't divisible by
        ``step``.

        .. versionadded:: 0.20

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value of None changed from 3-fold to 5-fold.

    scoring : str, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int or None, default=None
        Number of cores to run in parallel while fitting across folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.18

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a `coef_`
        or `feature_importances_` attributes of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance.
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        :class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should
        return importance for each feature.

        .. versionadded:: 0.24

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only available when `estimator` is a classifier.

    estimator_ : ``Estimator`` instance
        The fitted estimator used to select features.

    grid_scores_ : ndarray of shape (n_subsets_of_features,)
        The cross-validation scores such that
        ``grid_scores_[i]`` corresponds to
        the CV score of the i-th subset of features.

        .. deprecated:: 1.0
            The `grid_scores_` attribute is deprecated in version 1.0 in favor
            of `cv_results_` and will be removed in version 1.2.

    cv_results_ : dict of ndarrays
        A dict with keys:

        split(k)_test_score : ndarray of shape (n_features,)
            The cross-validation scores across (k)th fold.

        mean_test_score : ndarray of shape (n_features,)
            Mean of scores over the folds.

        std_test_score : ndarray of shape (n_features,)
            Standard deviation of scores over the folds.

        .. versionadded:: 1.0

    n_features_ : int
        The number of selected features with cross-validation.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    ranking_ : narray of shape (n_features,)
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    See Also
    --------
    RFE : Recursive feature elimination.

    Notes
    -----
    The size of ``grid_scores_`` is equal to
    ``ceil((n_features - min_features_to_select) / step) + 1``,
    where step is the number of features removed at each iteration.

    Allows NaN/Inf in the input if the underlying estimator does as well.

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.

    Examples
    --------
    The following example shows how to retrieve the a-priori not known 5
    informative features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFECV
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFECV(estimator, step=1, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
    """

    def __init__(
        self,
        estimator,
        *,
        step=1,
        tune_step_at=None,
        tuning_step=1,
        reducing_step=False,
        min_features_to_select=1,
        cv=None,
        scoring=None,
        verbose=0,
        n_jobs=None,
        importance_getter="auto",
        param_routing=None,
        penalty_factor_meta_col=None,
    ):
        self.estimator = estimator
        self.step = step
        self.tune_step_at = tune_step_at
        self.tuning_step = tuning_step
        self.reducing_step = reducing_step
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.importance_getter = importance_getter
        self.min_features_to_select = min_features_to_select
        self.penalty_factor_meta_col = penalty_factor_meta_col
        self.param_routing = param_routing
        self.router = check_routing(
            self.param_routing,
            ["estimator", "cv", "scoring"],
            {"cv": "groups", "estimator": "-groups"},
        )

    def set_params(self, **params):
        super().set_params(**params)
        if "param_routing" in params:
            self.router = check_routing(
                self.param_routing,
                ["estimator", "cv", "scoring"],
                {"cv": "groups", "estimator": "-groups"},
            )
        return self

    def fit(self, X, y, groups=None, feature_meta=None, **fit_params):
        """Fit the RFE model and automatically tune the number of selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the total number of features.

        y : array-like of shape (n_samples,)
            Target values (integers for classification, real numbers for
            regression).

        groups : array-like of shape (n_samples,) or None, default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

            .. versionadded:: 0.20

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        tags = self._get_tags()
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
            multi_output=True,
        )

        # Initialization
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]
        if self.penalty_factor_meta_col is not None:
            penalty_factor = feature_meta[self.penalty_factor_meta_col].to_numpy(
                dtype=float
            )
            n_features -= np.count_nonzero(penalty_factor == 0)

        # Build an RFE object, which will evaluate and score each possible
        # feature count, down to self.min_features_to_select
        rfe = ExtendedRFE(
            estimator=self.estimator,
            n_features_to_select=self.min_features_to_select,
            step=self.step,
            tune_step_at=self.tune_step_at,
            tuning_step=self.tuning_step,
            reducing_step=self.reducing_step,
            verbose=self.verbose,
            importance_getter=self.importance_getter,
            penalty_factor_meta_col=self.penalty_factor_meta_col,
        )

        # Determine the number of subsets of features by fitting across the
        # train folds and choosing the "features_to_select" parameter that
        # gives the least averaged error across all folds.

        # Note that joblib raises a non-picklable error for bound methods even
        # if n_jobs is set to 1 with the default multiprocessing backend. This
        # branching is done so that to make sure that user code that sets
        # n_jobs to 1 and provides bound methods as scorers is not broken with
        # the addition of n_jobs parameter in version 0.18.

        # so feature metadata/properties can work
        feature_params = {k: v for k, v in fit_params.items() if k == "feature_meta"}
        fit_params = {k: v for k, v in fit_params.items() if k != "feature_meta"}

        X, y, *fit_params_values = indexable(X, y, *fit_params.values())
        fit_params = dict(zip(fit_params.keys(), fit_params_values))
        fit_params = _check_fit_params(X, fit_params)

        (fit_params, cv_params, score_params), remainder = self.router(fit_params)
        if remainder:
            raise TypeError(
                "fit() got unexpected keyword arguments %r" % sorted(remainder)
            )

        if effective_n_jobs(self.n_jobs) == 1:
            parallel, func = list, _rfe_single_fit
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            func = delayed(_rfe_single_fit)

        fit_and_score_kwargs = dict(
            fit_params=fit_params,
            score_params=score_params,
            feature_params=feature_params,
        )

        scores, n_remaining_feature_steps = zip(
            *parallel(
                func(
                    rfe,
                    self.estimator,
                    X,
                    y,
                    train,
                    test,
                    scorer,
                    **fit_and_score_kwargs,
                )
                for train, test in cv.split(X, y, **cv_params)
            )
        )

        scores = np.array(scores)
        scores_sum = np.sum(scores, axis=0)
        scores_sum_rev = scores_sum[::-1]
        # Each same so just get first
        n_remaining_feature_steps = n_remaining_feature_steps[0]
        # Reverse scores and num remaining feature steps to select argmax score
        # with lowest number of features in case of a score tie
        n_features_to_select = n_remaining_feature_steps[::-1][
            np.argmax(scores_sum_rev)
        ]

        if self.tune_step_at is not None:
            if self.tune_step_at >= 1.0:
                tune_step_at = int(self.tune_step_at)
            elif 0.0 < self.tune_step_at < 1.0:
                tune_step_at = int(max(1, self.tune_step_at * n_features))
            if tune_step_at <= n_features_to_select:
                tune_step_at = None
        else:
            tune_step_at = None

        # Re-execute an elimination with best_k over the whole set
        rfe = ExtendedRFE(
            estimator=self.estimator,
            n_features_to_select=n_features_to_select,
            step=self.step,
            tune_step_at=tune_step_at,
            tuning_step=self.tuning_step,
            reducing_step=self.reducing_step,
            verbose=self.verbose,
            importance_getter=self.importance_getter,
            penalty_factor_meta_col=self.penalty_factor_meta_col,
        )

        rfe.fit(X, y, **fit_params, **feature_params)

        # Set final attributes
        self.support_ = rfe.support_
        self.n_features_ = rfe.n_features_
        self.ranking_ = rfe.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), y, **fit_params, **feature_params)
        self.n_remaining_feature_steps_ = n_remaining_feature_steps

        # reverse to stay consistent with before
        scores_rev = scores[:, ::-1]
        self.cv_results_ = {}
        self.cv_results_["mean_test_score"] = np.mean(scores_rev, axis=0)
        self.cv_results_["std_test_score"] = np.std(scores_rev, axis=0)

        for i in range(scores.shape[0]):
            self.cv_results_[f"split{i}_test_score"] = scores_rev[i]

        return self

    # TODO: Remove in v1.2 when grid_scores_ is removed
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "The `grid_scores_` attribute is deprecated in version 1.0 in favor "
        "of `cv_results_` and will be removed in version 1.2."
    )
    @property
    def grid_scores_(self):
        # remove 2 for mean_test_score, std_test_score
        grid_size = len(self.cv_results_) - 2
        return np.asarray(
            [self.cv_results_[f"split{i}_test_score"] for i in range(grid_size)]
        ).T
