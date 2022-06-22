"""Univariate model feature selection."""

import numpy as np

from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.utils import check_X_y
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted, check_memory

from ..feature_selection import ExtendedSelectorMixin


def _get_scores(estimator, X, y, **fit_params):
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        Xj = X[:, [j]]
        scores[j] = estimator.fit(Xj, y, **fit_params).score(Xj, y)
    return scores


class SelectFromUnivariateModel(
    ExtendedSelectorMixin, MetaEstimatorMixin, BaseEstimator
):
    """Select features according to scores calculated from model fitting on
    each individual feaure.

    Parameters
    ----------
    estimator : object
        The external estimator used to calculate univariate feature scores.

    k : int or "all" (default = "all")
        Number of top features to select.
        The "all" option bypasses selection, for use in a parameter search.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    estimator_ : an estimator
        The external estimator fit on the reduced dataset.

    scores_ : array-like of shape (n_features,)
        Feature scores.
    """

    def __init__(self, estimator, k="all", memory=None):
        self.estimator = estimator
        self.k = k
        self.memory = memory

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        return self.estimator_.classes_

    def fit(self, X, y, **fit_params):
        """Use model to fit each feature individually and calculate scores.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The training input samples.

        y : array-like, shape = (n_samples,)
            The target values.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        estimator = clone(self.estimator)
        self.scores_ = memory.cache(_get_scores)(estimator, X, y, **fit_params)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), y, **fit_params)
        return self

    @if_delegate_has_method(delegate="estimator")
    def predict(self, X):
        """Reduce X to the selected features and then predict using the
           underlying estimator.

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

    @if_delegate_has_method(delegate="estimator")
    def score(self, X, y, sample_weight=None):
        """Reduce X to the selected features and then return the score of the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the estimator.
        """
        check_is_fitted(self)
        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        return self.estimator_.score(self.transform(X), y, **score_params)

    @if_delegate_has_method(delegate="estimator")
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

    @if_delegate_has_method(delegate="estimator")
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

    @if_delegate_has_method(delegate="estimator")
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
        return {"allow_nan": estimator_tags.get("allow_nan", True)}

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be >=0, <= n_features = %d; got %r. "
                "Use k='all' to return all features." % (X.shape[1], self.k)
            )

    def _get_support_mask(self):
        check_is_fitted(self)
        if self.k == "all":
            return np.ones_like(self.scores_, dtype=bool)
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.k > 0:
            mask[np.argsort(self.scores_, kind="mergesort")[-self.k :]] = True
        return mask
