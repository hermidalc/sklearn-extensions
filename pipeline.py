"""
The :mod:`sklearn.pipeline` module implements utilities to build a composite
estimator, as a chain of transforms and estimators.
"""

# Author: Edouard Duchesnay
#         Gael Varoquaux
#         Virgile Fritsch
#         Alexandre Gramfort
#         Lars Buitinck
#         Leandro Hermida
# License: BSD

from collections import defaultdict
from itertools import islice

import numpy as np
import pandas as pd
from joblib import Parallel
from scipy import sparse

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import Bunch, _print_elapsed_time
from sklearn.utils._estimator_html_repr import _VisualBlock
from sklearn.utils._tags import _safe_tags
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, check_memory

from .base import ExtendedTransformerMixin
from .utils.metaestimators import check_routing


__all__ = [
    "ExtendedPipeline",
    "ExtendedFeatureUnion",
    "transform_feature_meta",
    "make_extended_pipeline",
    "make_extended_union",
]


def _final_estimator_has(attr):
    """Check that final_estimator has `attr`.

    Used together with `avaliable_if` in `Pipeline`."""

    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self._final_estimator, attr)
        return True

    return check


class ExtendedPipeline(Pipeline):
    """
    Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement `fit` and `transform` methods.
    The final estimator only needs to implement `fit`.
    The transformers in the pipeline can be cached using ``memory`` argument.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters. For this, it
    enables setting parameters of the various steps using their names and the
    parameter name separated by a `'__'`, as in the example below. A step's
    estimator may be replaced entirely by setting the parameter with its name
    to another estimator, or a transformer removed by setting it to
    `'passthrough'` or `None`.

    Read more in the :ref:`User Guide <pipeline>`.

    .. versionadded:: 0.5

    Parameters
    ----------
    steps : list of tuple
        List of (name, transform) tuples (implementing `fit`/`transform`) that
        are chained, in the order in which they are chained, with the last
        object an estimator.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only exist if the last step of the pipeline is a
        classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying first estimator in `steps` exposes such an attribute
        when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    make_pipeline : Convenience function for simplified pipeline construction.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.pipeline import Pipeline
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
    >>> # The pipeline can be used as any other estimator
    >>> # and avoids leaking the test set into the train set
    >>> pipe.fit(X_train, y_train)
    Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])
    >>> pipe.score(X_test, y_test)
    0.88
    """

    # BaseEstimator interface
    _required_parameters = ["steps"]

    def __init__(self, steps, *, memory=None, verbose=False, param_routing=None):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        self.param_routing = param_routing
        self.router = check_routing(
            self.param_routing,
            [[name, "*"] for name, _ in self.steps],
            self._default_routing,
        )

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `steps` of the `Pipeline`.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("steps", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``. Note that
        you can directly set the parameters of the estimators contained in
        `steps`.

        Parameters
        ----------
        **kwargs : dict
            Parameters of this estimator or parameters of estimators contained
            in `steps`. Parameters of the steps may be set using its name and
            the parameter name separated by a '__'.

        Returns
        -------
        self : object
            Pipeline class instance.
        """
        self._set_params("steps", **kwargs)
        if "param_routing" in kwargs:
            self.router = check_routing(
                self.param_routing,
                [[name, "*"] for name, _ in self.steps],
                self._default_routing,
            )
        return self

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if t is None or t == "passthrough":
                continue
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
                t, "transform"
            ):
                raise TypeError(
                    "All intermediate steps should be "
                    "transformers and implement fit and transform "
                    "or be the string 'passthrough' "
                    "'%s' (type %s) doesn't" % (t, type(t))
                )

        # We allow last estimator to be None as an identity transformation
        if (
            estimator is not None
            and estimator != "passthrough"
            and not hasattr(estimator, "fit")
        ):
            raise TypeError(
                "Last step of Pipeline should implement fit "
                "or be the string 'passthrough'. "
                "'%s' (type %s) doesn't" % (estimator, type(estimator))
            )

    def _iter(self, with_final=True, filter_passthrough=True):
        """
        Generate (idx, (name, trans)) tuples from self.steps

        When filter_passthrough is True, 'passthrough' and None transformers
        are filtered out.
        """
        stop = len(self.steps)
        if not with_final:
            stop -= 1

        for idx, (name, trans) in enumerate(islice(self.steps, 0, stop)):
            if not filter_passthrough:
                yield idx, name, trans
            elif trans is not None and trans != "passthrough":
                yield idx, name, trans

    def __len__(self):
        """
        Returns the length of the Pipeline
        """
        return len(self.steps)

    def __getitem__(self, ind):
        """Returns a sub-pipeline or a single estimator in the pipeline

        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.
        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            return self.__class__(
                self.steps[ind], memory=self.memory, verbose=self.verbose
            )
        try:
            name, est = self.steps[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_steps[ind]
        return est

    @property
    def _estimator_type(self):
        return self.steps[-1][1]._estimator_type

    @property
    def named_steps(self):
        """Access the steps by name.

        Read-only attribute to access any step by given name.
        Keys are steps names and values are the steps objects."""
        # Use Bunch object to improve autocomplete
        return Bunch(**dict(self.steps))

    @property
    def _final_estimator(self):
        estimator = self.steps[-1][1]
        return "passthrough" if estimator is None else estimator

    def _log_message(self, step_idx):
        if not self.verbose:
            return None
        name, _ = self.steps[step_idx]

        return "(step %d of %d) Processing %s" % (step_idx + 1, len(self.steps), name)

    def _default_routing(self, params):
        names = [name for name, _ in self.steps]
        step_params = {name: {} for name in names}
        remainder = set()
        for k, v in params.items():
            if v is not None:
                # XXX: not sure if we need to remove Nones
                name, prop = k.split("__", 1)
                try:
                    step_params[name][prop] = v
                except KeyError:
                    remainder.add(k)
        return [step_params[name] for name in names], remainder

    def _transform_pipeline(self, caller_name, X, params):
        step_params, remainder = self.router(params)
        if remainder:
            raise TypeError(
                "%s() got unexpected keyword arguments %r" % (caller_name, remainder)
            )
        feature_meta = params.get("feature_meta", None)
        if caller_name == "transform":
            step_iter = self._iter()
        elif caller_name == "inverse_transform":
            step_iter = reversed(list(self._iter()))
        else:
            step_iter = self._iter(with_final=False)
        for step_idx, _, transformer in step_iter:
            if (
                step_idx > 0
                and feature_meta is not None
                and "feature_meta" in step_params[step_idx]
            ):
                step_params[step_idx]["feature_meta"] = feature_meta
            X = transformer.transform(X, **step_params[step_idx])
            if feature_meta is not None:
                feature_meta = transform_feature_meta(transformer, feature_meta)
        if caller_name in ["transform", "inverse_transform"]:
            return X
        if "feature_meta" in step_params[-1]:
            step_params[-1]["feature_meta"] = feature_meta
        return X, step_params[-1]

    # Estimator interface

    def _fit(self, X, y=None, **fit_params):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        step_fit_params, remainder = self.router(fit_params)
        if remainder:
            raise TypeError("Got unexpected keyword arguments %r" % sorted(remainder))
        feature_meta = fit_params.get("feature_meta", None)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)

            if step_idx > 0 and "feature_meta" in step_fit_params[step_idx]:
                step_fit_params[step_idx]["feature_meta"] = feature_meta

            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **step_fit_params[step_idx],
            )

            if feature_meta is not None:
                feature_meta = transform_feature_meta(fitted_transformer, feature_meta)
                if X.shape[1] != feature_meta.shape[0]:
                    raise ValueError(
                        (
                            "X ({:d}) and feature_meta ({:d}) have "
                            "different feature dimensions"
                        ).format(X.shape[1], feature_meta.shape[0])
                    )

            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)

        if self._final_estimator == "passthrough":
            return X, {}

        if "feature_meta" in step_fit_params[-1]:
            step_fit_params[-1]["feature_meta"] = feature_meta

        return X, step_fit_params[-1]

    def fit(self, X, y=None, **fit_params):
        """Fit the model.

        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        Xt, fit_params = self._fit(X, y, **fit_params)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                self._final_estimator.fit(Xt, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator.

        Fits all the transformers one after the other and transform the
        data. Then uses `fit_transform` on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        Xt, fit_params = self._fit(X, y, **fit_params)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator == "passthrough":
                return Xt
            if hasattr(self._final_estimator, "fit_transform"):
                return self._final_estimator.fit_transform(Xt, y, **fit_params)
            else:
                return self._final_estimator.fit(Xt, y, **fit_params).transform(
                    Xt, **fit_params
                )

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

            .. versionadded:: 0.20

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the final estimator.
        """
        Xt, predict_params = self._transform_pipeline("predict", X, predict_params)
        return self.steps[-1][-1].predict(Xt, **predict_params)

    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        """Transform the data, and apply `fit_predict` with the final estimator.

        Call `fit_transform` of each transformer in the pipeline. The
        transformed data are finally passed to the final estimator that calls
        `fit_predict` method. Only valid if the final estimator implements
        `fit_predict`.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : ndarray
            Result of calling `fit_predict` on the final estimator.
        """
        Xt, fit_params = self._fit(X, y, **fit_params)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][-1].fit_predict(Xt, y, **fit_params)
        return y_pred

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        """Transform the data, and apply `predict_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_proba` method. Only valid if the final estimator implements
        `predict_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_proba_params : dict of string -> object
            Parameters to the `predict_proba` called at the end of all
            transformations in the pipeline.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_proba` on the final estimator.
        """
        Xt, predict_proba_params = self._transform_pipeline(
            "predict_proba", X, predict_proba_params
        )
        return self.steps[-1][-1].predict_proba(Xt, **predict_proba_params)

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X, **decision_function_params):
        """Transform the data, and apply `decision_function` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `decision_function` method. Only valid if the final estimator
        implements `decision_function`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : ndarray of shape (n_samples, n_classes)
            Result of calling `decision_function` on the final estimator.
        """
        Xt, decision_function_params = self._transform_pipeline(
            "decision_function", X, decision_function_params
        )
        return self.steps[-1][-1].decision_function(Xt, **decision_function_params)

    @available_if(_final_estimator_has("score_samples"))
    def score_samples(self, X, **score_sample_params):
        """Transform the data, and apply `score_samples` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score_samples` method. Only valid if the final estimator implements
        `score_samples`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            Result of calling `score_samples` on the final estimator.
        """
        Xt, score_sample_params = self._transform_pipeline(
            "score_samples", X, score_sample_params
        )
        return self.steps[-1][-1].score_samples(Xt, **score_sample_params)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        """Transform the data, and apply `predict_log_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_log_proba` method. Only valid if the final estimator
        implements `predict_log_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_log_proba_params : dict of string -> object
            Parameters to the ``predict_log_proba`` called at the end of all
            transformations in the pipeline.

        Returns
        -------
        y_log_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_log_proba` on the final estimator.
        """
        Xt, predict_log_proba_params = self._transform_pipeline(
            "predict_log_proba", X, predict_log_proba_params
        )
        return self.steps[-1][-1].predict_log_proba(Xt, **predict_log_proba_params)

    def _can_transform(self):
        return self._final_estimator == "passthrough" or hasattr(
            self._final_estimator, "transform"
        )

    @available_if(_can_transform)
    def transform(self, X, **transform_params):
        """Transform the data, and apply `transform` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.

        This also works where final estimator is `None` in which case all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        Xt = self._transform_pipeline("transform", X, transform_params)
        return Xt

    def _can_inverse_transform(self):
        return all(hasattr(t, "inverse_transform") for _, _, t in self._iter())

    @available_if(_can_inverse_transform)
    def inverse_transform(self, Xt, **inverse_transform_params):
        """Apply `inverse_transform` for each step in a reverse order.

        All estimators in the pipeline must support `inverse_transform`.

        Parameters
        ----------
        Xt : array-like of shape (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Inverse transformed data, that is, data in the original feature
            space.
        """
        Xt = self._transform_pipeline("inverse_transform", Xt, inverse_transform_params)
        return Xt

    @available_if(_final_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None, **transform_params):
        """Transform the data, and apply `score` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score` method. Only valid if the final estimator implements `score`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
            Result of calling `score` on the final estimator.
        """
        Xt, _ = self._transform_pipeline("score", X, transform_params)
        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        return self.steps[-1][-1].score(Xt, y, **score_params)

    @property
    def classes_(self):
        """The classes labels. Only exist if the last step is a classifier."""
        return self.steps[-1][1].classes_

    def _more_tags(self):
        # check if first estimator expects pairwise input
        return {"pairwise": _safe_tags(self.steps[0][1], "pairwise")}

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Transform input features using the pipeline.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        feature_names_out = input_features
        for _, name, transform in self._iter():
            if not hasattr(transform, "get_feature_names_out"):
                raise AttributeError(
                    "Estimator {} does not provide get_feature_names_out. "
                    "Did you mean to call pipeline[:-1].get_feature_names_out"
                    "()?".format(name)
                )
            feature_names_out = transform.get_feature_names_out(feature_names_out)
        return feature_names_out

    @property
    def n_features_in_(self):
        """Number of features seen during first step `fit` method."""
        # delegate to first step (which will call _check_is_fitted)
        return self.steps[0][1].n_features_in_

    @property
    def feature_names_in_(self):
        """Names of features seen during first step `fit` method."""
        # delegate to first step (which will call _check_is_fitted)
        return self.steps[0][1].feature_names_in_

    def __sklearn_is_fitted__(self):
        """Indicate whether pipeline has been fit."""
        try:
            # check if the last step of the pipeline is fitted
            # we only check the last step since if the last step is fit, it
            # means the previous steps should also be fit. This is faster than
            # checking if every step of the pipeline is fit.
            check_is_fitted(self.steps[-1][1])
            return True
        except NotFittedError:
            return False

    def _sk_visual_block_(self):
        _, estimators = zip(*self.steps)

        def _get_name(name, est):
            if est is None or est == "passthrough":
                return f"{name}: passthrough"
            # Is an estimator
            return f"{name}: {est.__class__.__name__}"

        names = [_get_name(name, est) for name, est in self.steps]
        name_details = [str(est) for est in estimators]
        return _VisualBlock(
            "serial",
            estimators,
            names=names,
            name_details=name_details,
            dash_wrapped=False,
        )


def transform_feature_meta(estimator, feature_meta):
    if isinstance(estimator, ColumnTransformer):
        transformed_feature_meta = None
        for _, trf_transformer, trf_columns in estimator.transformers_:
            if isinstance(trf_transformer, str) and trf_transformer == "drop":
                trf_feature_meta = feature_meta.iloc[
                    ~feature_meta.index.isin(trf_columns)
                ]
            elif (
                isinstance(trf_columns, slice)
                and (
                    isinstance(trf_columns.start, str)
                    or isinstance(trf_columns.stop, str)
                )
            ) or isinstance(trf_columns[0], str):
                trf_feature_meta = feature_meta.loc[trf_columns]
            else:
                trf_feature_meta = feature_meta.iloc[trf_columns]
            if isinstance(trf_transformer, Pipeline):
                for transformer in trf_transformer:
                    trf_feature_meta = transform_feature_meta(
                        transformer, trf_feature_meta
                    )
            if transformed_feature_meta is None:
                transformed_feature_meta = trf_feature_meta
            else:
                transformed_feature_meta = pd.concat(
                    [transformed_feature_meta, trf_feature_meta], axis=0
                )
    elif hasattr(estimator, "get_support"):
        transformed_feature_meta = feature_meta.loc[estimator.get_support()]
    # TODO: fix logic to handle all types of feature changing estimators
    elif hasattr(estimator, "get_feature_names"):
        transformed_feature_meta = None
        feature_names = estimator.get_feature_names_out().astype(str)
        for feature_name in feature_meta.index:
            f_feature_meta = pd.concat(
                [feature_meta.loc[[feature_name]]]
                * np.sum(np.char.startswith(feature_names, "{}_".format(feature_name))),
                axis=0,
                ignore_index=False,
            )
            if transformed_feature_meta is None:
                transformed_feature_meta = f_feature_meta
            else:
                transformed_feature_meta = pd.concat(
                    [transformed_feature_meta, f_feature_meta],
                    axis=0,
                    ignore_index=False,
                )
        transformed_feature_meta = transformed_feature_meta.set_index(feature_names)
    else:
        transformed_feature_meta = feature_meta
    return transformed_feature_meta


def _name_estimators(estimators):
    """Generate names for estimators."""

    names = [
        estimator if isinstance(estimator, str) else type(estimator).__name__.lower()
        for estimator in estimators
    ]
    namecount = defaultdict(int)
    for est, name in zip(estimators, names):
        namecount[name] += 1

    for k, v in list(namecount.items()):
        if v == 1:
            del namecount[k]

    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1

    return list(zip(names, estimators))


def make_extended_pipeline(*steps, memory=None, verbose=False, param_routing=None):
    """Construct a :class:`Pipeline` from the given estimators.

    This is a shorthand for the :class:`Pipeline` constructor; it does not
    require, and does not permit, naming the estimators. Instead, their names
    will be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of Estimator objects
        List of the scikit-learn estimators that are chained together.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Returns
    -------
    p : Pipeline
        Returns a scikit-learn :class:`Pipeline` object.

    See Also
    --------
    Pipeline : Class for creating a pipeline of transforms with a final
        estimator.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('gaussiannb', GaussianNB())])
    """
    return ExtendedPipeline(
        _name_estimators(steps),
        memory=memory,
        verbose=verbose,
        param_routing=param_routing,
    )


def _transform_one(
    transformer, X, y, weight, message_clsname="", message=None, **transform_params
):
    res = transformer.transform(X, **transform_params)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight


def _fit_transform_one(
    transformer, X, y, weight, message_clsname="", message=None, **fit_params
):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, "fit_transform"):
            res = transformer.fit_transform(X, y, **fit_params)
        else:
            res = transformer.fit(X, y, **fit_params).transform(X, **fit_params)

    if weight is None:
        return res, transformer
    return res * weight, transformer


def _fit_one(transformer, X, y, weight, message_clsname="", message=None, **fit_params):
    """
    Fits ``transformer`` to ``X`` and ``y``.
    """
    with _print_elapsed_time(message_clsname, message):
        return transformer.fit(X, y, **fit_params)


class ExtendedFeatureUnion(ExtendedTransformerMixin, FeatureUnion):
    """Concatenates results of multiple transformer objects.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer, removed by
    setting to 'drop' or disabled by setting to 'passthrough' (features are
    passed without transformation).

    Read more in the :ref:`User Guide <feature_union>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    transformer_list : list of (str, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer. The transformer can
        be 'drop' for it to be ignored or can be 'passthrough' for features to
        be passed unchanged.

        .. versionadded:: 1.1
           Added the option `"passthrough"`.

        .. versionchanged:: 0.22
           Deprecated `None` as a transformer in favor of 'drop'.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
        Raises ValueError if key not present in ``transformer_list``.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying first transformer in `transformer_list` exposes such an
        attribute when fit.

        .. versionadded:: 0.24

    See Also
    --------
    make_union : Convenience function for simplified feature union
        construction.

    Examples
    --------
    >>> from sklearn.pipeline import FeatureUnion
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> union = FeatureUnion([("pca", PCA(n_components=1)),
    ...                       ("svd", TruncatedSVD(n_components=2))])
    >>> X = [[0., 1., 3], [2., 2., 5]]
    >>> union.fit_transform(X)
    array([[ 1.5       ,  3.0...,  0.8...],
           [-1.5       ,  5.7..., -0.4...]])
    """

    _required_parameters = ["transformer_list"]

    def __init__(
        self,
        transformer_list,
        *,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        param_routing=None,
    ):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.param_routing = param_routing
        self.router = check_routing(
            self.param_routing,
            [[name, "*"] for name, _ in self.transformer_list],
            self._default_routing,
        )

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `transformer_list` of the
        `FeatureUnion`.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("transformer_list", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``. Note that
        you can directly set the parameters of the estimators contained in
        `tranformer_list`.

        Parameters
        ----------
        **kwargs : dict
            Parameters of this estimator or parameters of estimators contained
            in `transform_list`. Parameters of the transformers may be set
            using its name and the parameter name separated by a '__'.

        Returns
        -------
        self : object
            FeatureUnion class instance.
        """
        self._set_params("transformer_list", **kwargs)
        if "param_routing" in kwargs:
            self.router = check_routing(
                self.param_routing,
                [[name, "*"] for name, _ in self.transformer_list],
                self._default_routing,
            )
        return self

    def _validate_transformers(self):
        names, transformers = zip(*self.transformer_list)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t in ("drop", "passthrough"):
                continue
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
                t, "transform"
            ):
                raise TypeError(
                    "All estimators should implement fit and "
                    "transform. '%s' (type %s) doesn't" % (t, type(t))
                )

    def _validate_transformer_weights(self):
        if not self.transformer_weights:
            return

        transformer_names = set(name for name, _ in self.transformer_list)
        for name in self.transformer_weights:
            if name not in transformer_names:
                raise ValueError(
                    f'Attempting to weight transformer "{name}", '
                    "but it is not present in transformer_list."
                )

    def _iter(self):
        """
        Generate (name, trans, weight) tuples excluding None and
        'drop' transformers.
        """

        get_weight = (self.transformer_weights or {}).get

        for name, trans in self.transformer_list:
            if trans == "drop":
                continue
            if trans == "passthrough":
                trans = FunctionTransformer()
            yield (name, trans, get_weight(name))

    def _default_routing(self, params):
        names = [name for name, _ in self.transformer_list]
        transformer_params = {name: {} for name in names}
        remainder = set()
        for k, v in params.items():
            if v is not None:
                # XXX: not sure if we need to remove Nones
                name, prop = k.split("__", 1)
                try:
                    transformer_params[name][prop] = v
                except KeyError:
                    remainder.add(k)
        return [transformer_params[name] for name in names], remainder

    @deprecated(
        "get_feature_names is deprecated in 1.0 and will be removed "
        "in 1.2. Please use get_feature_names_out instead."
    )
    def get_feature_names(self):
        """Get feature names from all transformers.

        Returns
        -------
        feature_names : list of strings
            Names of the features produced by transform.
        """
        feature_names = []
        for name, trans, weight in self._iter():
            if not hasattr(trans, "get_feature_names"):
                raise AttributeError(
                    "Transformer %s (type %s) does not provide get_feature_names."
                    % (str(name), type(trans).__name__)
                )
            feature_names.extend([name + "__" + f for f in trans.get_feature_names()])
        return feature_names

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        feature_names = []
        for name, trans, _ in self._iter():
            if not hasattr(trans, "get_feature_names_out"):
                raise AttributeError(
                    "Transformer %s (type %s) does not provide get_feature_names_out."
                    % (str(name), type(trans).__name__)
                )
            feature_names.extend(
                [f"{name}__{f}" for f in trans.get_feature_names_out(input_features)]
            )
        return np.asarray(feature_names, dtype=object)

    def fit(self, X, y=None, **fit_params):
        """Fit all transformers using X.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data, used to fit transformers.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **fit_params : dict, default=None
            Parameters to pass to the fit method of the estimator.

        Returns
        -------
        self : object
            FeatureUnion class instance.
        """
        transformers = self._parallel_func(X, y, fit_params, _fit_one)
        if not transformers:
            # All transformers are None
            return self

        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **fit_params : dict, default=None
            Parameters to pass to the fit method of the estimator.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        return self._hstack(Xs)

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return "(step %d of %d) Processing %s" % (idx, total, name)

    def _parallel_func(self, X, y, fit_params, func):
        """Runs func in parallel on X and y"""
        transformer_fit_params, remainder = self.router(fit_params)
        if remainder:
            raise TypeError(
                "%s got unexpected keyword arguments %r" % (func, sorted(remainder))
            )

        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        self._validate_transformer_weights()
        transformers = list(self._iter())

        return Parallel(n_jobs=self.n_jobs)(
            delayed(func)(
                transformer,
                X,
                y,
                weight,
                message_clsname="FeatureUnion",
                message=self._log_message(name, idx, len(transformers)),
                **fit_params,
            )
            for idx, ((name, transformer, weight), fit_params) in (
                enumerate(zip(transformers, transformer_fit_params), 1)
            )
        )

    def transform(self, X, **transform_params):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        transformer_transform_params, remainder = self.router(transform_params)
        if remainder:
            raise TypeError("Got unexpected keyword arguments %r" % sorted(remainder))

        transformers = list(self._iter())

        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(transformer, X, None, weight, **transform_params)
            for _, ((_, transformer, weight), transform_params) in zip(
                transformers, transformer_transform_params
            )
        )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        return self._hstack(Xs)

    def _hstack(self, Xs):
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def _update_transformer_list(self, transformers):
        transformers = iter(transformers)
        self.transformer_list[:] = [
            (name, old if old == "drop" else next(transformers))
            for name, old in self.transformer_list
        ]

    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""

        # X is passed to all transformers so we just delegate to the first one
        return self.transformer_list[0][1].n_features_in_

    def __sklearn_is_fitted__(self):
        # Delegate whether feature union was fitted
        for _, transformer, _ in self._iter():
            check_is_fitted(transformer)
        return True

    def _sk_visual_block_(self):
        names, transformers = zip(*self.transformer_list)
        return _VisualBlock("parallel", transformers, names=names)


def make_extended_union(*transformers, n_jobs=None, verbose=False, param_routing=None):
    """
    Construct a FeatureUnion from the given transformers.

    This is a shorthand for the FeatureUnion constructor; it does not require,
    and does not permit, naming the transformers. Instead, they will be given
    names automatically based on their types. It also does not allow weighting.

    Parameters
    ----------
    *transformers : list of estimators

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Returns
    -------
    f : FeatureUnion

    See Also
    --------
    FeatureUnion : Class for concatenating the results of multiple transformer
        objects.

    Examples
    --------
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> from sklearn.pipeline import make_union
    >>> make_union(PCA(), TruncatedSVD())
     FeatureUnion(transformer_list=[('pca', PCA()),
                                   ('truncatedsvd', TruncatedSVD())])
    """
    return ExtendedFeatureUnion(
        _name_estimators(transformers),
        n_jobs=n_jobs,
        verbose=verbose,
        param_routing=param_routing,
    )
