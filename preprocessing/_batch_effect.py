import os
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_memory

from ..base import ExtendedTransformerMixin

if "limma_removeba_it" not in ro.globalenv:
    r_base = importr("base")
    r_base.source(os.path.dirname(__file__) + "/_batch_effect.R")
r_limma_removeba_fit = ro.globalenv["limma_removeba_fit"]
r_limma_removeba_transform = ro.globalenv["limma_removeba_transform"]
r_stica_removeba_fit = ro.globalenv["stica_removeba_fit"]
r_stica_removeba_transform = ro.globalenv["stica_removeba_transform"]
r_bapred_removeba_fit = ro.globalenv["bapred_removeba_fit"]
r_bapred_removeba_transform = ro.globalenv["bapred_removeba_transform"]


def stica_removeba_fit(X, sample_meta, method, k, alpha):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        xt, params = r_stica_removeba_fit(
            X, sample_meta, method=method, k=k, alpha=alpha
        )
        return np.array(xt, dtype=float), params


def stica_removeba_transform(X, params):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        return np.array(r_stica_removeba_transform(X, params), dtype=float)


class LimmaBatchEffectRemover(ExtendedTransformerMixin, BaseEstimator):
    """limma removeBatchEffect transformer for log-transformed expression data

    Parameters
    ----------
    preserve_design : bool (default = True)
        Whether batch effect correction should protect target design from
        being removed.

    Attributes
    ----------
    batch_adj_ : pandas.DataFrame, shape (n_features, n_batches)
        removeBatchEffect feature batch adjustments
    """

    def __init__(self, preserve_design=True):
        self.preserve_design = preserve_design

    def fit(self, X, y, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training log-transformed data matrix.

        y : ignored

        sample_meta : pandas.DataFrame, pandas.Series \
            shape = (n_samples, n_metadata)
            Training sample metadata.
        """
        X = self._validate_data(X)
        if sample_meta is None:
            sample_meta = ro.NULL
        with (
            ro.default_converter + numpy2ri.converter + pandas2ri.converter
        ).context():
            self.batch_adj_ = r_limma_removeba_fit(
                X, sample_meta, preserve_design=self.preserve_design
            )
        return self

    def transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input log-transformed data matrix.

        sample_meta : pandas.DataFrame, pandas.Series \
            shape = (n_samples, n_metadata)
            Sample metadata.

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            Batched corrected log-transformed input data matrix.
        """
        check_is_fitted(self, "batch_adj_")
        X = self._validate_data(X, dtype=None, reset=False)
        with (
            ro.default_converter + numpy2ri.converter + pandas2ri.converter
        ).context():
            X = np.array(
                r_limma_removeba_transform(X, sample_meta, self.batch_adj_), dtype=float
            )
        return X

    def inverse_transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input batched corrected log-transformed data matrix.

        sample_meta : Ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError("inverse_transform not implemented.")


class stICABatchEffectRemover(ExtendedTransformerMixin, BaseEstimator):
    """stICA batch effect removal transformer

    Parameters
    ----------
    k : int (default = 20)
        Number of components to estimate

    alpha : float (default = 0.5)
        Number between 0 and 1 specifying the trade-off between spatial ICA
        (alpha = 0) and temporal ICA (alpha = 1)

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    params_ : R/rpy2 list
        Training data parameter settings
    """

    def __init__(self, k=20, alpha=0.5, memory=None):
        self.k = k
        self.alpha = alpha
        self.memory = memory

    def fit(self, X, y, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data matrix.

        y : ignored

        sample_meta : pandas.DataFrame, pandas.Series \
            shape = (n_samples, n_metadata)
            Training sample metadata.
        """
        X = self._validate_data(X)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        self._Xt, self.params_ = memory.cache(stica_removeba_fit)(
            X, sample_meta, method="stICA", k=self.k, alpha=self.alpha
        )
        return self

    def transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input data matrix.

        sample_meta : ignored

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            Batched corrected log-transformed input data matrix.
        """
        check_is_fitted(self, "_Xt")
        X = self._validate_data(X, dtype=None, reset=False)
        if hasattr(self, "_train_done"):
            memory = check_memory(self.memory)
            X = memory.cache(stica_removeba_transform)(X, self.params_)
        else:
            X = self._Xt
            self._train_done = True
        return X

    def inverse_transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input batched corrected data matrix.

        sample_meta : Ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError("inverse_transform not implemented.")


class SVDBatchEffectRemover(ExtendedTransformerMixin, BaseEstimator):
    """Singular value decomposition (SVD) batch effect removal transformer

    Parameters
    ----------
    k : int (default = 20)
        Number of components to estimate

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    params_ : R/rpy2 list
        Training data parameter settings
    """

    def __init__(self, k=20, memory=None):
        self.k = k
        self.memory = memory

    def fit(self, X, y, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data matrix.

        y : ignored

        sample_meta : pandas.DataFrame, pandas.Series \
            shape = (n_samples, n_metadata)
            Training sample metadata.
        """
        X = self._validate_data(X)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        self._Xt, self.params_ = memory.cache(stica_removeba_fit)(
            X, sample_meta, method="SVD", k=self.k
        )
        return self

    def transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input data matrix.

        sample_meta : ignored

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            Batched corrected log-transformed input data matrix.
        """
        check_is_fitted(self, "_Xt")
        X = self._validate_data(X, dtype=None, reset=False)
        if hasattr(self, "_train_done"):
            memory = check_memory(self.memory)
            X = memory.cache(stica_removeba_transform)(X, self.params_)
        else:
            X = self._Xt
            self._train_done = True
        return X

    def inverse_transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input batched corrected data matrix.

        sample_meta : Ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError("inverse_transform not implemented.")
