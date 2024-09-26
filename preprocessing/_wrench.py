import os
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_memory

from ..base import ExtendedTransformerMixin

r_base = importr("base")
if "wrench_fit" not in ro.globalenv:
    r_base.source(os.path.dirname(__file__) + "/_wrench.R")
r_wrench_fit = ro.globalenv["wrench_fit"]
r_wrench_cpm_transform = ro.globalenv["wrench_cpm_transform"]


def wrench_fit(X, sample_meta, ref_type):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        res = r_wrench_fit(X, sample_meta, ref_type=ref_type)
        return (
            np.array(res["nzrows"], dtype=bool),
            np.array(res["qref"], dtype=float),
            np.array(res["s2"], dtype=float),
        )


def wrench_cpm_transform(X, sample_meta, nzrows, qref, s2, est_type, log, prior_count):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        return r_wrench_cpm_transform(
            X,
            sample_meta,
            nzrows=nzrows,
            qref=qref,
            s2=s2,
            est_type=est_type,
            log=log,
            prior_count=prior_count,
        )


class WrenchCPM(ExtendedTransformerMixin, BaseEstimator):
    """Wrench normalization and CPM transform for sparse, under-sampled count data

    Parameters
    ----------
    est_type : str (default = "w.marg.mean")
        Wrench estimator type.

    ref_type : str (default = "sw.means")
        Wrench reference vector type.

    log : bool (default = True)
        Whether to return log2 transformed values.

    prior_count : float (default = 1)
        Average count to add to each observation to avoid taking log of zero.
        Larger values produce stronger moderation of low counts and more
        shrinkage of the corresponding log fold changes.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    nzrows_ : array, shape (n_features, )
        Non-zero count feature mask

    qref_ : array, shape (n_nonzero_features,)
        Wrench reference vector

    s2_ : array, shape (n_nonzero_features,)
        Wrench variance estimates for logged feature-wise counts
    """

    def __init__(
        self,
        est_type="w.marg.mean",
        ref_type="sw.means",
        log=True,
        prior_count=1,
        memory=None,
    ):
        self.est_type = est_type
        self.ref_type = ref_type
        self.log = log
        self.prior_count = prior_count
        self.memory = memory

    def fit(self, X, y, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        y : ignored

        sample_meta : pandas.DataFrame, pandas.Series (default = None) \
            shape = (n_samples, n_metadata)
            Training sample metadata.
        """
        X = self._validate_data(X, dtype=int)
        self.nzrows_, self.qref_, self.s2_ = wrench_fit(
            X, sample_meta, ref_type=self.ref_type
        )
        return self

    def transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : pandas.DataFrame, pandas.Series (default = None) \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            edgeR TMM normalized CPM transformed data matrix.
        """
        check_is_fitted(self, "nzrows_")
        X = self._validate_data(X, dtype=int, reset=False)
        memory = check_memory(self.memory)
        X = memory.cache(wrench_cpm_transform)(
            X,
            sample_meta,
            nzrows=self.nzrows_,
            qref=self.qref_,
            s2=self.s2_,
            est_type=self.est_type,
            log=self.log,
            prior_count=self.prior_count,
        )
        return X

    def inverse_transform(self, X, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta : ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _more_tags(self):
        return {"requires_positive_X": True}
