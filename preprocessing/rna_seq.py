import os
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted, check_memory
from ..base import TransformerMixin

numpy2ri.deactivate()
pandas2ri.deactivate()
numpy2ri.activate()
pandas2ri.activate()

if 'deseq2_vst_fit' not in robjects.globalenv:
    r_base = importr('base')
    r_base.source(os.path.dirname(__file__) + '/rna_seq.R')
r_deseq2_vst_fit = robjects.globalenv['deseq2_vst_fit']
r_deseq2_vst_transform = robjects.globalenv['deseq2_vst_transform']
r_edger_tmm_logcpm_fit = robjects.globalenv['edger_tmm_logcpm_fit']
r_edger_tmm_logcpm_transform = robjects.globalenv['edger_tmm_logcpm_transform']


def deseq2_vst_fit(X, y, sample_meta, blind, fit_type, model_batch):
    xt, gm, sf, df = r_deseq2_vst_fit(
        X, y, sample_meta=sample_meta, blind=blind, fit_type=fit_type,
        model_batch=model_batch)
    return (np.array(xt, dtype=float), np.array(gm, dtype=float),
            np.array(sf, dtype=float), df)


def deseq2_vst_transform(X, geo_means, size_factors, disp_func, fit_type):
    return np.array(r_deseq2_vst_transform(
        X, geo_means=geo_means, size_factors=size_factors, disp_func=disp_func,
        fit_type=fit_type), dtype=float)


def edger_tmm_logcpm_fit(X, prior_count):
    return np.array(r_edger_tmm_logcpm_fit(X, prior_count=prior_count),
                    dtype=float)


def edger_tmm_logcpm_transform(X, ref_sample, prior_count):
    return np.array(r_edger_tmm_logcpm_transform(
        X, ref_sample=ref_sample, prior_count=prior_count), dtype=float)


class DESeq2RLEVST(BaseEstimator, TransformerMixin):
    """DESeq2 median-of-ratios normalization and VST transformation for count
    data

    Parameters
    ----------
    blind : bool (default = False)
        varianceStabilizingTransformation blind option.

    fit_type : str (default = "local")
        estimateDispersions and varianceStabilizingTransformation fitType
        option.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and Batch column
        exists.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    geo_means_ : array, shape (n_features,)
        Feature geometric means.

    size_factors_ : array, shape (n_features,)
        RLE normalization size factors.

    disp_func_ : R/rpy2 function
        RLE normalization dispersion function.
    """

    def __init__(self, blind=False, fit_type='local', model_batch=False,
                 memory=None):
        self.blind = blind
        self.fit_type = fit_type
        self.model_batch = model_batch
        self.memory = memory

    def fit(self, X, y, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series (default = None) \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=int)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = robjects.NULL
        (self._vst_data, self.geo_means_, self.size_factors_,
         self.disp_func_) = memory.cache(deseq2_vst_fit)(
             X, y, sample_meta=sample_meta, blind=self.blind,
             fit_type=self.fit_type, model_batch=self.model_batch)
        return self

    def transform(self, X, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : Ignored.

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            DESeq2 median-of-ratios normalized VST transformed counts data
            matrix.
        """
        check_is_fitted(self, '_vst_data')
        X = check_array(X, dtype=int)
        if hasattr(self, '_train_done'):
            memory = check_memory(self.memory)
            X = memory.cache(deseq2_vst_transform)(
                X, geo_means=self.geo_means_, size_factors=self.size_factors_,
                disp_func=self.disp_func_, fit_type=self.fit_type)
        else:
            X = self._vst_data
            self._train_done = True
        return X

    def inverse_transform(self, X, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta : Ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError('inverse_transform not implemented.')


class EdgeRTMMLogCPM(BaseEstimator, TransformerMixin):
    """edgeR TMM normalization and log-CPM transformation for count data

    Parameters
    ----------
    prior_count : int (default = 1)
        Average count to add to each observation to avoid taking log of zero.
        Larger values for produce stronger moderation of the values for low
        counts and more shrinkage of the corresponding log fold changes.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    ref_sample_ : array, shape (n_features,)
        TMM normalization reference sample feature vector.
    """

    def __init__(self, prior_count=1, memory=None):
        self.prior_count = prior_count
        self.memory = memory

    def fit(self, X, y=None, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        y : ignored

        sample_meta: ignored
        """
        X = check_array(X, dtype=int)
        memory = check_memory(self.memory)
        self._log_cpms, self.ref_sample_ = memory.cache(edger_tmm_logcpm_fit)(
            X, prior_count=self.prior_count)
        return self

    def transform(self, X, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : ignored

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            edgeR TMM normalized log-CPM transformed counts data matrix.
        """
        check_is_fitted(self, '_log_cpms')
        X = check_array(X, dtype=int)
        if hasattr(self, '_train_done'):
            memory = check_memory(self.memory)
            X = memory.cache(edger_tmm_logcpm_transform)(
                X, ref_sample=self.ref_sample_, prior_count=self.prior_count)
        else:
            X = self._log_cpms
            self._train_done = True
        return X

    def inverse_transform(self, X, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta: ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError('inverse_transform not implemented.')
