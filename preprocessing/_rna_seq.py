import os
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y
from ..base import ExtendedTransformerMixin
from ..utils.validation import check_is_fitted, check_memory

numpy2ri.deactivate()
pandas2ri.deactivate()
numpy2ri.activate()
pandas2ri.activate()

if 'deseq2_vst_fit' not in robjects.globalenv:
    r_base = importr('base')
    r_base.source(os.path.dirname(__file__) + '/_rna_seq.R')
r_deseq2_vst_fit = robjects.globalenv['deseq2_vst_fit']
r_deseq2_vst_transform = robjects.globalenv['deseq2_vst_transform']
r_edger_tmm_logcpm_fit = robjects.globalenv['edger_tmm_logcpm_fit']
r_edger_tmm_logcpm_transform = robjects.globalenv['edger_tmm_logcpm_transform']
r_edger_tmm_tpm_fit = robjects.globalenv['edger_tmm_tpm_fit']
r_edger_tmm_tpm_transform = robjects.globalenv['edger_tmm_tpm_transform']


def deseq2_vst_fit(X, y, sample_meta, fit_type, blind, model_batch,
                   is_classif):
    xt, gm, df = r_deseq2_vst_fit(
        X, y, sample_meta=sample_meta, fit_type=fit_type, blind=blind,
        model_batch=model_batch, is_classif=is_classif)
    return np.array(xt, dtype=float), np.array(gm, dtype=float), df


def deseq2_vst_transform(X, geo_means, disp_func):
    return np.array(r_deseq2_vst_transform(
        X, geo_means=geo_means, disp_func=disp_func), dtype=float)


def edger_tmm_logcpm_fit(X, prior_count):
    xt, rs = r_edger_tmm_logcpm_fit(X, prior_count=prior_count)
    return np.array(xt, dtype=float), np.array(rs, dtype=float)


def edger_tmm_logcpm_transform(X, ref_sample, prior_count):
    return np.array(r_edger_tmm_logcpm_transform(
        X, ref_sample=ref_sample, prior_count=prior_count), dtype=float)


def edger_tmm_tpm_fit(X, feature_meta, meta_col):
    xt, rs = r_edger_tmm_tpm_fit(X, feature_meta, meta_col=meta_col)
    return np.array(xt, dtype=float), np.array(rs, dtype=float)


def edger_tmm_tpm_transform(X, feature_meta, ref_sample, meta_col):
    return np.array(r_edger_tmm_tpm_transform(
        X, feature_meta, ref_sample=ref_sample, meta_col=meta_col),
                    dtype=float)


class DESeq2RLEVST(ExtendedTransformerMixin, BaseEstimator):
    """DESeq2 median-of-ratios normalization and VST transformation for count
    data

    Parameters
    ----------
    fit_type : str (default = "parametric")
        estimateDispersions fitType option.

    blind : bool (default = False)
        varianceStabilizingTransformation blind option.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and Batch column
        exists.

    is_classif : bool (default = True)
        Whether this is a classification design.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    geo_means_ : array, shape (n_features,)
        Feature geometric means.

    disp_func_ : R/rpy2 function
        RLE normalization dispersion function.
    """

    def __init__(self, fit_type='parametric', blind=False, model_batch=False,
                 is_classif=True, memory=None):
        self.fit_type = fit_type
        self.blind = blind
        self.model_batch = model_batch
        self.is_classif = is_classif
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
        self._vst_data, self.geo_means_, self.disp_func_ = (
            memory.cache(deseq2_vst_fit)(
                X, y, sample_meta=sample_meta, fit_type=self.fit_type,
                blind=self.blind, model_batch=self.model_batch,
                is_classif=self.is_classif))
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
            DESeq2 median-of-ratios normalized VST transformed data matrix.
        """
        check_is_fitted(self, '_vst_data')
        X = check_array(X, dtype=int)
        if hasattr(self, '_train_done'):
            memory = check_memory(self.memory)
            X = memory.cache(deseq2_vst_transform)(
                X, geo_means=self.geo_means_, disp_func=self.disp_func_)
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

    def _more_tags(self):
        return {'requires_positive_X': True}


class EdgeRTMMLogCPM(ExtendedTransformerMixin, BaseEstimator):
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
            edgeR TMM normalized log-CPM transformed data matrix.
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

        sample_meta : ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError('inverse_transform not implemented.')

    def _more_tags(self):
        return {'requires_positive_X': True}


class EdgeRTMMTPM(ExtendedTransformerMixin, BaseEstimator):
    """edgeR TMM normalization and TPM transformation for count data

    Parameters
    ----------
    meta_col : str (default = "Length")
        Feature metadata column name holding CDS lengths.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    ref_sample_ : array, shape (n_features,)
        TMM normalization reference sample feature vector.
    """

    def __init__(self, meta_col='Length', memory=None):
        self.meta_col = meta_col
        self.memory = memory

    def fit(self, X, y, feature_meta, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        y : ignored

        feature_meta : pandas.DataFrame, pandas.Series \
            shape = (n_features, n_metadata)
            Feature metadata.

        sample_meta: ignored
        """
        X = check_array(X, dtype=int)
        self._check_params(X, y, feature_meta)
        memory = check_memory(self.memory)
        self._tpms, self.ref_sample_ = memory.cache(edger_tmm_tpm_fit)(
            X, feature_meta, meta_col=self.meta_col)
        return self

    def transform(self, X, feature_meta, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        feature_meta : pandas.DataFrame, pandas.Series \
            shape = (n_features, n_metadata)
            Feature metadata.

        sample_meta : ignored

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            edgeR TMM normalized TPM transformed data matrix.
        """
        check_is_fitted(self, '_tpms')
        X = check_array(X, dtype=int)
        if hasattr(self, '_train_done'):
            memory = check_memory(self.memory)
            X = memory.cache(edger_tmm_tpm_transform)(
                X, feature_meta, ref_sample=self.ref_sample_,
                meta_col=self.meta_col)
        else:
            X = self._tpms
            self._train_done = True
        return X

    def inverse_transform(self, X, feature_meta=None, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        feature_meta : ignored

        sample_meta : ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError('inverse_transform not implemented.')

    def _check_params(self, X, y, feature_meta):
        if X.shape[1] != feature_meta.shape[0]:
            raise ValueError('X ({:d}) and feature_meta ({:d}) have '
                             'different feature dimensions'
                             .format(X.shape[1], feature_meta.shape[0]))
        if self.meta_col not in feature_meta.columns:
            raise ValueError('{} feature_meta column does not exist.'
                             .format(self.meta_col))

    def _more_tags(self):
        return {'requires_positive_X': True}
