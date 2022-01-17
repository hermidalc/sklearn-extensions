import os
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y
from ..base import ExtendedTransformerMixin
from ..utils.validation import check_is_fitted

numpy2ri.deactivate()
pandas2ri.deactivate()
numpy2ri.activate()
pandas2ri.activate()

r_base = importr('base')
if 'deseq2_vst_fit' not in robjects.globalenv:
    r_base.source(os.path.dirname(__file__) + '/_rna_seq.R')
r_deseq2_vst_fit = robjects.globalenv['deseq2_vst_fit']
r_deseq2_vst_transform = robjects.globalenv['deseq2_vst_transform']
r_edger_tmm_fit = robjects.globalenv['edger_tmm_fit']
r_edger_tmm_logcpm_transform = robjects.globalenv['edger_tmm_logcpm_transform']
r_edger_tmm_logtpm_transform = robjects.globalenv['edger_tmm_logtpm_transform']


def deseq2_vst_fit(X, y, sample_meta, fit_type, model_batch, is_classif):
    gm, df = r_deseq2_vst_fit(
        X, y, sample_meta=sample_meta, fit_type=fit_type,
        model_batch=model_batch, is_classif=is_classif)
    return np.array(gm, dtype=float), df


class DESeq2RLEVST(ExtendedTransformerMixin, BaseEstimator):
    """DESeq2 median-of-ratios normalization and VST transformation for count
    data

    Parameters
    ----------
    fit_type : str (default = "parametric")
        estimateDispersions fitType option.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and Batch column
        exists.

    is_classif : bool (default = True)
        Whether this is a classification design.

    Attributes
    ----------
    geo_means_ : array, shape (n_features,)
        Feature geometric means.

    disp_func_ : R/rpy2 function
        RLE normalization dispersion function.
    """

    def __init__(self, fit_type='parametric', model_batch=False,
                 is_classif=True):
        self.fit_type = fit_type
        self.model_batch = model_batch
        self.is_classif = is_classif

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
        if sample_meta is None:
            sample_meta = robjects.NULL
        self.geo_means_, self.disp_func_ = deseq2_vst_fit(
                X, y, sample_meta=sample_meta, fit_type=self.fit_type,
                model_batch=self.model_batch, is_classif=self.is_classif)
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
        check_is_fitted(self, 'geo_means_')
        X = check_array(X, dtype=int)
        X = np.array(r_deseq2_vst_transform(
            X, geo_means=self.geo_means_, disp_func=self.disp_func_),
                     dtype=float)
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
    prior_count : float (default = 2)
        Average count to add to each observation to avoid taking log of zero.
        Larger values for produce stronger moderation of the values for low
        counts and more shrinkage of the corresponding log fold changes.

    Attributes
    ----------
    ref_sample_ : array, shape (n_features,)
        TMM normalization reference sample feature vector.
    """

    def __init__(self, prior_count=2):
        self.prior_count = prior_count

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
        self.ref_sample_ = np.array(r_edger_tmm_fit(X), dtype=int)
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
        check_is_fitted(self, 'ref_sample_')
        X = check_array(X, dtype=int)
        X = np.array(r_edger_tmm_logcpm_transform(
            X, ref_sample=self.ref_sample_, prior_count=self.prior_count),
                     dtype=float)
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


class EdgeRTMMLogTPM(ExtendedTransformerMixin, BaseEstimator):
    """edgeR TMM normalization and log-TPM transformation for count data

    Parameters
    ----------
    prior_count : float (default = 2)
        Average count to add to each observation to avoid taking log of zero.
        Larger values for produce stronger moderation of the values for low
        counts and more shrinkage of the corresponding log fold changes.

    meta_col : str (default = "Length")
        Feature metadata column name holding CDS lengths.

    Attributes
    ----------
    ref_sample_ : array, shape (n_features,)
        TMM normalization reference sample feature vector.
    """

    def __init__(self, prior_count=2, meta_col='Length'):
        self.prior_count = prior_count
        self.meta_col = meta_col

    def fit(self, X, y, feature_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        y : ignored

        feature_meta : ignored
        """
        X = check_array(X, dtype=int)
        self.ref_sample_ = np.array(r_edger_tmm_fit(X), dtype=int)
        return self

    def transform(self, X, feature_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        feature_meta : pandas.DataFrame, pandas.Series \
            shape = (n_features, n_metadata)
            Feature metadata.

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            edgeR TMM normalized TPM transformed data matrix.
        """
        check_is_fitted(self, 'ref_sample_')
        X = check_array(X, dtype=int)
        X = np.array(r_edger_tmm_logtpm_transform(
            X, feature_meta, ref_sample=self.ref_sample_,
            prior_count=self.prior_count, meta_col=self.meta_col), dtype=float)
        return X

    def inverse_transform(self, X, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        feature_meta : ignored

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
