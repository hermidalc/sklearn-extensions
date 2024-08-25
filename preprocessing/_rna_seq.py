import os
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_memory

from ..base import ExtendedTransformerMixin

numpy2ri.deactivate()
pandas2ri.deactivate()
numpy2ri.activate()
pandas2ri.activate()

r_base = importr("base")
if "deseq2_norm_fit" not in robjects.globalenv:
    r_base.source(os.path.dirname(__file__) + "/_rna_seq.R")
r_deseq2_norm_fit = robjects.globalenv["deseq2_norm_fit"]
r_deseq2_norm_vst_transform = robjects.globalenv["deseq2_norm_vst_transform"]
r_edger_tmm_fit = robjects.globalenv["edger_tmm_fit"]
r_edger_tmm_cpm_transform = robjects.globalenv["edger_tmm_cpm_transform"]
r_edger_tmm_tpm_transform = robjects.globalenv["edger_tmm_tpm_transform"]


def deseq2_norm_fit(X, y, sample_meta, norm_type, fit_type, is_classif, model_batch):
    gm, df = r_deseq2_norm_fit(
        X,
        y=y,
        sample_meta=sample_meta,
        type=norm_type,
        fit_type=fit_type,
        is_classif=is_classif,
        model_batch=model_batch,
    )
    return np.array(gm, dtype=float), df


def deseq2_norm_vst_transform(X, geo_means, disp_func):
    return np.array(
        r_deseq2_norm_vst_transform(X, geo_means=geo_means, disp_func=disp_func),
        dtype=float,
    )


def edger_tmm_cpm_transform(X, ref_sample, log, prior_count):
    return np.array(
        r_edger_tmm_cpm_transform(
            X, ref_sample=ref_sample, log=log, prior_count=prior_count
        ),
        dtype=float,
    )


def edger_tmm_tpm_transform(
    X, feature_meta, ref_sample, log, prior_count, gene_length_col
):
    return np.array(
        r_edger_tmm_tpm_transform(
            X,
            feature_meta=feature_meta,
            ref_sample=ref_sample,
            log=log,
            prior_count=prior_count,
            gene_length_col=gene_length_col,
        ),
        dtype=float,
    )


class DESeq2NormVST(ExtendedTransformerMixin, BaseEstimator):
    """DESeq2 normalization and VST transformation for count data

    Parameters
    ----------
    norm_type : str (default = "ratio")
        estimateSizeFactors type option.

    fit_type : str (default = "parametric")
        estimateDispersions fitType option.

    is_classif : bool (default = True)
        Whether this is a classification design.

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

    disp_func_ : R/rpy2 function
        Normalization dispersion function.
    """

    def __init__(
        self,
        norm_type="ratio",
        fit_type="parametric",
        is_classif=True,
        model_batch=False,
        memory=None,
    ):
        self.norm_type = norm_type
        self.fit_type = fit_type
        self.is_classif = is_classif
        self.model_batch = model_batch
        self.memory = memory

    def fit(self, X, y=None, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like (default = None), shape = (n_samples,)
            Training class labels. Ignored if is_classif=False.

        sample_meta : pandas.DataFrame, pandas.Series (default = None) \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.is_classif:
            X, y = self._validate_data(X, y, dtype=int)
        else:
            X = self._validate_data(X, dtype=int)
        if y is None:
            y = robjects.NULL
        if sample_meta is None:
            sample_meta = robjects.NULL
        self.geo_means_, self.disp_func_ = deseq2_norm_fit(
            X,
            y=y,
            sample_meta=sample_meta,
            norm_type=self.norm_type,
            fit_type=self.fit_type,
            is_classif=self.is_classif,
            model_batch=self.model_batch,
        )
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
        check_is_fitted(self, "geo_means_")
        # X = check_array(X, dtype=int)
        memory = check_memory(self.memory)
        X = memory.cache(deseq2_norm_vst_transform)(
            X, geo_means=self.geo_means_, disp_func=self.disp_func_
        )
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
        raise NotImplementedError("inverse_transform not implemented.")

    def _more_tags(self):
        return {"requires_positive_X": True}


class EdgeRTMMCPM(ExtendedTransformerMixin, BaseEstimator):
    """edgeR TMM normalization and CPM transformation for count data

    Parameters
    ----------
    log : bool (default = True)
        Whether to return log2 transformed values.

    prior_count : float (default = 2)
        Average count to add to each observation to avoid taking log of zero.
        Larger values produce stronger moderation of low counts and more
        shrinkage of the corresponding log fold changes.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    ref_sample_ : array, shape (n_features,)
        TMM normalization reference sample feature vector.
    """

    def __init__(self, log=True, prior_count=2, memory=None):
        self.log = log
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
        X = self._validate_data(X, dtype=int)
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
            edgeR TMM normalized CPM transformed data matrix.
        """
        check_is_fitted(self, "ref_sample_")
        # X = check_array(X, dtype=int)
        memory = check_memory(self.memory)
        X = memory.cache(edger_tmm_cpm_transform)(
            X, ref_sample=self.ref_sample_, log=self.log, prior_count=self.prior_count
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


class EdgeRTMMTPM(ExtendedTransformerMixin, BaseEstimator):
    """edgeR TMM normalization and TPM transformation for count data

    Parameters
    ----------
    log : bool (default = True)
        Whether to return log2 transformed values.

    prior_count : float (default = 2)
        Average count to add to each observation to avoid taking log of zero.
        Larger values produce stronger moderation of low counts and more
        shrinkage of the corresponding log fold changes.

    gene_length_col : str (default = "Length")
        Feature metadata column name holding gene CDS lengths for used in TPM
        transformation method.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    ref_sample_ : array, shape (n_features,)
        TMM normalization reference sample feature vector.
    """

    def __init__(self, log=True, prior_count=2, gene_length_col="Length", memory=None):
        self.log = log
        self.prior_count = prior_count
        self.gene_length_col = gene_length_col
        self.memory = memory

    def fit(self, X, y=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        y : ignored

        feature_meta : ignored
        """
        X = self._validate_data(X, dtype=int)
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
        check_is_fitted(self, "ref_sample_")
        # X = check_array(X, dtype=int)
        memory = check_memory(self.memory)
        X = memory.cache(edger_tmm_tpm_transform)(
            X,
            feature_meta,
            ref_sample=self.ref_sample_,
            log=self.log,
            prior_count=self.prior_count,
            gene_length_col=self.gene_length_col,
        )
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
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, y, feature_meta):
        if X.shape[1] != feature_meta.shape[0]:
            raise ValueError(
                "X ({:d}) and feature_meta ({:d}) have "
                "different feature dimensions".format(X.shape[1], feature_meta.shape[0])
            )
        if self.gene_length_col not in feature_meta.columns:
            raise ValueError(
                "{} feature_meta column does not exist.".format(self.gene_length_col)
            )

    def _more_tags(self):
        return {"requires_positive_X": True}
