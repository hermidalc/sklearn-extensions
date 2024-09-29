import os
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_memory

from ..base import ExtendedTransformerMixin

r_base = importr("base")
if "deseq2_norm_fit" not in ro.globalenv:
    r_base.source(os.path.dirname(__file__) + "/_rna_seq.R")
r_deseq2_norm_fit = ro.globalenv["deseq2_norm_fit"]
r_deseq2_norm_vst_transform = ro.globalenv["deseq2_norm_vst_transform"]
r_edger_norm_fit = ro.globalenv["edger_norm_fit"]
r_edger_norm_cpm_transform = ro.globalenv["edger_norm_cpm_transform"]
r_edger_norm_tpm_transform = ro.globalenv["edger_norm_tpm_transform"]


def deseq2_norm_fit(X, y, sample_meta, norm_type, fit_type, is_classif, model_batch):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        res = r_deseq2_norm_fit(
            X,
            y=y,
            sample_meta=sample_meta,
            type=norm_type,
            fit_type=fit_type,
            is_classif=is_classif,
            model_batch=model_batch,
        )
        return np.array(res["geo_means"], dtype=float), res["disp_func"]


def deseq2_norm_vst_transform(X, geo_means, disp_func):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        return r_deseq2_norm_vst_transform(X, geo_means=geo_means, disp_func=disp_func)


def edger_norm_cpm_transform(X, ref_sample, norm_type, log, prior_count):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        return r_edger_norm_cpm_transform(
            X,
            ref_sample=ref_sample,
            norm_type=norm_type,
            log=log,
            prior_count=prior_count,
        )


def edger_norm_tpm_transform(
    X, feature_meta, ref_sample, norm_type, log, prior_count, gene_length_col
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        return r_edger_norm_tpm_transform(
            X,
            feature_meta=feature_meta,
            ref_sample=ref_sample,
            norm_type=norm_type,
            log=log,
            prior_count=prior_count,
            gene_length_col=gene_length_col,
        )


class DESeq2Normalizer(ExtendedTransformerMixin, BaseEstimator):
    """DESeq2 normalization and transformation for count data

    Parameters
    ----------
    norm_type : str (default = "ratio")
        estimateSizeFactors type option.

    fit_type : str (default = "parametric")
        estimateDispersions fitType option.

    trans_type : str (default = "vst")
        Transformation method

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
        trans_type="vst",
        is_classif=True,
        model_batch=False,
        memory=None,
    ):
        self.norm_type = norm_type
        self.fit_type = fit_type
        self.trans_type = trans_type
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
        self._check_params(X)
        if y is None:
            y = ro.NULL
        if sample_meta is None:
            sample_meta = ro.NULL
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
        X = self._validate_data(X, dtype=int, reset=False)
        memory = check_memory(self.memory)
        if self.trans_type == "vst":
            Xt = memory.cache(deseq2_norm_vst_transform)(
                X, geo_means=self.geo_means_, disp_func=self.disp_func_
            )
        return Xt

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

    def _check_params(self, X):
        if self.norm_type not in ("ratio", "poscounts"):
            raise ValueError("invalid norm_type %s" % self.norm_type)
        if self.fit_type not in ("parametric"):
            raise ValueError("invalid fit_type %s" % self.fit_type)
        if self.trans_type not in ("vst"):
            raise ValueError("invalid trans_type %s" % self.trans_type)

    def _more_tags(self):
        return {"requires_positive_X": True}


class EdgeRNormalizer(ExtendedTransformerMixin, BaseEstimator):
    """edgeR TMM normalization and CPM transformation for count data

    Parameters
    ----------
    norm_type : str (default = "TMM")
        estimateSizeFactors type option.

    trans_type : str (default = "cpm")
        Transformation method. Allowed types are "cpm" or "tpm".

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

    def __init__(
        self,
        norm_type="TMM",
        trans_type="cpm",
        log=True,
        prior_count=2,
        gene_length_col="Length",
        memory=None,
    ):
        self.norm_type = norm_type
        self.trans_type = trans_type
        self.log = log
        self.prior_count = prior_count
        self.gene_length_col = gene_length_col
        self.memory = memory

    def fit(self, X, y=None, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        y : ignored

        sample_meta : ignored

        feature_meta : ignored for cpm required for tpm
        """
        X = self._validate_data(X, dtype=int)
        self._check_params(X, feature_meta)
        with (
            ro.default_converter + numpy2ri.converter + pandas2ri.converter
        ).context():
            self.ref_sample_ = np.array(
                r_edger_norm_fit(X, type=self.norm_type), dtype=int
            )
        return self

    def transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : ignored

        feature_meta : ignored for cpm required for tpm

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            edgeR TMM normalized CPM transformed data matrix.
        """
        check_is_fitted(self, "ref_sample_")
        X = self._validate_data(X, dtype=int, reset=False)
        memory = check_memory(self.memory)
        if feature_meta is None:
            feature_meta = ro.NULL
        if self.trans_type == "cpm":
            Xt = memory.cache(edger_norm_cpm_transform)(
                X,
                ref_sample=self.ref_sample_,
                norm_type=self.norm_type,
                log=self.log,
                prior_count=self.prior_count,
            )
        elif self.trans_type == "tpm":
            Xt = memory.cache(edger_norm_tpm_transform)(
                X,
                feature_meta,
                ref_sample=self.ref_sample_,
                norm_type=self.norm_type,
                log=self.log,
                prior_count=self.prior_count,
                gene_length_col=self.gene_length_col,
            )
        return Xt

    def inverse_transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta : ignored

        feature_meta : ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, feature_meta):
        if self.norm_type not in ("TMM"):
            raise ValueError("invalid norm_type %s" % self.norm_type)
        if self.trans_type not in ("cpm", "tpm"):
            raise ValueError("invalid trans_type %s" % self.trans_type)
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
