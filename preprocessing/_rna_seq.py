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
r_deseq2_norm_transform = ro.globalenv["deseq2_norm_transform"]
r_deseq2_wrench_fit = ro.globalenv["deseq2_wrench_fit"]
r_deseq2_wrench_transform = ro.globalenv["deseq2_wrench_transform"]

r_edger_norm_fit = ro.globalenv["edger_norm_fit"]
r_edger_norm_transform = ro.globalenv["edger_norm_transform"]
r_edger_wrench_fit = ro.globalenv["edger_wrench_fit"]
r_edger_wrench_transform = ro.globalenv["edger_wrench_transform"]


def deseq2_norm_fit(X, y, sample_meta, norm_type, fit_type, is_classif, model_batch):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        Xr = ro.conversion.get_conversion().py2rpy(X)
        yr = ro.conversion.get_conversion().py2rpy(y)
        sample_meta_r = ro.conversion.get_conversion().py2rpy(sample_meta)
    res = r_deseq2_norm_fit(
        Xr,
        y=yr,
        sample_meta=sample_meta_r,
        norm_type=norm_type,
        fit_type=fit_type,
        is_classif=is_classif,
        model_batch=model_batch,
    )
    return np.array(res["geo_means"], dtype=float), res["disp_func"]


def deseq2_norm_transform(X, geo_means, disp_func, norm_type, trans_type):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        return r_deseq2_norm_transform(
            X, geo_means, disp_func, norm_type=norm_type, trans_type=trans_type
        )


def deseq2_wrench_fit(X, sample_meta, est_type, ref_type, z_adj, fit_type):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        Xr = ro.conversion.get_conversion().py2rpy(X)
        sample_meta_r = ro.conversion.get_conversion().py2rpy(sample_meta)
    res = r_deseq2_wrench_fit(
        Xr,
        sample_meta=sample_meta_r,
        est_type=est_type,
        ref_type=ref_type,
        z_adj=z_adj,
        fit_type=fit_type,
    )
    return (
        np.array(res["nzrows"], dtype=bool),
        np.array(res["qref"], dtype=float),
        np.array(res["s2"], dtype=float),
        np.array(res["s2thetag"], dtype=float),
        np.array(res["thetag"], dtype=float),
        res["pi0_fit"],
        res["disp_func"],
    )


def deseq2_wrench_transform(
    X,
    sample_meta,
    nzrows,
    qref,
    s2,
    s2thetag,
    thetag,
    pi0_fit,
    disp_func,
    est_type,
    ref_type,
    z_adj,
    trans_type,
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        return r_deseq2_wrench_transform(
            X,
            sample_meta,
            nzrows,
            qref,
            s2,
            s2thetag,
            thetag,
            pi0_fit,
            disp_func,
            est_type=est_type,
            ref_type=ref_type,
            z_adj=z_adj,
            trans_type=trans_type,
        )


def edger_norm_fit(X, norm_type):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        res = r_edger_norm_fit(X, norm_type=norm_type)
    return np.array(res["ref_sample"], dtype=int)


def edger_norm_transform(
    X,
    ref_sample,
    feature_meta,
    norm_type,
    trans_type,
    log,
    prior_count,
    gene_length_col,
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        return r_edger_norm_transform(
            X,
            ref_sample,
            feature_meta=feature_meta,
            norm_type=norm_type,
            trans_type=trans_type,
            log=log,
            prior_count=prior_count,
            gene_length_col=gene_length_col,
        )


def edger_wrench_fit(X, sample_meta, est_type, ref_type, z_adj):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        Xr = ro.conversion.get_conversion().py2rpy(X)
        sample_meta_r = ro.conversion.get_conversion().py2rpy(sample_meta)
    res = r_edger_wrench_fit(
        Xr, sample_meta=sample_meta_r, est_type=est_type, ref_type=ref_type, z_adj=z_adj
    )
    return (
        np.array(res["nzrows"], dtype=bool),
        np.array(res["qref"], dtype=float),
        np.array(res["s2"], dtype=float),
        np.array(res["s2thetag"], dtype=float),
        np.array(res["thetag"], dtype=float),
        res["pi0_fit"],
    )


def edger_wrench_transform(
    X,
    sample_meta,
    nzrows,
    qref,
    s2,
    s2thetag,
    thetag,
    pi0_fit,
    est_type,
    ref_type,
    z_adj,
    feature_meta,
    trans_type,
    log,
    prior_count,
    gene_length_col,
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        return r_edger_wrench_transform(
            X,
            sample_meta,
            nzrows,
            qref,
            s2,
            s2thetag,
            thetag,
            pi0_fit,
            est_type=est_type,
            ref_type=ref_type,
            z_adj=z_adj,
            feature_meta=feature_meta,
            trans_type=trans_type,
            log=log,
            prior_count=prior_count,
            gene_length_col=gene_length_col,
        )


class DESeq2Normalizer(ExtendedTransformerMixin, BaseEstimator):
    """DESeq2 normalization and transformation for count data

    Parameters
    ----------
    norm_type : str (default = "ratio")
        estimateSizeFactors type.

    fit_type : str (default = "parametric")
        estimateDispersions fitType.

    trans_type : str (default = "vst")
        Transformation method.

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
        Dispersion function.
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

        sample_meta : ignored

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            Normalized and transformed data matrix.
        """
        check_is_fitted(self, "geo_means_")
        X = self._validate_data(X, dtype=int, reset=False)
        memory = check_memory(self.memory)
        Xt = memory.cache(deseq2_norm_transform)(
            X,
            geo_means=self.geo_means_,
            disp_func=self.disp_func_,
            norm_type=self.norm_type,
            trans_type=self.trans_type_,
        )
        return Xt

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

    def _check_params(self, X):
        if self.norm_type not in ("ratio", "poscounts"):
            raise ValueError("invalid norm_type %s" % self.norm_type)
        if self.fit_type not in ("parametric"):
            raise ValueError("invalid fit_type %s" % self.fit_type)
        if self.trans_type not in ("vst"):
            raise ValueError("invalid trans_type %s" % self.trans_type)

    def _more_tags(self):
        return {"requires_positive_X": True}


class DESeq2WrenchNormalizer(ExtendedTransformerMixin, BaseEstimator):
    """DESeq2 + Wrench normalization and transformation for count data

    Parameters
    ----------
    est_type : str (default = "w.marg.mean")
        Wrench estimator type.

    ref_type : str (default = "sw.means")
        Wrench reference vector type.

    z_adj : bool (default = False)
        Whether Wrench feature-wise ratios need to be adjusted by hurdle
        probabilities (arises when taking marginal expectation)

    fit_type : str (default = "parametric")
        estimateDispersions fitType option.

    trans_type : str (default = "vst")
        Transformation method.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    nzrows_ : array, shape (n_features,)
        Wrench non-zero count feature mask.

    qref_ : array, shape (n_nonzero_features,)
        Wrench reference vector.

    s2_ : array, shape (n_nonzero_features,)
        Wrench variance estimates for logged feature-wise counts.

    s2thetag_ : array, shape (n_conditions,)
        Wrench s2thetag.

    thetag_ : array, shape (n_conditions,)
        Wrench thetag.

    pi0_fit_ : R/rpy2 list, shape (n_nonzero_features_,)
        Wrench feature-wise hurdle model glm fitted objects.

    disp_func_ : R/rpy2 function
        Dispersion function.
    """

    def __init__(
        self,
        est_type="w.marg.mean",
        ref_type="sw.means",
        z_adj=False,
        fit_type="parametric",
        trans_type="vst",
        memory=None,
    ):
        self.est_type = est_type
        self.ref_type = ref_type
        self.z_adj = z_adj
        self.fit_type = fit_type
        self.trans_type = trans_type
        self.memory = memory

    def fit(self, X, y=None, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : ignored

        sample_meta : pandas.DataFrame, pandas.Series (default = None) \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_data(X, dtype=int)
        self._check_params(X, sample_meta)
        (
            self.nzrows_,
            self.qref_,
            self.s2_,
            self.s2thetag_,
            self.thetag_,
            self.pi0_fit_,
            self.disp_func_,
        ) = deseq2_wrench_fit(
            X,
            sample_meta,
            est_type=self.est_type,
            ref_type=self.ref_type,
            z_adj=self.z_adj,
            fit_type=self.fit_type,
        )
        return self

    def transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Sample metadata.

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            Normalized and transformed data matrix.
        """
        check_is_fitted(self, "nzrows_")
        X = self._validate_data(X, dtype=int, reset=False)
        memory = check_memory(self.memory)
        Xt = memory.cache(deseq2_wrench_transform, ignore=["pi0_fit"])(
            X,
            sample_meta,
            nzrows=self.nzrows_,
            qref=self.qref_,
            s2=self.s2_,
            s2thetag=self.s2thetag_,
            thetag=self.thetag_,
            pi0_fit=self.pi0_fit_,
            disp_func=self.disp_func_,
            est_type=self.est_type,
            ref_type=self.ref_type,
            z_adj=self.z_adj,
            trans_type=self.trans_type,
        )
        return Xt

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

    def _check_params(self, X, sample_meta):
        if sample_meta is None:
            raise ValueError("sample_meta is required")
        if self.est_type not in ("w.marg.mean", "hurdle.w.mean", "s2.w.mean"):
            raise ValueError("invalid est_type %s" % self.est_type)
        if self.ref_type not in ("sw.means", "logistic"):
            raise ValueError("invalid ref_type %s" % self.ref_type)
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
        calcNormFactors method.

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
        TMM reference sample vector.
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

    def fit(self, X, y=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        y : ignored

        feature_meta : andas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Feature metadata. Required for "tpm" ignored for "cpm".
        """
        X = self._validate_data(X, dtype=int)
        self._check_params(X, feature_meta)
        self.ref_sample_ = edger_norm_fit(X, norm_type=self.norm_type)
        return self

    def transform(self, X, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        feature_meta : andas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Feature metadata. Required for "tpm" ignored for "cpm".

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            Normalized and transformed data matrix.
        """
        check_is_fitted(self, "ref_sample_")
        X = self._validate_data(X, dtype=int, reset=False)
        memory = check_memory(self.memory)
        if feature_meta is None:
            feature_meta = ro.NULL
        Xt = memory.cache(edger_norm_transform)(
            X,
            ref_sample=self.ref_sample_,
            feature_meta=feature_meta,
            norm_type=self.norm_type,
            trans_type=self.trans_type,
            log=self.log,
            prior_count=self.prior_count,
            gene_length_col=self.gene_length_col,
        )
        return Xt

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

    def _check_params(self, X, feature_meta):
        if self.norm_type not in ("TMM"):
            raise ValueError("invalid norm_type %s" % self.norm_type)
        if self.trans_type not in ("cpm", "tpm"):
            raise ValueError("invalid trans_type %s" % self.trans_type)
        if self.trans_type == "tpm":
            if feature_meta is None:
                raise ValueError("feature_meta required for tpm")
            if X.shape[1] != feature_meta.shape[0]:
                raise ValueError(
                    "X ({:d}) and feature_meta ({:d}) have "
                    "different feature dimensions".format(
                        X.shape[1], feature_meta.shape[0]
                    )
                )
            if self.gene_length_col not in feature_meta.columns:
                raise ValueError(
                    "{} feature_meta column does not exist.".format(
                        self.gene_length_col
                    )
                )

    def _more_tags(self):
        return {"requires_positive_X": True}


class EdgeRWrenchNormalizer(ExtendedTransformerMixin, BaseEstimator):
    """edgeR + Wrench normalization and transformation for count data

    Parameters
    ----------
    est_type : str (default = "w.marg.mean")
        Wrench estimator type.

    ref_type : str (default = "sw.means")
        Wrench reference vector type.

    z_adj : bool (default = False)
        Whether Wrench feature-wise ratios need to be adjusted by hurdle
        probabilities (arises when taking marginal expectation)

    trans_type : str (default = "vst")
        Transformation method.

    log : bool (default = True)
        Whether to return log2 transformed values.

    prior_count : float (default = 1)
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
    nzrows_ : array, shape (n_features,)
        Wrench non-zero count feature mask.

    qref_ : array, shape (n_nonzero_features,)
        Wrench reference vector.

    s2_ : array, shape (n_nonzero_features,)
        Wrench variance estimates for logged feature-wise counts.

    s2thetag_ : array, shape (n_conditions,)
        Wrench s2thetag.

    thetag_ : array, shape (n_conditions,)
        Wrench thetag.

    pi0_fit_ : R/rpy2 list, shape (n_nonzero_features_,)
        Wrench feature-wise hurdle model glm fitted objects.
    """

    def __init__(
        self,
        est_type="w.marg.mean",
        ref_type="sw.means",
        z_adj=False,
        trans_type="cpm",
        log=True,
        prior_count=1,
        gene_length_col="Length",
        memory=None,
    ):
        self.est_type = est_type
        self.ref_type = ref_type
        self.z_adj = z_adj
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
            Training counts data matrix.

        y : ignored

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        feature_meta : andas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Feature metadata. Required for "tpm" ignored for "cpm".

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_data(X, dtype=int)
        self._check_params(X, sample_meta, feature_meta)
        (
            self.nzrows_,
            self.qref_,
            self.s2_,
            self.s2thetag_,
            self.thetag_,
            self.pi0_fit_,
        ) = edger_wrench_fit(
            X,
            sample_meta,
            est_type=self.est_type,
            ref_type=self.ref_type,
            z_adj=self.z_adj,
        )
        return self

    def transform(self, X, sample_meta, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Sample metadata.

        feature_meta : andas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Feature metadata. Required for "tpm" ignored for "cpm".

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            Normalized and transformed data matrix.
        """
        check_is_fitted(self, "nzrows_")
        X = self._validate_data(X, dtype=int, reset=False)
        memory = check_memory(self.memory)
        Xt = memory.cache(edger_wrench_transform, ignore=["pi0_fit"])(
            X,
            sample_meta,
            nzrows=self.nzrows_,
            qref=self.qref_,
            s2=self.s2_,
            s2thetag=self.s2thetag_,
            thetag=self.thetag_,
            pi0_fit=self.pi0_fit_,
            est_type=self.est_type,
            ref_type=self.ref_type,
            z_adj=self.z_adj,
            feature_meta=feature_meta,
            trans_type=self.trans_type,
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

    def _check_params(self, X, sample_meta, feature_meta):
        if sample_meta is None:
            raise ValueError("sample_meta is required")
        if self.est_type not in ("w.marg.mean", "hurdle.w.mean", "s2.w.mean"):
            raise ValueError("invalid est_type %s" % self.est_type)
        if self.ref_type not in ("sw.means", "logistic"):
            raise ValueError("invalid ref_type %s" % self.ref_type)
        if self.trans_type not in ("cpm", "tpm"):
            raise ValueError("invalid trans_type %s" % self.trans_type)
        if self.trans_type == "tpm":
            if feature_meta is None:
                raise ValueError("feature_meta required for tpm")
            if X.shape[1] != feature_meta.shape[0]:
                raise ValueError(
                    "X ({:d}) and feature_meta ({:d}) have "
                    "different feature dimensions".format(
                        X.shape[1], feature_meta.shape[0]
                    )
                )
            if self.gene_length_col not in feature_meta.columns:
                raise ValueError(
                    "{} feature_meta column does not exist.".format(
                        self.gene_length_col
                    )
                )

    def _more_tags(self):
        return {"requires_positive_X": True}
