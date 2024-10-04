import os
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_memory

from ._base import ExtendedSelectorMixin

r_base = importr("base")

if "deseq2_feature_score" not in ro.globalenv:
    r_base.source(os.path.dirname(__file__) + "/_rna_seq.R")

r_deseq2_feature_score = ro.globalenv["deseq2_feature_score"]
r_deseq2_wrench_feature_score = ro.globalenv["deseq2_wrench_feature_score"]
r_deseq2_zinbwave_feature_score = ro.globalenv["deseq2_zinbwave_feature_score"]
r_deseq2_wrench_zinbwave_feature_score = ro.globalenv[
    "deseq2_wrench_zinbwave_feature_score"
]

r_edger_filterbyexpr_mask = ro.globalenv["edger_filterbyexpr_mask"]
r_edger_feature_score = ro.globalenv["edger_feature_score"]
r_edger_wrench_feature_score = ro.globalenv["edger_wrench_feature_score"]
r_edger_zinbwave_feature_score = ro.globalenv["edger_zinbwave_feature_score"]
r_edger_wrench_zinbwave_feature_score = ro.globalenv[
    "edger_wrench_zinbwave_feature_score"
]

r_limma_voom_feature_score = ro.globalenv["limma_voom_feature_score"]
r_limma_voom_wrench_feature_score = ro.globalenv["limma_voom_wrench_feature_score"]
r_dream_voom_feature_score = ro.globalenv["dream_voom_feature_score"]
r_limma_feature_score = ro.globalenv["limma_feature_score"]

if "r_deseq2_norm_transform" not in ro.globalenv:
    r_base.source(os.path.dirname(__file__) + "/../preprocessing/_rna_seq.R")

r_deseq2_norm_transform = ro.globalenv["deseq2_norm_transform"]
r_deseq2_wrench_transform = ro.globalenv["deseq2_wrench_transform"]

r_edger_norm_transform = ro.globalenv["edger_norm_transform"]
r_edger_wrench_transform = ro.globalenv["edger_wrench_transform"]


def deseq2_feature_score(
    X,
    y,
    sample_meta,
    norm_type,
    fit_type,
    score_type,
    lfc,
    lfc_shrink,
    model_batch,
    n_threads,
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        Xr = ro.conversion.get_conversion().py2rpy(X)
        yr = ro.conversion.get_conversion().py2rpy(y)
        sample_meta_r = ro.conversion.get_conversion().py2rpy(sample_meta)
    res = r_deseq2_feature_score(
        Xr,
        yr,
        sample_meta=sample_meta_r,
        norm_type=norm_type,
        fit_type=fit_type,
        score_type=score_type,
        lfc=lfc,
        lfc_shrink=lfc_shrink,
        model_batch=model_batch,
        n_threads=n_threads,
    )
    return (
        np.array(res["scores"], dtype=float),
        np.array(res["padj"], dtype=float),
        np.array(res["geo_means"], dtype=float),
        res["disp_func"],
    )


def deseq2_wrench_feature_score(
    X,
    y,
    sample_meta,
    est_type,
    ref_type,
    z_adj,
    fit_type,
    score_type,
    lfc,
    lfc_shrink,
    n_threads,
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        Xr = ro.conversion.get_conversion().py2rpy(X)
        yr = ro.conversion.get_conversion().py2rpy(y)
        sample_meta_r = ro.conversion.get_conversion().py2rpy(sample_meta)
    res = r_deseq2_wrench_feature_score(
        Xr,
        yr,
        sample_meta=sample_meta_r,
        est_type=est_type,
        ref_type=ref_type,
        z_adj=z_adj,
        fit_type=fit_type,
        score_type=score_type,
        lfc=lfc,
        lfc_shrink=lfc_shrink,
        n_threads=n_threads,
    )
    return (
        np.array(res["scores"], dtype=float),
        np.array(res["padj"], dtype=float),
        np.array(res["nzrows"], dtype=bool),
        np.array(res["qref"], dtype=float),
        np.array(res["s2"], dtype=float),
        np.array(res["s2thetag"], dtype=float),
        np.array(res["thetag"], dtype=float),
        res["pi0_fit"],
        res["disp_func"],
    )


def deseq2_zinbwave_feature_score(
    X,
    y,
    sample_meta,
    epsilon,
    norm_type,
    fit_type,
    score_type,
    lfc,
    lfc_shrink,
    model_batch,
    n_threads,
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        Xr = ro.conversion.get_conversion().py2rpy(X)
        yr = ro.conversion.get_conversion().py2rpy(y)
        sample_meta_r = ro.conversion.get_conversion().py2rpy(sample_meta)
    res = r_deseq2_zinbwave_feature_score(
        Xr,
        yr,
        sample_meta=sample_meta_r,
        epsilon=epsilon,
        norm_type=norm_type,
        fit_type=fit_type,
        score_type=score_type,
        lfc=lfc,
        lfc_shrink=lfc_shrink,
        model_batch=model_batch,
        n_threads=n_threads,
    )
    return (
        np.array(res["scores"], dtype=float),
        np.array(res["padj"], dtype=float),
        np.array(res["geo_means"], dtype=float),
        res["disp_func"],
    )


def deseq2_wrench_zinbwave_feature_score(
    X,
    y,
    sample_meta,
    est_type,
    ref_type,
    z_adj,
    epsilon,
    fit_type,
    score_type,
    lfc,
    lfc_shrink,
    n_threads,
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        Xr = ro.conversion.get_conversion().py2rpy(X)
        yr = ro.conversion.get_conversion().py2rpy(y)
        sample_meta_r = ro.conversion.get_conversion().py2rpy(sample_meta)
    res = r_deseq2_wrench_zinbwave_feature_score(
        Xr,
        yr,
        sample_meta=sample_meta_r,
        est_type=est_type,
        ref_type=ref_type,
        z_adj=z_adj,
        epsilon=epsilon,
        fit_type=fit_type,
        score_type=score_type,
        lfc=lfc,
        lfc_shrink=lfc_shrink,
        n_threads=n_threads,
    )
    return (
        np.array(res["scores"], dtype=float),
        np.array(res["padj"], dtype=float),
        np.array(res["nzrows"], dtype=bool),
        np.array(res["qref"], dtype=float),
        np.array(res["s2"], dtype=float),
        np.array(res["s2thetag"], dtype=float),
        np.array(res["thetag"], dtype=float),
        res["pi0_fit"],
        res["disp_func"],
    )


def edger_feature_score(
    X, y, sample_meta, norm_type, score_type, lfc, robust, model_batch
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        res = r_edger_feature_score(
            X,
            y,
            sample_meta=sample_meta,
            norm_type=norm_type,
            score_type=score_type,
            lfc=lfc,
            robust=robust,
            model_batch=model_batch,
        )
    return (
        np.array(res["scores"], dtype=float),
        np.array(res["padj"], dtype=float),
        np.array(res["ref_sample"], dtype=float),
    )


def edger_wrench_feature_score(
    X, y, sample_meta, est_type, ref_type, z_adj, score_type, lfc, robust
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        Xr = ro.conversion.get_conversion().py2rpy(X)
        yr = ro.conversion.get_conversion().py2rpy(y)
        sample_meta_r = ro.conversion.get_conversion().py2rpy(sample_meta)
    res = r_edger_wrench_feature_score(
        Xr,
        yr,
        sample_meta=sample_meta_r,
        est_type=est_type,
        ref_type=ref_type,
        z_adj=z_adj,
        score_type=score_type,
        lfc=lfc,
        robust=robust,
    )
    return (
        np.array(res["scores"], dtype=float),
        np.array(res["padj"], dtype=float),
        np.array(res["nzrows"], dtype=bool),
        np.array(res["qref"], dtype=float),
        np.array(res["s2"], dtype=float),
        np.array(res["s2thetag"], dtype=float),
        np.array(res["thetag"], dtype=float),
        res["pi0_fit"],
    )


def edger_zinbwave_feature_score(
    X, y, sample_meta, epsilon, norm_type, score_type, robust, model_batch, n_threads
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        res = r_edger_zinbwave_feature_score(
            X,
            y,
            sample_meta=sample_meta,
            epsilon=epsilon,
            norm_type=norm_type,
            score_type=score_type,
            robust=robust,
            model_batch=model_batch,
            n_threads=n_threads,
        )
    return (
        np.array(res["scores"], dtype=float),
        np.array(res["padj"], dtype=float),
        np.array(res["ref_sample"], dtype=float),
    )


def edger_wrench_zinbwave_feature_score(
    X,
    y,
    sample_meta,
    est_type,
    ref_type,
    z_adj,
    epsilon,
    score_type,
    robust,
    n_threads,
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        Xr = ro.conversion.get_conversion().py2rpy(X)
        yr = ro.conversion.get_conversion().py2rpy(y)
        sample_meta_r = ro.conversion.get_conversion().py2rpy(sample_meta)
    res = r_edger_wrench_zinbwave_feature_score(
        Xr,
        yr,
        sample_meta=sample_meta_r,
        est_type=est_type,
        ref_type=ref_type,
        z_adj=z_adj,
        epsilon=epsilon,
        score_type=score_type,
        robust=robust,
        n_threads=n_threads,
    )
    return (
        np.array(res["scores"], dtype=float),
        np.array(res["padj"], dtype=float),
        np.array(res["nzrows"], dtype=bool),
        np.array(res["qref"], dtype=float),
        np.array(res["s2"], dtype=float),
        np.array(res["s2thetag"], dtype=float),
        np.array(res["thetag"], dtype=float),
        res["pi0_fit"],
    )


def limma_voom_feature_score(
    X, y, sample_meta, norm_type, score_type, lfc, robust, model_batch, model_dupcor
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        res = r_limma_voom_feature_score(
            X,
            y,
            sample_meta=sample_meta,
            norm_type=norm_type,
            score_type=score_type,
            lfc=lfc,
            robust=robust,
            model_batch=model_batch,
            model_dupcor=model_dupcor,
        )
    return (
        np.array(res["scores"], dtype=float),
        np.array(res["padj"], dtype=float),
        np.array(res["ref_sample"], dtype=float),
    )


def limma_voom_wrench_feature_score(
    X,
    y,
    sample_meta,
    est_type,
    ref_type,
    z_adj,
    score_type,
    lfc,
    robust,
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        Xr = ro.conversion.get_conversion().py2rpy(X)
        yr = ro.conversion.get_conversion().py2rpy(y)
        sample_meta_r = ro.conversion.get_conversion().py2rpy(sample_meta)
    res = r_limma_voom_wrench_feature_score(
        Xr,
        yr,
        sample_meta=sample_meta_r,
        est_type=est_type,
        ref_type=ref_type,
        z_adj=z_adj,
        score_type=score_type,
        lfc=lfc,
        robust=robust,
    )
    return (
        np.array(res["scores"], dtype=float),
        np.array(res["padj"], dtype=float),
        np.array(res["nzrows"], dtype=bool),
        np.array(res["qref"], dtype=float),
        np.array(res["s2"], dtype=float),
        np.array(res["s2thetag"], dtype=float),
        np.array(res["thetag"], dtype=float),
        res["pi0_fit"],
    )


def dream_voom_feature_score(
    X,
    y,
    sample_meta,
    norm_type,
    score_type,
    lfc,
    model_batch,
    n_threads,
):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        res = r_dream_voom_feature_score(
            X,
            y,
            sample_meta,
            norm_type=norm_type,
            score_type=score_type,
            lfc=lfc,
            model_batch=model_batch,
            n_threads=n_threads,
        )
    return (
        np.array(res["scores"], dtype=float),
        np.array(res["padj"], dtype=float),
        np.array(res["ref_sample"], dtype=float),
    )


def limma_feature_score(X, y, sample_meta, score_type, lfc, robust, trend, model_batch):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        res = r_limma_feature_score(
            X,
            y,
            sample_meta=sample_meta,
            score_type=score_type,
            lfc=lfc,
            robust=robust,
            trend=trend,
            model_batch=model_batch,
        )
    return (
        np.array(res["scores"], dtype=float),
        np.array(res["padj"], dtype=float),
        np.array(res["ref_sample"], dtype=float),
    )


def deseq2_norm_transform(X, geo_means, disp_func, norm_type, trans_type):
    with (ro.default_converter + numpy2ri.converter + pandas2ri.converter).context():
        return r_deseq2_norm_transform(
            X, geo_means, disp_func, norm_type=norm_type, trans_type=trans_type
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


class CountThreshold(ExtendedSelectorMixin, BaseEstimator):
    """Minimum counts in minimum number of samples feature selector.

    Parameters
    ----------
    min_count : int (default = 1)
        Minimum feature count threshold.

    min_total_count : int (default = 1)
        Minimum feature total count threshold.

    min_samples : int (default = 1)
        Minimum number of samples meeting count thresholds. Specify either
        `min_samples` or `min_prop`.

    min_prop : float (default = 0.25)
        Minimum proportion of samples in the smallest group meeting count thresholds.
        Should be between 0 and 1. Specify either `min_samples` or `min_prop`.
    """

    def __init__(
        self,
        min_count=1,
        min_total_count=1,
        min_samples=1,
        min_prop=0.25,
    ):
        self.min_count = min_count
        self.min_total_count = min_total_count
        self.min_samples = min_samples
        self.min_prop = min_prop

    def fit(self, X, y=None, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like (default = None), shape = (n_samples,)
            Training class labels. Ignored if is_classif=False.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        if y is not None:
            X, y = self._validate_data(X, y, dtype=int)
        else:
            X = self._validate_data(X, dtype=int)
        self._mask = np.sum(X >= self.min_count, axis=0) >= self.min_samples
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
        Xr : array of shape (n_samples, n_selected_features)
            Data matrix with only the selected features.
        """
        check_is_fitted(self, "_mask")
        return super().transform(X)

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
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _get_support_mask(self):
        check_is_fitted(self, "_mask")
        return self._mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class DESeq2Selector(ExtendedSelectorMixin, BaseEstimator):
    """DESeq2 differential expression feature selector, normalizer, and
    transformer for count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. Specifying k = "all" and pv = 1.0
        bypasses selection, for use in a parameter search. When pv is also
        specified then returns the intersection of both parameter results.

    pv : float (default = 1.0)
        Select top features below an adjusted p-value threshold. Specifying
        k = "all" and pv = 1.0 bypasses selection, for use in a parameter
        search. When k is also specified returns the intersection of both
        parameter results.

    fc : float (default = 1.0)
        lfcShrink absolute fold change minimum threshold.

    norm_type : str (default = "ratio")
        estimateSizeFactors type option. Available types "ratio", "poscounts".

    fit_type : str (default = "parametric")
        estimateDispersions fitType option.

    score_type : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    trans_type : str (default = "vst")
        Transformation method.

    lfc_shrink : bool (default = True)
        Run lfcShrink after differential expression testing.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and "Batch" column
        exists.

    n_threads : int (default = 1)
        Number of DESeq2 parallel threads. This should be carefully selected
        when using within Grid/RandomizedSearchCV to not oversubscribe CPU
        and memory resources.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    scores_ : array, shape (n_features,)
        Feature scores.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

    geo_means_ : array, shape (n_features,)
        Feature geometric means.

    disp_func_ : R/rpy2 function
        Dispersion function.
    """

    def __init__(
        self,
        k="all",
        pv=1,
        fc=1,
        norm_type="ratio",
        fit_type="parametric",
        score_type="pv",
        trans_type="vst",
        lfc_shrink=True,
        model_batch=False,
        n_threads=1,
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.norm_type = norm_type
        self.fit_type = fit_type
        self.score_type = score_type
        self.trans_type = trans_type
        self.lfc_shrink = lfc_shrink
        self.model_batch = model_batch
        self.n_threads = n_threads
        self.memory = memory

    def fit(self, X, y, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        self.scores_, self.padjs_, self.geo_means_, self.disp_func_ = memory.cache(
            deseq2_feature_score
        )(
            X,
            y,
            sample_meta=sample_meta,
            norm_type=self.norm_type,
            fit_type=self.fit_type,
            score_type=self.score_type,
            lfc=np.log2(self.fc),
            lfc_shrink=self.lfc_shrink,
            model_batch=self.model_batch,
            n_threads=self.n_threads,
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
        Xr : array of shape (n_samples, n_selected_features)
            Normalized and transformed data matrix with only the selected features.
        """
        check_is_fitted(self, "scores_")
        memory = check_memory(self.memory)
        Xt = memory.cache(deseq2_norm_transform)(
            X,
            geo_means=self.geo_means_,
            disp_func=self.disp_func_,
            norm_type=self.norm_type,
            trans_type=self.trans_type_,
        )
        return super().transform(Xt)

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
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features." % self.k
            )
        if not 0 <= self.pv <= 1:
            raise ValueError("pv should be 0 <= pv <= 1; got %r." % self.pv)
        if self.fc < 1:
            raise ValueError("fold change threshold should be >= 1; got %r." % self.fc)
        if self.score_type not in ("pv", "lfc_pv"):
            raise ValueError("invalid score_type %s" % self.score_type)
        if self.norm_type not in ("ratio", "poscounts"):
            raise ValueError("invalid norm_type %s" % self.norm_type)
        if self.fit_type not in ("parametric"):
            raise ValueError("invalid fit_type %s" % self.fit_type)
        if self.trans_type not in ("vst"):
            raise ValueError("invalid trans_type %s" % self.trans_type)

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.pv > 0:
            if self.k == "all":
                mask = np.ones_like(self.scores_, dtype=bool)
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
            elif self.k > 0:
                mask[np.argsort(self.scores_, kind="mergesort")[: self.k]] = True
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
        return mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class DESeq2WrenchSelector(ExtendedSelectorMixin, BaseEstimator):
    """DESeq2 + Wrench differential expression feature selector, normalizer,
    and transformer for zero-inflated count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. Specifying k = "all" and pv = 1.0
        bypasses selection, for use in a parameter search. When pv is also
        specified then returns the intersection of both parameter results.

    pv : float (default = 1.0)
        Select top features below an adjusted p-value threshold. Specifying
        k = "all" and pv = 1.0 bypasses selection, for use in a parameter
        search. When k is also specified returns the intersection of both
        parameter results.

    fc : float (default = 1.0)
        lfcShrink absolute fold change minimum threshold.

    est_type : str (default = "w.marg.mean")
        Wrench estimator type.

    ref_type : str (default = "sw.means")
        Wrench reference vector type.

    z_adj : bool (default = False)
        Whether Wrench feature-wise ratios need to be adjusted by hurdle
        probabilities (arises when taking marginal expectation)

    fit_type : str (default = "parametric")
        estimateDispersions fitType option.

    score_type : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    trans_type : str (default = "vst")
        Transformation method.

    lfc_shrink : bool (default = True)
        Run lfcShrink after differential expression testing.

    n_threads : int (default = 1)
        Number of DESeq2 parallel threads. This should be carefully selected
        when using within Grid/RandomizedSearchCV to not oversubscribe CPU
        and memory resources.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    scores_ : array, shape (n_features,)
        Feature scores.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

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
        k="all",
        pv=1,
        fc=1,
        est_type="w.marg.mean",
        ref_type="sw.means",
        z_adj=False,
        fit_type="parametric",
        score_type="pv",
        trans_type="vst",
        lfc_shrink=True,
        n_threads=1,
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.est_type = est_type
        self.ref_type = ref_type
        self.z_adj = z_adj
        self.fit_type = fit_type
        self.score_type = score_type
        self.trans_type = trans_type
        self.lfc_shrink = lfc_shrink
        self.n_threads = n_threads
        self.memory = memory

    def fit(self, X, y, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, sample_meta)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        (
            self.scores_,
            self.padjs_,
            self.nzrows_,
            self.qref_,
            self.s2_,
            self.s2thetag_,
            self.thetag_,
            self.pi0_fit_,
            self.disp_func_,
        ) = memory.cache(deseq2_wrench_feature_score)(
            X,
            y,
            sample_meta=sample_meta,
            est_type=self.est_type,
            ref_type=self.ref_type,
            z_adj=self.z_adj,
            fit_type=self.fit_type,
            score_type=self.score_type,
            lfc_shrink=self.lfc_shrink,
            n_threads=self.n_threads,
        )
        return self

    def transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            Normalized and transformed data matrix with only the selected features.
        """
        check_is_fitted(self, "scores_")
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
        return super().transform(Xt)

    def inverse_transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, sample_meta):
        if sample_meta is None:
            raise ValueError("sample_meta is required")
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features." % self.k
            )
        if not 0 <= self.pv <= 1:
            raise ValueError("pv should be 0 <= pv <= 1; got %r." % self.pv)
        if self.fc < 1:
            raise ValueError("fold change threshold should be >= 1; got %r." % self.fc)
        if self.score_type not in ("pv", "lfc_pv"):
            raise ValueError("invalid score_type %s" % self.score_type)
        if self.est_type not in ("w.marg.mean", "hurdle.w.mean", "s2.w.mean"):
            raise ValueError("invalid est_type %s" % self.est_type)
        if self.ref_type not in ("sw.means", "logistic"):
            raise ValueError("invalid ref_type %s" % self.ref_type)
        if self.fit_type not in ("parametric"):
            raise ValueError("invalid fit_type %s" % self.fit_type)
        if self.trans_type not in ("vst"):
            raise ValueError("invalid trans_type %s" % self.trans_type)

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.pv > 0:
            if self.k == "all":
                mask = np.ones_like(self.scores_, dtype=bool)
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
            elif self.k > 0:
                mask[np.argsort(self.scores_, kind="mergesort")[: self.k]] = True
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
        return mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class DESeq2ZINBWaVESelector(ExtendedSelectorMixin, BaseEstimator):
    """DESeq2 + ZINB-WaVE differential expression feature selector, normalizer,
    and transformer for zero-inflated count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. Specifying k = "all" and pv = 1.0
        bypasses selection, for use in a parameter search. When pv is also
        specified then returns the intersection of both parameter results.

    pv : float (default = 1.0)
        Select top features below an adjusted p-value threshold. Specifying
        k = "all" and pv = 1.0 bypasses selection, for use in a parameter
        search. When k is also specified returns the intersection of both
        parameter results.

    fc : float (default = 1.0)
        lfcShrink absolute fold change minimum threshold.

    epsilon : float (default = 1e12)
        ZINB-WaVE regularization hyperparameter.

    norm_type : str (default = "poscounts")
        estimateSizeFactors type option.

    fit_type : str (default = "parametric")
        estimateDispersions fitType option.

    score_type : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    trans_type : str (default = "vst")
        Transformation method.

    lfc_shrink : bool (default = True)
        Run lfcShrink after differential expression testing.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and "Batch" column
        exists.

    n_threads : int (default = 1)
        Number of DESeq2 parallel threads. This should be carefully selected
        when using within Grid/RandomizedSearchCV to not oversubscribe CPU
        and memory resources.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    scores_ : array, shape (n_features,)
        Feature scores.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

    geo_means_ : array, shape (n_features,)
        Feature geometric means.

    disp_func_ : R/rpy2 function
        Dispersion function.
    """

    def __init__(
        self,
        k="all",
        pv=1,
        fc=1,
        epsilon=1e12,
        norm_type="poscounts",
        fit_type="parametric",
        score_type="pv",
        trans_type="vst",
        lfc_shrink=True,
        model_batch=False,
        n_threads=1,
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.epsilon = epsilon
        self.norm_type = norm_type
        self.fit_type = fit_type
        self.score_type = score_type
        self.trans_type = trans_type
        self.lfc_shrink = lfc_shrink
        self.model_batch = model_batch
        self.n_threads = n_threads
        self.memory = memory

    def fit(self, X, y, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        self.scores_, self.padjs_, self.geo_means_, self.disp_func_ = memory.cache(
            deseq2_zinbwave_feature_score
        )(
            X,
            y,
            sample_meta=sample_meta,
            epsilon=self.epsilon,
            norm_type=self.norm_type,
            fit_type=self.fit_type,
            score_type=self.score_type,
            lfc=np.log2(self.fc),
            lfc_shrink=self.lfc_shrink,
            model_batch=self.model_batch,
            n_threads=self.n_threads,
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
        Xr : array of shape (n_samples, n_selected_features)
            Normalized and transformed data matrix with only the selected features.
        """
        check_is_fitted(self, "scores_")
        memory = check_memory(self.memory)
        Xt = memory.cache(deseq2_norm_transform)(
            X,
            geo_means=self.geo_means_,
            disp_func=self.disp_func_,
            norm_type=self.norm_type,
            trans_type=self.trans_type_,
        )
        return super().transform(Xt)

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
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features." % self.k
            )
        if not 0 <= self.pv <= 1:
            raise ValueError("pv should be 0 <= pv <= 1; got %r." % self.pv)
        if self.fc < 1:
            raise ValueError("fold change threshold should be >= 1; got %r." % self.fc)
        if self.score_type not in ("pv", "lfc_pv"):
            raise ValueError("invalid score_type %s" % self.score_type)
        if self.norm_type not in ("poscounts"):
            raise ValueError("invalid norm_type %s" % self.norm_type)
        if self.fit_type not in ("parametric"):
            raise ValueError("invalid fit_type %s" % self.fit_type)
        if self.trans_type not in ("vst"):
            raise ValueError("invalid trans_type %s" % self.trans_type)

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.pv > 0:
            if self.k == "all":
                mask = np.ones_like(self.scores_, dtype=bool)
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
            elif self.k > 0:
                mask[np.argsort(self.scores_, kind="mergesort")[: self.k]] = True
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
        return mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class DESeq2WrenchZINBWaVESelector(ExtendedSelectorMixin, BaseEstimator):
    """DESeq2 + Wrench + ZINB-WaVE differential expression feature selector,
    normalizer, and transformer for zero-inflated count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. Specifying k = "all" and pv = 1.0
        bypasses selection, for use in a parameter search. When pv is also
        specified then returns the intersection of both parameter results.

    pv : float (default = 1.0)
        Select top features below an adjusted p-value threshold. Specifying
        k = "all" and pv = 1.0 bypasses selection, for use in a parameter
        search. When k is also specified returns the intersection of both
        parameter results.

    fc : float (default = 1.0)
        lfcShrink absolute fold change minimum threshold.

    est_type : str (default = "w.marg.mean")
        Wrench estimator type.

    ref_type : str (default = "sw.means")
        Wrench reference vector type.

    z_adj : bool (default = False)
        Whether Wrench feature-wise ratios need to be adjusted by hurdle
        probabilities (arises when taking marginal expectation)

    epsilon : float (default = 1e12)
        ZINB-WaVE regularization hyperparameter.

    fit_type : str (default = "parametric")
        estimateDispersions fitType option.

    score_type : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    trans_type : str (default = "vst")
        Transformation method.

    lfc_shrink : bool (default = True)
        Run lfcShrink after differential expression testing.

    n_threads : int (default = 1)
        Number of DESeq2 parallel threads. This should be carefully selected
        when using within Grid/RandomizedSearchCV to not oversubscribe CPU
        and memory resources.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    scores_ : array, shape (n_features,)
        Feature scores.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

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
        k="all",
        pv=1,
        fc=1,
        est_type="w.marg.mean",
        ref_type="sw.means",
        z_adj=False,
        epsilon=1e12,
        fit_type="parametric",
        score_type="pv",
        trans_type="vst",
        lfc_shrink=True,
        n_threads=1,
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.est_type = est_type
        self.ref_type = ref_type
        self.z_adj = z_adj
        self.epsilon = epsilon
        self.fit_type = fit_type
        self.score_type = score_type
        self.trans_type = trans_type
        self.lfc_shrink = lfc_shrink
        self.n_threads = n_threads
        self.memory = memory

    def fit(self, X, y, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, sample_meta)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        (
            self.scores_,
            self.padjs_,
            self.nzrows_,
            self.qref_,
            self.s2_,
            self.s2thetag_,
            self.thetag_,
            self.pi0_fit_,
            self.disp_func_,
        ) = memory.cache(deseq2_wrench_zinbwave_feature_score)(
            X,
            y,
            sample_meta=sample_meta,
            est_type=self.est_type,
            ref_type=self.ref_type,
            z_adj=self.z_adj,
            epsilon=self.epsilon,
            fit_type=self.fit_type,
            score_type=self.score_type,
            lfc=np.log2(self.fc),
            lfc_shrink=self.lfc_shrink,
            n_threads=self.n_threads,
        )
        return self

    def transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            Normalized and transformed data matrix with only the selected features.
        """
        check_is_fitted(self, "scores_")
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
        return super().transform(Xt)

    def inverse_transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, sample_meta):
        if sample_meta is None:
            raise ValueError("sample_meta is required")
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features." % self.k
            )
        if not 0 <= self.pv <= 1:
            raise ValueError("pv should be 0 <= pv <= 1; got %r." % self.pv)
        if self.fc < 1:
            raise ValueError("fold change threshold should be >= 1; got %r." % self.fc)
        if self.score_type not in ("pv", "lfc_pv"):
            raise ValueError("invalid score_type %s" % self.score_type)
        if self.est_type not in ("w.marg.mean", "hurdle.w.mean", "s2.w.mean"):
            raise ValueError("invalid est_type %s" % self.est_type)
        if self.ref_type not in ("sw.means", "logistic"):
            raise ValueError("invalid ref_type %s" % self.ref_type)
        if self.fit_type not in ("parametric"):
            raise ValueError("invalid fit_type %s" % self.fit_type)
        if self.trans_type not in ("vst"):
            raise ValueError("invalid trans_type %s" % self.trans_type)

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.pv > 0:
            if self.k == "all":
                mask = np.ones_like(self.scores_, dtype=bool)
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
            elif self.k > 0:
                mask[np.argsort(self.scores_, kind="mergesort")[: self.k]] = True
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
        return mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class EdgeRFilterByExpr(ExtendedSelectorMixin, BaseEstimator):
    """edgeR filterByExpr feature selector for count data

    Parameters
    ----------
    min_count : int (default = 10)
        Minimum count required for at least some samples.

    min_total_count : int (default = 15)
        Minimum total count required.

    large_n : int (default = 10)
        Number of samples per group that is considered to be “large”.

    min_prop : float (default = 0.7)
        Minimum proportion of samples in the smallest group that express the gene.
        Should be between 0 and 1.

    is_classif : bool (default = True)
        Whether this is a classification design.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and Batch column
        exists.
    """

    def __init__(
        self,
        min_count=10,
        min_total_count=15,
        large_n=10,
        min_prop=0.7,
        is_classif=True,
        model_batch=False,
    ):
        self.min_count = min_count
        self.min_total_count = min_total_count
        self.large_n = large_n
        self.min_prop = min_prop
        self.is_classif = is_classif
        self.model_batch = model_batch

    def fit(self, X, y=None, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like (default = None), shape = (n_samples,)
            Training class labels. Ignored if is_classif=False.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
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
            y = ro.NULL
        if sample_meta is None:
            sample_meta = ro.NULL
        with (
            ro.default_converter + numpy2ri.converter + pandas2ri.converter
        ).context():
            self._mask = np.array(
                r_edger_filterbyexpr_mask(
                    X,
                    y=y,
                    sample_meta=sample_meta,
                    min_count=self.min_count,
                    min_total_count=self.min_total_count,
                    large_n=self.large_n,
                    min_prop=self.min_prop,
                    is_classif=self.is_classif,
                    model_batch=self.model_batch,
                ),
                dtype=bool,
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
        Xr : array of shape (n_samples, n_selected_features)
            Data matrix with only the selected features.
        """
        check_is_fitted(self, "_mask")
        return super().transform(X)

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
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _get_support_mask(self):
        check_is_fitted(self, "_mask")
        return self._mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class EdgeRSelector(ExtendedSelectorMixin, BaseEstimator):
    """edgeR differential expression feature selector, normalizer, and
     transformer for count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. Specifying k = "all" and pv = 1.0
        bypasses selection, for use in a parameter search. When pv is also
        specified then returns the intersection of both parameter results.

    pv : float (default = 1.0)
        Select top features below an adjusted p-value threshold. Specifying
        k = "all" and pv = 1.0 bypasses selection, for use in a parameter
        search. When k is also specified returns the intersection of both
        parameter results.

    fc : float (default = 1.0)
        glmTreat absolute fold change minimum threshold. Default value of 1.0
        gives glmQLFTest results.

    norm_type : str (default = "TMM")
        estimateSizeFactors type option.

    score_type : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    trans_type : str (default = "cpm")
        Transformation method to use on count data after differential
        expression testing. Available methods are "cpm" and "tpm".

    robust : bool (default = True)
        estimateDisp and glmQLFit robust option.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and "Batch" column
        exists.

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
    scores_ : array, shape (n_features,)
        Feature scores.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

    ref_sample_ : array, shape (n_features,)
        TMM reference sample feature vector.
    """

    def __init__(
        self,
        k="all",
        pv=1,
        fc=1,
        norm_type="TMM",
        score_type="pv",
        trans_type="cpm",
        robust=True,
        model_batch=False,
        log=True,
        prior_count=2,
        gene_length_col="Length",
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.norm_type = norm_type
        self.score_type = score_type
        self.trans_type = trans_type
        self.robust = robust
        self.model_batch = model_batch
        self.log = log
        self.prior_count = prior_count
        self.gene_length_col = gene_length_col
        self.memory = memory

    def fit(self, X, y, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        feature_meta : ignored

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, feature_meta)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        self.scores_, self.padjs_, self.ref_sample_ = memory.cache(edger_feature_score)(
            X,
            y,
            sample_meta=sample_meta,
            norm_type=self.norm_type,
            score_type=self.score_type,
            lfc=np.log2(self.fc),
            robust=self.robust,
            model_batch=self.model_batch,
        )
        return self

    def transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : ignored

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata for "tpm" transform, otherwise ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            Normalized and transformed data matrix with only the selected features.
        """
        check_is_fitted(self, "scores_")
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
        return super().transform(Xt)

    def inverse_transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta : ignored

        feature_meta: ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, feature_meta):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features." % self.k
            )
        if not 0 <= self.pv <= 1:
            raise ValueError("pv should be 0 <= pv <= 1; got %r." % self.pv)
        if self.fc < 1:
            raise ValueError("fold change threshold should be >= 1; got %r." % self.fc)
        if self.score_type not in ("pv", "lfc_pv"):
            raise ValueError("invalid score_type %s" % self.score_type)
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

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.pv > 0:
            if self.k == "all":
                mask = np.ones_like(self.scores_, dtype=bool)
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
            elif self.k > 0:
                mask[np.argsort(self.scores_, kind="mergesort")[: self.k]] = True
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
        return mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class EdgeRWrenchSelector(ExtendedSelectorMixin, BaseEstimator):
    """edgeR + Wrench differential expression feature selector, normalizer,
    and transformer for count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. Specifying k = "all" and pv = 1.0
        bypasses selection, for use in a parameter search. When pv is also
        specified then returns the intersection of both parameter results.

    pv : float (default = 1.0)
        Select top features below an adjusted p-value threshold. Specifying
        k = "all" and pv = 1.0 bypasses selection, for use in a parameter
        search. When k is also specified returns the intersection of both
        parameter results.

    fc : float (default = 1.0)
        glmTreat absolute fold change minimum threshold. Default value of 1.0
        gives glmQLFTest results.

    est_type : str (default = "w.marg.mean")
        Wrench estimator type.

    ref_type : str (default = "sw.means")
        Wrench reference vector type.

    z_adj : bool (default = False)
        Whether Wrench feature-wise ratios need to be adjusted by hurdle
        probabilities (arises when taking marginal expectation)

    score_type : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    trans_type : str (default = "cpm")
        Transformation method to use on count data after differential
        expression testing. Available methods are "cpm" and "tpm".

    robust : bool (default = True)
        estimateDisp and glmQLFit robust option.

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
    scores_ : array, shape (n_features,)
        Feature scores.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

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
        k="all",
        pv=1,
        fc=1,
        est_type="w.marg.mean",
        ref_type="sw.means",
        z_adj=False,
        score_type="pv",
        trans_type="cpm",
        robust=True,
        log=True,
        prior_count=1,
        gene_length_col="Length",
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.est_type = est_type
        self.ref_type = ref_type
        self.z_adj = z_adj
        self.score_type = score_type
        self.trans_type = trans_type
        self.robust = robust
        self.log = log
        self.prior_count = prior_count
        self.gene_length_col = gene_length_col
        self.memory = memory

    def fit(self, X, y, sample_meta, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        feature_meta : ignored

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, sample_meta, feature_meta)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        (
            self.scores_,
            self.padjs_,
            self.nzrows_,
            self.qref_,
            self.s2_,
            self.s2thetag_,
            self.thetag_,
            self.pi0_fit_,
        ) = memory.cache(edger_wrench_feature_score)(
            X,
            y,
            sample_meta=sample_meta,
            est_type=self.est_type,
            ref_type=self.ref_type,
            z_adj=self.z_adj,
            score_type=self.score_type,
            lfc=np.log2(self.fc),
            robust=self.robust,
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

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata for "tpm" transform, otherwise ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            Normalized and transformed data matrix with only the selected features.
        """
        check_is_fitted(self, "scores_")
        memory = check_memory(self.memory)
        if feature_meta is None:
            feature_meta = ro.NULL
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
        return super().transform(Xt)

    def inverse_transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta : ignored

        feature_meta: ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, sample_meta, feature_meta):
        if sample_meta is None:
            raise ValueError("sample_meta is required")
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features." % self.k
            )
        if not 0 <= self.pv <= 1:
            raise ValueError("pv should be 0 <= pv <= 1; got %r." % self.pv)
        if self.fc < 1:
            raise ValueError("fold change threshold should be >= 1; got %r." % self.fc)
        if self.score_type not in ("pv", "lfc_pv"):
            raise ValueError("invalid score_type %s" % self.score_type)
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

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.pv > 0:
            if self.k == "all":
                mask = np.ones_like(self.scores_, dtype=bool)
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
            elif self.k > 0:
                mask[np.argsort(self.scores_, kind="mergesort")[: self.k]] = True
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
        return mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class EdgeRZINBWaVESelector(ExtendedSelectorMixin, BaseEstimator):
    """edgeR + ZINB-WaVE differential expression feature selector, normalizer,
    and transformer for zero-inflated count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. Specifying k = "all" and pv = 1.0
        bypasses selection, for use in a parameter search. When pv is also
        specified then returns the intersection of both parameter results.

    pv : float (default = 1.0)
        Select top features below an adjusted p-value threshold. Specifying
        k = "all" and pv = 1.0 bypasses selection, for use in a parameter
        search. When k is also specified returns the intersection of both
        parameter results.

    epsilon : float (default = 1e12)
        ZINB-WaVE regularization hyperparameter.

    norm_type : str (default = "TMM")
        estimateSizeFactors type option.

    score_type : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    trans_type : str (default = "cpm")
        Transformation method to use on count data after differential
        expression testing. Available methods are "cpm" and "tpm".

    robust : bool (default = True)
        estimateDisp robust option.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and "Batch" column
        exists.

    log : bool (default = True)
        Whether to return log2 transformed values.

    prior_count : float (default = 1)
        Average count to add to each observation to avoid taking log of zero.
        Larger values produce stronger moderation of low counts and more
        shrinkage of the corresponding log fold changes.

    gene_length_col : str (default = "Length")
        Feature metadata column name holding gene CDS lengths for used in TPM
        transformation method.

    n_threads : int (default = 1)
        Number of ZINB-WaVE parallel threads. This should be carefully selected
        when using within Grid/RandomizedSearchCV to not oversubscribe CPU
        and memory resources.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    scores_ : array, shape (n_features,)
        Feature scores.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

    ref_sample_ : array, shape (n_features,)
        TMM reference sample feature vector.
    """

    def __init__(
        self,
        k="all",
        pv=1,
        epsilon=1e12,
        norm_type="TMM",
        score_type="pv",
        trans_type="cpm",
        robust=True,
        model_batch=False,
        log=True,
        prior_count=1,
        gene_length_col="Length",
        n_threads=1,
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.epsilon = epsilon
        self.norm_type = norm_type
        self.score_type = score_type
        self.trans_type = trans_type
        self.robust = robust
        self.model_batch = model_batch
        self.trans_type = trans_type
        self.log = log
        self.prior_count = prior_count
        self.gene_length_col = gene_length_col
        self.n_threads = n_threads
        self.memory = memory

    def fit(self, X, y, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        feature_meta : ignored

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, feature_meta)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        self.scores_, self.padjs_, self.ref_sample_ = memory.cache(
            edger_zinbwave_feature_score
        )(
            X,
            y,
            sample_meta=sample_meta,
            epsilon=self.epsilon,
            norm_type=self.norm_type,
            score_type=self.score_type,
            robust=self.robust,
            model_batch=self.model_batch,
            n_threads=self.n_threads,
        )
        return self

    def transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : ignored

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata for "tpm" transform, otherwise ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            Normalized and transformed data matrix with only the selected features.
        """
        check_is_fitted(self, "scores_")
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
        return super().transform(Xt)

    def inverse_transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta : ignored

        feature_meta: ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, y, feature_meta):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features." % self.k
            )
        if not 0 <= self.pv <= 1:
            raise ValueError("pv should be 0 <= pv <= 1; got %r." % self.pv)
        if self.score_type not in ("pv", "lfc_pv"):
            raise ValueError("invalid score_type %s" % self.score_type)
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

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.pv > 0:
            if self.k == "all":
                mask = np.ones_like(self.scores_, dtype=bool)
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
            elif self.k > 0:
                mask[np.argsort(self.scores_, kind="mergesort")[: self.k]] = True
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
        return mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class EdgeRWrenchZINBWaVESelector(ExtendedSelectorMixin, BaseEstimator):
    """edgeR + Wrench + ZINB-WaVE differential expression feature selector,
    normalizer, and transformer for count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. Specifying k = "all" and pv = 1.0
        bypasses selection, for use in a parameter search. When pv is also
        specified then returns the intersection of both parameter results.

    pv : float (default = 1.0)
        Select top features below an adjusted p-value threshold. Specifying
        k = "all" and pv = 1.0 bypasses selection, for use in a parameter
        search. When k is also specified returns the intersection of both
        parameter results.

    est_type : str (default = "w.marg.mean")
        Wrench estimator type.

    ref_type : str (default = "sw.means")
        Wrench reference vector type.

    z_adj : bool (default = False)
        Whether Wrench feature-wise ratios need to be adjusted by hurdle
        probabilities (arises when taking marginal expectation)

    epsilon : float (default = 1e12)
        ZINB-WaVE regularization hyperparameter.

    score_type : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    trans_type : str (default = "cpm")
        Transformation method to use on count data after differential
        expression testing. Available methods are "cpm" and "tpm".

    robust : bool (default = True)
        estimateDisp and glmQLFit robust option.

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
    scores_ : array, shape (n_features,)
        Feature scores.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

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
        k="all",
        pv=1,
        est_type="w.marg.mean",
        ref_type="sw.means",
        z_adj=False,
        epsilon=1e12,
        score_type="pv",
        trans_type="cpm",
        robust=True,
        log=True,
        prior_count=1,
        gene_length_col="Length",
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.est_type = est_type
        self.ref_type = ref_type
        self.z_adj = z_adj
        self.epsilon = epsilon
        self.score_type = score_type
        self.trans_type = trans_type
        self.robust = robust
        self.log = log
        self.prior_count = prior_count
        self.gene_length_col = gene_length_col
        self.memory = memory

    def fit(self, X, y, sample_meta, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        feature_meta : ignored

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, sample_meta, feature_meta)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        (
            self.scores_,
            self.padjs_,
            self.nzrows_,
            self.qref_,
            self.s2_,
            self.s2thetag_,
            self.thetag_,
            self.pi0_fit_,
        ) = memory.cache(edger_wrench_zinbwave_feature_score)(
            X,
            y,
            sample_meta=sample_meta,
            est_type=self.est_type,
            ref_type=self.ref_type,
            z_adj=self.z_adj,
            epsilon=self.epsilon,
            score_type=self.score_type,
            robust=self.robust,
            n_threads=self.n_threads,
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

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata for "tpm" transform, otherwise ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            Normalized and transformed data matrix with only the selected features.
        """
        check_is_fitted(self, "scores_")
        memory = check_memory(self.memory)
        if feature_meta is None:
            feature_meta = ro.NULL
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
        return super().transform(Xt)

    def inverse_transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta : ignored

        feature_meta: ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, sample_meta, feature_meta):
        if sample_meta is None:
            raise ValueError("sample_meta is required")
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features." % self.k
            )
        if not 0 <= self.pv <= 1:
            raise ValueError("pv should be 0 <= pv <= 1; got %r." % self.pv)
        if self.fc < 1:
            raise ValueError("fold change threshold should be >= 1; got %r." % self.fc)
        if self.score_type not in ("pv", "lfc_pv"):
            raise ValueError("invalid score_type %s" % self.score_type)
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

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.pv > 0:
            if self.k == "all":
                mask = np.ones_like(self.scores_, dtype=bool)
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
            elif self.k > 0:
                mask[np.argsort(self.scores_, kind="mergesort")[: self.k]] = True
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
        return mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class LimmaVoomSelector(ExtendedSelectorMixin, BaseEstimator):
    """limma-voom differential expression feature selector, normalizer, and
     transformer for count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. Specifying k = "all" and pv = 1.0
        bypasses selection, for use in a parameter search. When pv is also
        specified then returns the intersection of both parameter results.

    pv : float (default = 1.0)
        Select top features below an adjusted p-value threshold. Specifying
        k = "all" and pv = 1.0 bypasses selection, for use in a parameter
        search. When k is also specified returns the intersection of both
        parameter results.

    fc : float (default = 1.0)
        treat absolute fold change minimum threshold. Default value of 1.0
        gives eBayes results.

    norm_type : str (default = "TMM")
        estimateSizeFactors type option.

    score_type : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    trans_type : str (default = "cpm")
        Transformation method to use on count data after differential
        expression testing. Available methods are "cpm" and "tpm".

    robust : bool (default = True)
        limma treat/eBayes robust option.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and "Batch" column
        exists.

    model_dupcor : bool (default = False)
        Model limma duplicateCorrelation if sample_meta passed to fit and Group
        column exists.

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
    scores_ : array, shape (n_features,)
        Feature scores.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

    ref_sample_ : array, shape (n_features,)
        TMM reference sample feature vector.
    """

    def __init__(
        self,
        k="all",
        pv=1,
        fc=1,
        norm_type="TMM",
        score_type="pv",
        trans_type="cpm",
        robust=True,
        model_batch=False,
        model_dupcor=False,
        log=True,
        prior_count=2,
        gene_length_col="Length",
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.norm_type = norm_type
        self.score_type = score_type
        self.trans_type = trans_type
        self.robust = robust
        self.model_batch = model_batch
        self.model_dupcor = model_dupcor
        self.log = log
        self.prior_count = prior_count
        self.gene_length_col = gene_length_col
        self.memory = memory

    def fit(self, X, y, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        feature_meta : ignored

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, feature_meta)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        self.scores_, self.padjs_, self.ref_sample_ = memory.cache(
            limma_voom_feature_score
        )(
            X,
            y,
            sample_meta=sample_meta,
            norm_type=self.norm_type,
            score_type=self.score_type,
            lfc=np.log2(self.fc),
            robust=self.robust,
            model_batch=self.model_batch,
            model_dupcor=self.model_dupcor,
        )
        return self

    def transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : ignored

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata for "tpm" transform, otherwise ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            Normalized and transformed data matrix with only the selected features.
        """
        check_is_fitted(self, "scores_")
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
        return super().transform(Xt)

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
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, feature_meta):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features." % self.k
            )
        if not 0 <= self.pv <= 1:
            raise ValueError("pv should be 0 <= pv <= 1; got %r." % self.pv)
        if self.fc < 1:
            raise ValueError("fold change threshold should be >= 1; got %r." % self.fc)
        if self.score_type not in ("pv", "lfc_pv"):
            raise ValueError("invalid score_type %s" % self.score_type)
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

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.pv > 0:
            if self.k == "all":
                mask = np.ones_like(self.scores_, dtype=bool)
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
            elif self.k > 0:
                mask[np.argsort(self.scores_, kind="mergesort")[: self.k]] = True
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
        return mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class LimmaVoomWrenchSelector(ExtendedSelectorMixin, BaseEstimator):
    """limma-voom + Wrench differential expression feature selector, normalizer,
    and transformer for count data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. Specifying k = "all" and pv = 1.0
        bypasses selection, for use in a parameter search. When pv is also
        specified then returns the intersection of both parameter results.

    pv : float (default = 1.0)
        Select top features below an adjusted p-value threshold. Specifying
        k = "all" and pv = 1.0 bypasses selection, for use in a parameter
        search. When k is also specified returns the intersection of both
        parameter results.

    fc : float (default = 1.0)
        treat absolute fold change minimum threshold. Default value of 1.0
        gives eBayes results.

    est_type : str (default = "w.marg.mean")
        Wrench estimator type.

    ref_type : str (default = "sw.means")
        Wrench reference vector type.

    z_adj : bool (default = False)
        Whether Wrench feature-wise ratios need to be adjusted by hurdle
        probabilities (arises when taking marginal expectation)

    score_type : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    trans_type : str (default = "cpm")
        Transformation method to use on count data after differential
        expression testing. Available methods are "cpm" and "tpm".

    robust : bool (default = True)
        limma treat/eBayes robust option.

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
    scores_ : array, shape (n_features,)
        Feature scores.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

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
        k="all",
        pv=1,
        fc=1,
        est_type="w.marg.mean",
        ref_type="sw.means",
        z_adj=False,
        score_type="pv",
        trans_type="cpm",
        robust=True,
        log=True,
        prior_count=1,
        gene_length_col="Length",
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.est_type = est_type
        self.ref_type = ref_type
        self.z_adj = z_adj
        self.score_type = score_type
        self.trans_type = trans_type
        self.robust = robust
        self.log = log
        self.prior_count = prior_count
        self.gene_length_col = gene_length_col
        self.memory = memory

    def fit(self, X, y, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        feature_meta : ignored

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, sample_meta, feature_meta)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        (
            self.scores_,
            self.padjs_,
            self.nzrows_,
            self.qref_,
            self.s2_,
            self.s2thetag_,
            self.thetag_,
            self.pi0_fit_,
        ) = memory.cache(limma_voom_wrench_feature_score)(
            X,
            y,
            sample_meta=sample_meta,
            est_type=self.est_type,
            ref_type=self.ref_type,
            z_adj=self.z_adj,
            score_type=self.score_type,
            lfc=np.log2(self.fc),
            robust=self.robust,
        )
        return self

    def transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : ignored

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata for "tpm" transform, otherwise ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            Normalized and transformed data matrix with only the selected features.
        """
        check_is_fitted(self, "scores_")
        memory = check_memory(self.memory)
        if feature_meta is None:
            feature_meta = ro.NULL
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
        return super().transform(Xt)

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
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, sample_meta, feature_meta):
        if sample_meta is None:
            raise ValueError("sample_meta is required")
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features." % self.k
            )
        if not 0 <= self.pv <= 1:
            raise ValueError("pv should be 0 <= pv <= 1; got %r." % self.pv)
        if self.fc < 1:
            raise ValueError("fold change threshold should be >= 1; got %r." % self.fc)
        if self.score_type not in ("pv", "lfc_pv"):
            raise ValueError("invalid score_type %s" % self.score_type)
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

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.pv > 0:
            if self.k == "all":
                mask = np.ones_like(self.scores_, dtype=bool)
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
            elif self.k > 0:
                mask[np.argsort(self.scores_, kind="mergesort")[: self.k]] = True
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
        return mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class DreamVoomSelector(ExtendedSelectorMixin, BaseEstimator):
    """dream limma-voom differential expression feature selector and
    normalizer/transformer for count data repeated measures designs

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. Specifying k = "all" and pv = 1.0
        bypasses selection, for use in a parameter search. When pv is also
        specified then returns the intersection of both parameter results.

    pv : float (default = 1.0)
        Select top features below an adjusted p-value threshold. Specifying
        k = "all" and pv = 1.0 bypasses selection, for use in a parameter
        search. When k is also specified returns the intersection of both
        parameter results.

    fc : float (default = 1.0)
        Absolute fold-change minimum threshold.

    norm_type : str (default = "TMM")
        estimateSizeFactors type option.

    score_type : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    trans_type : str (default = "cpm")
        Transformation method to use on count data after differential
        expression testing. Available methods are "cpm" and "tpm".

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and "Batch" column
        exists.

    n_threads : int (default = 1)
        Number of dream parallel threads. This should be carefully selected
        when using within Grid/RandomizedSearchCV to not oversubscribe CPU
        and memory resources.

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
    scores_ : array, shape (n_features,)
        Feature scores.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.

    ref_sample_ : array, shape (n_features,)
        TMM normalization reference sample feature vector.
    """

    def __init__(
        self,
        k="all",
        pv=1,
        fc=1,
        norm_type="TMM",
        score_type="pv",
        trans_type="cpm",
        model_batch=False,
        n_threads=1,
        log=True,
        prior_count=2,
        gene_length_col="Length",
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.norm_type = norm_type
        self.score_type = score_type
        self.trans_type = trans_type
        self.model_batch = model_batch
        self.n_threads = n_threads
        self.log = log
        self.prior_count = prior_count
        self.gene_length_col = gene_length_col
        self.memory = memory

    def fit(self, X, y, sample_meta, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        feature_meta : ignored

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X)
        memory = check_memory(self.memory)
        self.scores_, self.padjs_, self.ref_sample_ = memory.cache(
            dream_voom_feature_score
        )(
            X,
            y,
            sample_meta=sample_meta,
            norm_type=self.norm_type,
            score_type=self.score_type,
            lfc=np.log2(self.fc),
            model_batch=self.model_batch,
            n_threads=self.n_threads,
        )
        return self

    def transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : ignored

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata for "tpm" transform, otherwise ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            Normalized and transformed data matrix with only the selected features.
        """
        check_is_fitted(self, "ref_sample_")
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
        return super().transform(Xt)

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
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, feature_meta):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features." % self.k
            )
        if not 0 <= self.pv <= 1:
            raise ValueError("pv should be 0 <= pv <= 1; got %r." % self.pv)
        if self.fc < 1:
            raise ValueError("fold change threshold should be >= 1; got %r." % self.fc)
        if self.score_type not in ("pv", "lfc_pv"):
            raise ValueError("invalid score_type %s" % self.score_type)
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

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.pv > 0:
            if self.k == "all":
                mask = np.ones_like(self.scores_, dtype=bool)
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
            elif self.k > 0:
                mask[np.argsort(self.scores_, kind="mergesort")[: self.k]] = True
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
        return mask

    def _more_tags(self):
        return {"requires_positive_X": True}


class LimmaSelector(ExtendedSelectorMixin, BaseEstimator):
    """limma differential expression feature selector for gene expression data

    Parameters
    ----------
    k : int or "all" (default = "all")
        Number of top features to select. Specifying k = "all" and pv = 1.0
        bypasses selection, for use in a parameter search. When pv is also
        specified then returns the intersection of both parameter results.

    pv : float (default = 1.0)
        Select top features below an adjusted p-value threshold. Specifying
        k = "all" and pv = 1.0 bypasses selection, for use in a parameter
        search. When k is also specified returns the intersection of both
        parameter results.

    fc : float (default = 1.0)
        treat absolute fold change minimum threshold. Default value of 1.0
        gives eBayes results.

    score_type : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    robust : bool (default = False)
        limma treat/eBayes robust option.

    trend : bool (default = False)
        limma treat/eBayes trend option.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and Batch column
        exists.

    memory : None, str or object with the joblib.Memory interface \
        (default = None)
        Used for internal caching. By default, no caching is done.
        If a string is given, it is the path to the caching directory.

    Attributes
    ----------
    scores_ : array, shape (n_features,)
        Feature scores.

    padjs_ : array, shape (n_features,)
        Feature adjusted p-values.
    """

    def __init__(
        self,
        k="all",
        pv=1,
        fc=1,
        score_type="pv",
        robust=False,
        trend=False,
        model_batch=False,
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.score_type = score_type
        self.robust = robust
        self.trend = trend
        self.model_batch = model_batch
        self.memory = memory

    def fit(self, X, y, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training gene expression data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        sample_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_samples, n_metadata)
            Training sample metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = ro.NULL
        self.scores_, self.padjs_ = memory.cache(limma_feature_score)(
            X,
            y,
            sample_meta=sample_meta,
            lfc=np.log2(self.fc),
            score_type=self.score_type,
            robust=self.robust,
            trend=self.trend,
            model_batch=self.model_batch,
        )
        return self

    def transform(self, X, sample_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Gene expression data matrix.

        sample_meta : ignored

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            Gene expression data matrix with only the selected features.
        """
        check_is_fitted(self, "scores_")
        return super().transform(X)

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
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be 0 <= k <= n_features; got %r."
                "Use k='all' to return all features." % self.k
            )
        if not 0 <= self.pv <= 1:
            raise ValueError("pv should be 0 <= pv <= 1; got %r." % self.pv)
        if self.fc < 1:
            raise ValueError("fold change threshold should be >= 1; got %r." % self.fc)

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        mask = np.zeros_like(self.scores_, dtype=bool)
        if self.pv > 0:
            if self.k == "all":
                mask = np.ones_like(self.scores_, dtype=bool)
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
            elif self.k > 0:
                mask[np.argsort(self.scores_, kind="mergesort")[: self.k]] = True
                if self.pv < 1:
                    mask[self.padjs_ > self.pv] = False
        return mask
