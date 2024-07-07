import os

import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_memory

from ._base import ExtendedSelectorMixin

numpy2ri.deactivate()
pandas2ri.deactivate()
numpy2ri.activate()
pandas2ri.activate()

r_base = importr("base")
if "deseq2_feature_score" not in robjects.globalenv:
    r_base.source(os.path.dirname(__file__) + "/_rna_seq.R")
r_deseq2_feature_score = robjects.globalenv["deseq2_feature_score"]
r_edger_feature_score = robjects.globalenv["edger_feature_score"]
r_edger_filterbyexpr_mask = robjects.globalenv["edger_filterbyexpr_mask"]
r_limma_voom_feature_score = robjects.globalenv["limma_voom_feature_score"]
r_dream_voom_feature_score = robjects.globalenv["dream_voom_feature_score"]
r_limma_feature_score = robjects.globalenv["limma_feature_score"]
if "deseq2_rle_fit" not in robjects.globalenv:
    r_base.source(os.path.dirname(__file__) + "/../preprocessing/_rna_seq.R")
r_deseq2_rle_fit = robjects.globalenv["deseq2_rle_fit"]
r_deseq2_rle_vst_transform = robjects.globalenv["deseq2_rle_vst_transform"]
r_edger_tmm_fit = robjects.globalenv["edger_tmm_fit"]
r_edger_tmm_cpm_transform = robjects.globalenv["edger_tmm_cpm_transform"]
r_edger_tmm_tpm_transform = robjects.globalenv["edger_tmm_tpm_transform"]


def deseq2_feature_score(
    X, y, sample_meta, lfc, scoring_meth, fit_type, lfc_shrink, model_batch, n_threads
):
    sc, pa = r_deseq2_feature_score(
        X,
        y,
        sample_meta=sample_meta,
        lfc=lfc,
        scoring_meth=scoring_meth,
        fit_type=fit_type,
        lfc_shrink=lfc_shrink,
        model_batch=model_batch,
        n_threads=n_threads,
    )
    return np.array(sc, dtype=float), np.array(pa, dtype=float)


def edger_feature_score(X, y, sample_meta, lfc, scoring_meth, robust, model_batch):
    sc, pa = r_edger_feature_score(
        X,
        y,
        sample_meta=sample_meta,
        lfc=lfc,
        scoring_meth=scoring_meth,
        robust=robust,
        model_batch=model_batch,
    )
    return np.array(sc, dtype=float), np.array(pa, dtype=float)


def limma_voom_feature_score(
    X, y, sample_meta, lfc, scoring_meth, robust, model_batch, model_dupcor
):
    sc, pa = r_limma_voom_feature_score(
        X,
        y,
        sample_meta=sample_meta,
        lfc=lfc,
        scoring_meth=scoring_meth,
        robust=robust,
        model_batch=model_batch,
        model_dupcor=model_dupcor,
    )
    return np.array(sc, dtype=float), np.array(pa, dtype=float)


def dream_voom_feature_score(
    X, y, sample_meta, lfc, scoring_meth, model_batch, n_threads
):
    sc, pa = r_dream_voom_feature_score(
        X,
        y,
        sample_meta,
        lfc=lfc,
        scoring_meth=scoring_meth,
        model_batch=model_batch,
        n_threads=n_threads,
    )
    return np.array(sc, dtype=float), np.array(pa, dtype=float)


def limma_feature_score(
    X, y, sample_meta, lfc, scoring_meth, robust, trend, model_batch
):
    sc, pa = r_limma_feature_score(
        X,
        y,
        sample_meta=sample_meta,
        lfc=lfc,
        scoring_meth=scoring_meth,
        robust=robust,
        trend=trend,
        model_batch=model_batch,
    )
    return np.array(sc, dtype=float), np.array(pa, dtype=float)


def deseq2_rle_fit(X, y, sample_meta, fit_type, is_classif, model_batch):
    gm, df = r_deseq2_rle_fit(
        X,
        y=y,
        sample_meta=sample_meta,
        fit_type=fit_type,
        is_classif=is_classif,
        model_batch=model_batch,
    )
    return np.array(gm, dtype=float), df


def deseq2_rle_vst_transform(X, geo_means, disp_func):
    return np.array(
        r_deseq2_rle_vst_transform(X, geo_means=geo_means, disp_func=disp_func),
        dtype=float,
    )


def edger_tmm_fit(X):
    return np.array(r_edger_tmm_fit(X), dtype=int)


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


class DESeq2(ExtendedSelectorMixin, BaseEstimator):
    """DESeq2 differential expression feature selector and
    normalizer/transformer for RNA-seq count data

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

    scoring_meth : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    fit_type : str (default = "parametric")
        estimateDispersions fitType option.

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
        RLE normalization dispersion function.
    """

    def __init__(
        self,
        k="all",
        pv=1,
        fc=1,
        scoring_meth="pv",
        fit_type="parametric",
        lfc_shrink=True,
        model_batch=False,
        n_threads=1,
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.scoring_meth = scoring_meth
        self.fit_type = fit_type
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
        self._check_params(X, y)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = robjects.NULL
        self.scores_, self.padjs_ = memory.cache(deseq2_feature_score)(
            X,
            y,
            sample_meta=sample_meta,
            lfc=np.log2(self.fc),
            scoring_meth=self.scoring_meth,
            fit_type=self.fit_type,
            lfc_shrink=self.lfc_shrink,
            model_batch=self.model_batch,
            n_threads=self.n_threads,
        )
        self.geo_means_, self.disp_func_ = memory.cache(deseq2_rle_fit)(
            X,
            y,
            sample_meta=sample_meta,
            fit_type=self.fit_type,
            is_classif=True,
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
        Xr : array of shape (n_samples, n_selected_features)
            DESeq2 median-of-ratios normalized VST transformed data matrix
            with only the selected features.
        """
        check_is_fitted(self, "geo_means_")
        X = self._validate_data(X, reset=False, dtype=int)
        memory = check_memory(self.memory)
        X = memory.cache(deseq2_rle_vst_transform)(
            X, geo_means=self.geo_means_, disp_func=self.disp_func_
        )
        return super().transform(X)

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
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _more_tags(self):
        return {"requires_positive_X": True}

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


class EdgeRFilterByExpr(ExtendedSelectorMixin, BaseEstimator):
    """edgeR filterByExpr feature selector for count data

    Parameters
    ----------
    is_classif : bool (default = True)
        Whether this is a classification design.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and Batch column
        exists.
    """

    def __init__(self, is_classif=True, model_batch=False):
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
            y = robjects.NULL
        if sample_meta is None:
            sample_meta = robjects.NULL
        self._mask = np.array(
            r_edger_filterbyexpr_mask(
                X,
                y=y,
                sample_meta=sample_meta,
                model_batch=self.model_batch,
                is_classif=self.is_classif,
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

        sample_meta : Ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            edgeR filterByExpr counts data matrix with only the selected
            features.
        """
        check_is_fitted(self, "_mask")
        X = self._validate_data(X, reset=False, dtype=int)
        return super().transform(X)

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
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _more_tags(self):
        return {"requires_positive_X": True}

    def _get_support_mask(self):
        check_is_fitted(self, "_mask")
        return self._mask


class EdgeR(ExtendedSelectorMixin, BaseEstimator):
    """edgeR differential expression feature selector and
    normalizer/transformer for RNA-seq count data

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

    scoring_meth : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    robust : bool (default = True)
        estimateDisp and glmQLFit robust option.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and "Batch" column
        exists.

    transform_meth : str (default = "cpm")
        Transformation method to use on count data after differential
        expression testing. Available methods are "cpm" and "tpm".

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
        scoring_meth="pv",
        robust=True,
        model_batch=False,
        transform_meth="cpm",
        log=True,
        prior_count=2,
        gene_length_col="Length",
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.scoring_meth = scoring_meth
        self.robust = robust
        self.model_batch = model_batch
        self.transform_meth = transform_meth
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

        feature_meta : Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = robjects.NULL
        self.scores_, self.padjs_ = memory.cache(edger_feature_score)(
            X,
            y,
            sample_meta=sample_meta,
            lfc=np.log2(self.fc),
            scoring_meth=self.scoring_meth,
            robust=self.robust,
            model_batch=self.model_batch,
        )
        self.ref_sample_ = memory.cache(edger_tmm_fit)(X)
        return self

    def transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : Ignored.

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata for "tpm" transform, otherwise ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            edgeR TMM normalized CPM/TPM transformed data matrix with only the
            selected features.
        """
        check_is_fitted(self, "ref_sample_")
        X = self._validate_data(X, reset=False, dtype=int)
        memory = check_memory(self.memory)
        if feature_meta is None:
            feature_meta = robjects.NULL
        if self.transform_meth == "cpm":
            X = memory.cache(edger_tmm_cpm_transform)(
                X,
                ref_sample=self.ref_sample_,
                log=self.log,
                prior_count=self.prior_count,
            )
        else:
            X = memory.cache(edger_tmm_tpm_transform)(
                X,
                feature_meta=feature_meta,
                ref_sample=self.ref_sample_,
                log=self.log,
                prior_count=self.prior_count,
                gene_length_col=self.gene_length_col,
            )
        return super().transform(X)

    def inverse_transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta : Ignored.

        feature_meta: Ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _more_tags(self):
        return {"requires_positive_X": True}

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
        if self.scoring_meth not in ("pv", "lfc_pv"):
            raise ValueError("invalid scoring_meth %s" % self.scoring_meth)
        if self.transform_meth not in ("cpm", "tpm"):
            raise ValueError("invalid transform_meth %s" % self.transform_meth)

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


class LimmaVoom(ExtendedSelectorMixin, BaseEstimator):
    """limma-voom differential expression feature selector and
    normalizer/transformer for RNA-seq count data

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

    scoring_meth : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    robust : bool (default = True)
        limma treat/eBayes robust option.

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and "Batch" column
        exists.

    model_dupcor : bool (default = False)
        Model limma duplicateCorrelation if sample_meta passed to fit and Group
        column exists.

    transform_meth : str (default = "cpm")
        Transformation method to use on count data after differential
        expression testing. Available methods are "cpm" and "tpm".

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
        scoring_meth="pv",
        robust=True,
        model_batch=False,
        model_dupcor=False,
        transform_meth="cpm",
        log=True,
        prior_count=2,
        gene_length_col="Length",
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.scoring_meth = scoring_meth
        self.robust = robust
        self.model_batch = model_batch
        self.model_dupcor = model_dupcor
        self.transform_meth = transform_meth
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

        feature_meta : Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        if sample_meta is None:
            sample_meta = robjects.NULL
        self.scores_, self.padjs_ = memory.cache(limma_voom_feature_score)(
            X,
            y,
            sample_meta=sample_meta,
            lfc=np.log2(self.fc),
            scoring_meth=self.scoring_meth,
            robust=self.robust,
            model_batch=self.model_batch,
            model_dupcor=self.model_dupcor,
        )
        self.ref_sample_ = memory.cache(edger_tmm_fit)(X)
        return self

    def transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : Ignored.

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata for "tpm" transform, otherwise ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            edgeR TMM normalized CPM/TPM transformed data matrix with only the
            selected features.
        """
        check_is_fitted(self, "ref_sample_")
        X = self._validate_data(X, reset=False, dtype=int)
        memory = check_memory(self.memory)
        if feature_meta is None:
            feature_meta = robjects.NULL
        if self.transform_meth == "cpm":
            X = memory.cache(edger_tmm_cpm_transform)(
                X,
                ref_sample=self.ref_sample_,
                log=self.log,
                prior_count=self.prior_count,
            )
        else:
            X = memory.cache(edger_tmm_tpm_transform)(
                X,
                feature_meta=feature_meta,
                ref_sample=self.ref_sample_,
                log=self.log,
                prior_count=self.prior_count,
                gene_length_col=self.gene_length_col,
            )
        return super().transform(X)

    def inverse_transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta : Ignored.

        feature_meta : Ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _more_tags(self):
        return {"requires_positive_X": True}

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
        if self.scoring_meth not in ("pv", "lfc_pv"):
            raise ValueError("invalid scoring_meth %s" % self.scoring_meth)
        if self.transform_meth not in ("cpm", "tpm"):
            raise ValueError("invalid transform_meth %s" % self.transform_meth)

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


class DreamVoom(ExtendedSelectorMixin, BaseEstimator):
    """dream limma-voom differential expression feature selector and
    normalizer/transformer for RNA-seq count data repeated measures designs

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

    scoring_meth : str (default = "pv")
        Differential expression analysis feature scoring method. Available
        methods are "pv" or "lfc_pv".

    model_batch : bool (default = False)
        Model batch effect if sample_meta passed to fit and "Batch" column
        exists.

    n_threads : int (default = 1)
        Number of dream parallel threads. This should be carefully selected
        when using within Grid/RandomizedSearchCV to not oversubscribe CPU
        and memory resources.

    transform_meth : str (default = "cpm")
        Transformation method to use on count data after differential
        expression testing. Available methods are "cpm" and "tpm".

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
        scoring_meth="pv",
        model_batch=False,
        n_threads=1,
        transform_meth="cpm",
        log=True,
        prior_count=2,
        gene_length_col="Length",
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.scoring_meth = scoring_meth
        self.model_batch = model_batch
        self.n_threads = n_threads
        self.transform_meth = transform_meth
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

        feature_meta : Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, y)
        memory = check_memory(self.memory)
        self.scores_, self.padjs_ = memory.cache(dream_voom_feature_score)(
            X,
            y,
            sample_meta,
            lfc=np.log2(self.fc),
            scoring_meth=self.scoring_meth,
            model_batch=self.model_batch,
            n_threads=self.n_threads,
        )
        self.ref_sample_ = memory.cache(edger_tmm_fit)(X)
        return self

    def transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        sample_meta : Ignored.

        feature_meta : pandas.DataFrame, pandas.Series (default = None), \
            shape = (n_features, n_metadata)
            Feature metadata for "tpm" transform, otherwise ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_selected_features)
            edgeR TMM normalized CPM/TPM transformed data matrix with only the
            selected features.
        """
        check_is_fitted(self, "ref_sample_")
        X = self._validate_data(X, reset=False, dtype=int)
        memory = check_memory(self.memory)
        if feature_meta is None:
            feature_meta = robjects.NULL
        if self.transform_meth == "cpm":
            X = memory.cache(edger_tmm_cpm_transform)(
                X,
                ref_sample=self.ref_sample_,
                log=self.log,
                prior_count=self.prior_count,
            )
        else:
            X = memory.cache(edger_tmm_tpm_transform)(
                X,
                feature_meta=feature_meta,
                ref_sample=self.ref_sample_,
                log=self.log,
                prior_count=self.prior_count,
                gene_length_col=self.gene_length_col,
            )
        return super().transform(X)

    def inverse_transform(self, X, sample_meta=None, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        sample_meta : Ignored.

        feature_meta : Ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _more_tags(self):
        return {"requires_positive_X": True}

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
        if self.scoring_meth not in ("pv", "lfc_pv"):
            raise ValueError("invalid scoring_meth %s" % self.scoring_meth)
        if self.transform_meth not in ("cpm", "tpm"):
            raise ValueError("invalid transform_meth %s" % self.transform_meth)

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


class Limma(ExtendedSelectorMixin, BaseEstimator):
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

    scoring_meth : str (default = "pv")
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
        scoring_meth="pv",
        robust=False,
        trend=False,
        model_batch=False,
        memory=None,
    ):
        self.k = k
        self.pv = pv
        self.fc = fc
        self.scoring_meth = scoring_meth
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
            sample_meta = robjects.NULL
        self.scores_, self.padjs_ = memory.cache(limma_feature_score)(
            X,
            y,
            sample_meta=sample_meta,
            lfc=np.log2(self.fc),
            scoring_meth=self.scoring_meth,
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

        sample_meta : Ignored.

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

        sample_meta : Ignored.

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
