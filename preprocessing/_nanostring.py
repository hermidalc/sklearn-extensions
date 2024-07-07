import os
import numpy as np
from scipy import stats
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils import safe_mask
from sklearn.utils.validation import check_is_fitted

from ..base import ExtendedTransformerMixin

numpy2ri.deactivate()
pandas2ri.deactivate()
numpy2ri.activate()
pandas2ri.activate()

if "nanostringdiff_fit" not in robjects.globalenv:
    r_base = importr("base")
    r_base.source(os.path.dirname(__file__) + "/_nanostring.R")
r_nanostringdiff_fit = robjects.globalenv["nanostringdiff_fit"]
r_nanostringdiff_transform = robjects.globalenv["nanostringdiff_transform"]


def _mean_plus_2sd(x):
    # R uses Bessel's correction stdev with n - 1 denominator
    # numpy uses n denominator set ddof=1 to get same result as R
    return np.mean(x) + 2 * np.std(x, ddof=1)


class NanoStringNormalizer(ExtendedTransformerMixin, BaseEstimator):
    """NanoString nSolver/NanoStringNorm standard normalization methods.

    Parameters
    ----------
    probe : str (default = "adjust")
        Probe correction factor to be applied at the probe level prior to any
        normalization. Options are ``adjust`` and ``filter``. ``adjust`` uses
        the nCounter flagged probe correction factors concatenated to the gene
        names. Specify ``filter`` if you would like to remove all flagged
        genes.

    code_count : str or None (default = None)
        The method used to normalize for technical assay variation. Options
        are ``geo_mean``, ``sum``, and None which skips this normalization
        step. These step adjusts each sample based on its relative value to
        all the samples. ``geo_mean`` may be less susceptible to extreme
        values. Code Count normalization is applied first and is considered the
        most fundamental normalization step for technical variation.

    background : str or None (default = None)
        The method used to estimate the background count level. Background is
        calculated based on negative controls. Options are ``mean``,
        ``mean_2sd``, ``max``, ``geo_mean`` and None which skips this
        normalization step. ``mean`` and ``geo_mean`` are the least
        conservative, while ``mean_2sd``, ``max`` are the most robust to false
        positives. The calculated background is subtracted or used as a
        minimum threshold depending on the ``background_threshold`` flag.
        Background is calculated after Code Count normalization.

    sample_content : str or None (default = None)
        The method used to normalize for sample or RNA content. Options are
        ``hk_geo_mean``, ``hk_sum``, ``total_sum``, ``low_cv_geo_mean``,
        ``top_mean``, ``top_geo_mean``, and None which skips this normalization
        step. Housekeeping options require a set of annotated genes. If using
        housekeeping genes then ``hk_geo_mean`` is recommended. Sample Content
        normalization is applied after Code Count normalization and Background
        correction.

    background_threshold : bool (default = True)
        Flag whether calculated ``background`` should used as a minimum
        threshold or subtracted.

    round_values : bool (default = True)
        Whether final normalized data should be rounded to the nearest integer.
        This simplifies interpretation if data will be log transformed later by
        adjusting values between 0-1 which would result in negative logs.

    meta_col : str (default = "Code.Class")
        Feature metadata column name holding Code Class information.

    Attributes
    ----------
    pos_control_ : float, shape = (n_samples,)
        Summarized positive control counts.

    pos_norm_factor_ : float, shape = (n_samples,)
        Summarized positive control count normalization factors.

    bkgrd_level_ : float, shape = (n_samples,)
        Summarized background levels.

    rna_content_ : float, shape = (n_samples,)
        Summarized RNA content.

    rna_norm_factor_ : float, shape = (n_samples,)
        Summarized RNA content normalization factors.
    """

    def __init__(
        self,
        probe="adjust",
        code_count=None,
        background=None,
        sample_content=None,
        background_threshold=True,
        round_values=True,
        meta_col="Code.Class",
    ):
        self.probe = probe
        self.code_count = code_count
        self.background = background
        self.sample_content = sample_content
        self.background_threshold = background_threshold
        self.round_values = round_values
        self.meta_col = meta_col

    def fit(self, X, y, feature_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        feature_meta : pandas.DataFrame, pandas.Series \
            shape = (n_features, n_metadata)
            Feature metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, y, feature_meta)
        self.Xt_ = self._fit_transform(X.copy(), feature_meta, in_fit=True)
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
            Normalized data matrix.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype=int)
        if hasattr(self, "_train_done"):
            return self._fit_transform(X, feature_meta, in_fit=False)
        self._train_done = True
        return self.Xt_

    def inverse_transform(self, X, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        feature_meta : Ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _more_tags(self):
        return {"requires_positive_X": True}

    def _fit_transform(self, X, feature_meta, in_fit):
        if np.any(X < 0):
            raise ValueError("X should have only non-negative values.")
        if self.code_count is not None:
            pos_mask = feature_meta[self.meta_col].isin(["Positive"]).to_numpy()
            X_pos = X[:, safe_mask(X, pos_mask)]
            if self.code_count == "geo_mean":
                X_pos[X_pos < 1] = 1
                pos_control = stats.gmean(X_pos, axis=1)
            elif self.code_count == "sum":
                pos_control = np.sum(X_pos, axis=1)
            if in_fit:
                pos_norm_factor = np.mean(pos_control) / pos_control
                self.pos_control_ = pos_control
                self.pos_norm_factor_ = pos_norm_factor
            else:
                pos_norm_factor = np.array(
                    [np.mean(np.append(self.pos_control_, p)) / p for p in pos_control]
                )
            X = X * pos_norm_factor[:, np.newaxis]
        if self.background is not None:
            neg_mask = feature_meta[self.meta_col].isin(["Negative"]).to_numpy()
            X_neg = X[:, safe_mask(X, neg_mask)]
            if self.background == "mean":
                bkgrd_level = np.mean(X_neg, axis=1)
            elif self.background == "mean_2sd":
                bkgrd_level = np.apply_along_axis(_mean_plus_2sd, 1, X_neg)
            elif self.background == "max":
                bkgrd_level = np.max(X_neg, axis=1)
            elif self.background == "geo_mean":
                X_neg[X_neg < 1] = 1
                bkgrd_level = stats.gmean(X_neg, axis=1)
            if self.background_threshold:
                X = X.clip(bkgrd_level[:, np.newaxis])
            else:
                X = X - bkgrd_level[:, np.newaxis]
                X[X < 0] = 0
            if in_fit:
                self.bkgrd_level_ = bkgrd_level
        if self.sample_content is not None:
            if self.sample_content.startswith("hk"):
                hk_mask = (
                    feature_meta[self.meta_col]
                    .isin(["Control", "Housekeeping", "housekeeping"])
                    .to_numpy()
                )
                X_hk = X[:, safe_mask(X, hk_mask)]
                if self.sample_content == "hk_geo_mean":
                    X_hk[X_hk < 1] = 1
                    rna_content = stats.gmean(X_hk, axis=1)
                elif self.sample_content == "hk_sum":
                    rna_content = np.sum(X_hk, axis=1)
                rna_content[rna_content < 1] = 1
                if in_fit:
                    rna_norm_factor = np.mean(rna_content) / rna_content
                    self.rna_content_ = rna_content
                    self.rna_norm_factor_ = rna_norm_factor
                else:
                    rna_norm_factor = np.array(
                        [
                            np.mean(np.append(self.rna_content_, r)) / r
                            for r in rna_content
                        ]
                    )
                X = X * rna_norm_factor[:, np.newaxis]
        if self.round_values:
            X = np.round(X).astype(int)
        return X

    def _check_params(self, X, y, feature_meta):
        if X.shape[1] != feature_meta.shape[0]:
            raise ValueError(
                "X ({:d}) and feature_meta ({:d}) have "
                "different feature dimensions".format(X.shape[1], feature_meta.shape[0])
            )
        if self.meta_col not in feature_meta.columns:
            raise ValueError(
                "{} feature_meta column does not exist.".format(self.meta_col)
            )
        if not feature_meta[self.meta_col].isin(["Endogenous"]).any():
            raise ValueError(
                "{} feature_meta column does not have any "
                "Endogenous features".format(self.meta_col)
            )
        if (
            self.code_count is not None
            and not feature_meta[self.meta_col].isin(["Positive"]).any()
        ):
            raise ValueError(
                "Code Count normalization cannot be performed "
                "because {} feature_meta column does not have "
                "any Positive features".format(self.meta_col)
            )
        if (
            self.background is not None
            and not feature_meta[self.meta_col].isin(["Negative"]).any()
        ):
            raise ValueError(
                "Background correction cannot be performed "
                "because {} feature_meta column does not have "
                "any Negative features".format(self.meta_col)
            )
        if (
            self.sample_content is not None
            and self.sample_content.startswith("hk")
            and not feature_meta[self.meta_col]
            .isin(["Control", "Housekeeping", "housekeeping"])
            .any()
        ):
            raise ValueError(
                "Sample Content correction cannot be performed "
                "because {} feature_meta column does not have "
                "any Housekeeping features".format(self.meta_col)
            )


class NanoStringDiffNormalizer(ExtendedTransformerMixin, BaseEstimator):
    """NanoStringDiff normalization method.

    Parameters
    ----------
    background_threshold : bool (default = True)
        Flag whether calculated ``background`` should used as a minimum
        threshold or subtracted.

    meta_col : str (default = "Code.Class")
        Feature metadata column name holding Code Class information.

    Attributes
    ----------
    positive_factor_ : float, shape = (n_samples,)
        Positive control normalization factors.

    negative_factor_ : float, shape = (n_samples,)
        Negative control normalization factors.

    housekeeping_factor_ : float, shape = (n_samples,)
        Housekeeping control normalization factors.
    """

    def __init__(self, background_threshold=True, meta_col="Code.Class"):
        self.background_threshold = background_threshold
        self.meta_col = meta_col

    def fit(self, X, y, feature_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training counts data matrix.

        y : array-like, shape = (n_samples,)
            Training class labels.

        feature_meta : pandas.DataFrame, pandas.Series \
            shape = (n_features, n_metadata)
            Feature metadata.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(X, y, dtype=int)
        self._check_params(X, y, feature_meta)
        (
            self.positive_factor_,
            self.negative_factor_,
            self.housekeeping_factor_,
        ) = r_nanostringdiff_fit(X, y, feature_meta, meta_col=self.meta_col)
        return self

    def transform(self, X, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input counts data matrix.

        feature_meta : ignored

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            Normalized data matrix.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype=int)
        X = np.array(
            r_nanostringdiff_transform(
                X,
                self.positive_factor_,
                self.negative_factor_,
                self.housekeeping_factor_,
                background_threshold=self.background_threshold,
            ),
            dtype=int,
        )
        return X

    def inverse_transform(self, X, feature_meta=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input transformed data matrix.

        feature_meta : Ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError("inverse_transform not implemented.")

    def _more_tags(self):
        return {"requires_positive_X": True}

    def _check_params(self, X, y, feature_meta):
        if X.shape[1] != feature_meta.shape[0]:
            raise ValueError(
                "X ({:d}) and feature_meta ({:d}) have "
                "different feature dimensions".format(X.shape[1], feature_meta.shape[0])
            )
        if self.meta_col not in feature_meta.columns:
            raise ValueError(
                "{} feature_meta column does not exist.".format(self.meta_col)
            )
        if not feature_meta[self.meta_col].isin(["Endogenous"]).any():
            raise ValueError(
                "{} feature_meta column does not have any "
                "Endogenous features".format(self.meta_col)
            )
        if not feature_meta[self.meta_col].isin(["Positive"]).any():
            raise ValueError(
                "Code Count normalization cannot be performed "
                "because {} feature_meta column does not have "
                "any Positive features".format(self.meta_col)
            )
        if not feature_meta[self.meta_col].isin(["Negative"]).any():
            raise ValueError(
                "Background correction cannot be performed "
                "because {} feature_meta column does not have "
                "any Negative features".format(self.meta_col)
            )
        if (
            not feature_meta[self.meta_col]
            .isin(["Control", "Housekeeping", "housekeeping"])
            .any()
        ):
            raise ValueError(
                "Sample Content correction cannot be performed "
                "because {} feature_meta column does not have "
                "any Housekeeping features".format(self.meta_col)
            )
