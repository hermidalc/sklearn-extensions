import numpy as np
from scipy import stats

from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y, safe_mask
from ..base import ExtendedTransformerMixin
from ..utils.validation import check_is_fitted


def _mean_plus_2sd(x):
    # R uses Bessel's correction stdev with n - 1 denominator
    # numpy uses n denominator set ddof=1 to get same result as R
    return np.mean(x) + 2 * np.std(x, ddof=1)


class NanoStringNormalizer(ExtendedTransformerMixin, BaseEstimator):
    """NanoStringNorm normalization methods for NanoString count data.

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
        ``mean_2sd``, ``max``, and None which skips this normalization step.
        ``mean`` is the least is the least conservative, while ``mean_2sd`` and
        ``max`` are the most robust to false positives. The calculated
        background is subtracted from each sample. Background is calculated
        after Code Count normalization.

    sample_content : str or None (default = None)
        The method used to normalize for sample or RNA content. Options are
        ``hk_geo_mean``, ``hk_sum``, ``total_sum``, ``low_cv_geo_mean``,
        ``top_mean``, ``top_geo_mean``, and None which skips this normalization
        step. Housekeeping options require a set of annotated genes. If using
        housekeeping genes then ``hk_geo_mean`` is recommended. Sample Content
        normalization is applied after Code Count normalization and Background
        correction.

    meta_col : str (default = "Code.Class")
        Feature metadata column name holding Code Class information.

    Attributes
    ----------
    pos_control_ : float, shape = (n_samples,)
        Sample positive controls.

    pos_norm_factor_ : float, shape = (n_samples,)
        Sample positive normalization factors.

    bkgrd_level_ : float, shape = (n_samples,)
        Sample background levels.

    rna_content_ : float, shape = (n_samples,)
        Sample RNA content.

    rna_norm_factor_ : float, shape = (n_samples,)
        Sample RNA content normalization factors.
    """

    def __init__(self, probe='adjust', code_count=None, background=None,
                 sample_content=None, meta_col='Code.Class'):
        self.probe = probe
        self.code_count = code_count
        self.background = background
        self.sample_content = sample_content
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
        X, y = check_X_y(X, y, dtype=int)
        self._check_params(X, y, feature_meta)
        if np.any(X < 0):
            raise ValueError('X should have only non-negative values.')
        Xt = X.copy()
        if self.code_count is not None:
            pos_mask = (feature_meta[self.meta_col].isin(['Positive'])
                        .to_numpy())
            Xt_pos = Xt[:, safe_mask(Xt, pos_mask)]
            if self.code_count == 'geo_mean':
                Xt_pos[Xt_pos < 1] = 1
                pos_control = stats.gmean(Xt_pos, axis=1)
            elif self.code_count == 'sum':
                pos_control = np.sum(Xt_pos, axis=1)
            pos_norm_factor = np.mean(pos_control) / pos_control
            Xt = (Xt.T * pos_norm_factor).T
            self.pos_control_ = pos_control
            self.pos_norm_factor_ = pos_norm_factor
        if self.background is not None:
            neg_mask = (feature_meta[self.meta_col].isin(['Negative'])
                        .to_numpy())
            Xt_neg = Xt[:, safe_mask(Xt, neg_mask)]
            if self.background == 'mean':
                bkgrd_level = np.mean(Xt_neg, axis=1)
            elif self.background == 'mean_2sd':
                bkgrd_level = np.apply_along_axis(_mean_plus_2sd, 1, Xt_neg)
            elif self.background == 'max':
                bkgrd_level = np.max(Xt_neg, axis=1)
            Xt = (Xt.T - bkgrd_level).T
            Xt[Xt < 0] = 0
            self.bkgrd_level_ = bkgrd_level
        if self.sample_content is not None:
            if self.sample_content.startswith('hk'):
                hk_mask = feature_meta[self.meta_col].isin(
                    ['Control', 'Housekeeping', 'housekeeping']).to_numpy()
                Xt_hk = Xt[:, safe_mask(Xt, hk_mask)]
                if self.sample_content == 'hk_geo_mean':
                    Xt_hk[Xt_hk < 1] = 1
                    rna_content = stats.gmean(Xt_hk, axis=1)
                elif self.sample_content == 'hk_sum':
                    rna_content = np.sum(Xt_hk, axis=1)
                rna_content[rna_content < 1] = 1
                rna_norm_factor = np.mean(rna_content) / rna_content
                Xt = (Xt.T * rna_norm_factor).T
                self.rna_content_ = rna_content
                self.rna_norm_factor_ = rna_norm_factor
        self._Xt = Xt
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
            NanoStringNorm normalized data matrix.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=int)
        if np.any(X < 0):
            raise ValueError('X should have only non-negative values.')
        if hasattr(self, '_train_done'):
            if self.code_count is not None:
                pos_mask = (feature_meta[self.meta_col].isin(['Positive'])
                            .to_numpy())
                X_pos = X[:, safe_mask(X, pos_mask)]
                if self.code_count == 'geo_mean':
                    X_pos[X_pos < 1] = 1
                    pos_control = stats.gmean(X_pos, axis=1)
                elif self.code_count == 'sum':
                    pos_control = np.sum(X_pos, axis=1)
                pos_norm_factor = np.mean(self.pos_control_) / pos_control
                X = (X.T * pos_norm_factor).T
            if self.background is not None:
                neg_mask = (feature_meta[self.meta_col].isin(['Negative'])
                            .to_numpy())
                X_neg = X[:, safe_mask(X, neg_mask)]
                if self.background == 'mean':
                    bkgrd_level = np.mean(X_neg, axis=1)
                elif self.background == 'mean_2sd':
                    bkgrd_level = np.apply_along_axis(_mean_plus_2sd, 1, X_neg)
                elif self.background == 'max':
                    bkgrd_level = np.max(X_neg, axis=1)
                X = (X.T - bkgrd_level).T
                X[X < 0] = 0
            if self.sample_content is not None:
                if self.sample_content.startswith('hk'):
                    hk_mask = feature_meta[self.meta_col].isin(
                        ['Control', 'Housekeeping', 'housekeeping']).to_numpy()
                    X_hk = X[:, safe_mask(X, hk_mask)]
                    if self.sample_content == 'hk_geo_mean':
                        X_hk[X_hk < 1] = 1
                        rna_content = stats.gmean(X_hk, axis=1)
                    elif self.sample_content == 'hk_sum':
                        rna_content = np.sum(X_hk, axis=1)
                    rna_content[rna_content < 1] = 1
                    rna_norm_factor = np.mean(self.rna_content_) / rna_content
                    X = (X.T * rna_norm_factor).T
            return X
        self._train_done = True
        return self._Xt

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
        raise NotImplementedError('inverse_transform not implemented.')

    def _more_tags(self):
        return {'requires_positive_X': True}

    def _check_params(self, X, y, feature_meta):
        if X.shape[1] != feature_meta.shape[0]:
            raise ValueError('X ({:d}) and feature_meta ({:d}) have '
                             'different feature dimensions'
                             .format(X.shape[1], feature_meta.shape[0]))
        if self.meta_col not in feature_meta.columns:
            raise ValueError('{} feature_meta column does not exist.'
                             .format(self.meta_col))
        if not feature_meta[self.meta_col].isin(['Endogenous']).any():
            raise ValueError('{} feature_meta column does not have any '
                             'Endogenous features'.format(self.meta_col))
        if (self.code_count is not None
                and not feature_meta[self.meta_col].isin(['Positive']).any()):
            raise ValueError('Code Count normalization cannot be performed '
                             'because {} feature_meta column does not have '
                             'any Positive features'.format(self.meta_col))
        if (self.background is not None
                and not feature_meta[self.meta_col].isin(['Negative']).any()):
            raise ValueError('Background correction cannot be performed '
                             'because {} feature_meta column does not have '
                             'any Negative features'.format(self.meta_col))
        if (self.sample_content is not None
                and self.sample_content.startswith('hk')
                and not feature_meta[self.meta_col].isin(
                    ['Control', 'Housekeeping', 'housekeeping']).any()):
            raise ValueError('Sample Content correction cannot be performed '
                             'because {} feature_meta column does not have '
                             'any Housekeeping features'.format(self.meta_col))
