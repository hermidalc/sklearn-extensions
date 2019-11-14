import os
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from ..base import TransformerMixin

numpy2ri.deactivate()
pandas2ri.deactivate()
numpy2ri.activate()
pandas2ri.activate()

if 'limma_remove_ba_fit' not in robjects.globalenv:
    r_base = importr('base')
    r_base.source(os.path.dirname(__file__) + '/batch_effect.R')
r_limma_remove_ba_fit = robjects.globalenv['limma_remove_ba_fit']
r_limma_remove_ba_transform = robjects.globalenv['limma_remove_ba_transform']


class LimmaRemoveBatchEffect(BaseEstimator, TransformerMixin):
    """limma removeBatchEffect transformer for log-transformed expression data

    Parameters
    ----------
    preserve_design : bool (default = True)
        Whether batch effect correction should protect target design from
        being removed.

    Attributes
    ----------
    beta_ : array, shape (n_features, n_batches - 1)
        removeBatchEffect linear model coefficents
    """

    def __init__(self, preserve_design=True):
        self.preserve_design = preserve_design

    def fit(self, X, y, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input log-transformed data matrix.

        y : ignored

        sample_meta : pandas.DataFrame, pandas.Series \
            shape = (n_samples, n_metadata)
            Training sample metadata.
        """
        X = check_array(X)
        if sample_meta is None:
            sample_meta = robjects.NULL
            self.batch_ = None
        self.beta_ = np.array(r_limma_remove_ba_fit(
            X, sample_meta, preserve_design=self.preserve_design), dtype=float)
        return self

    def transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input log-transformed data matrix.

        sample_meta : pandas.DataFrame, pandas.Series \
            shape = (n_samples, n_metadata)
            Sample metadata.

        Returns
        -------
        Xt : array of shape (n_samples, n_features)
            Batched corrected log-transformed input data matrix.
        """
        check_is_fitted(self, 'beta_')
        X = check_array(X)
        X = np.array(r_limma_remove_ba_transform(
            X, sample_meta, beta=self.beta_), dtype=float)
        return X

    def inverse_transform(self, X, sample_meta):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Input batched corrected log-transformed data matrix.

        sample_meta : Ignored.

        Returns
        -------
        Xr : array of shape (n_samples, n_original_features)
        """
        raise NotImplementedError("inverse_transform not implemented.")
