"""
The :mod:`sklearn.feature_selection` module implements feature selection
algorithms. It currently includes univariate filter selection methods and the
recursive feature elimination algorithm.
"""

from .cached import (
    CachedANOVAFScorerClassification, CachedChi2Scorer,
    CachedMutualInfoScorerClassification, CachedLimmaScorerClassification)
from .column import ColumnSelector
from .from_model import SelectFromModel
from .multivariate import CFS, FCBF, ReliefF
from .rna_seq import (DESeq2, DreamVoom, EdgeR, EdgeRFilterByExpr,
                      LimmaScorerClassification, LimmaVoom)
from .rfe import RFE, RFECV
from .univariate_selection import (
    ANOVAFScorerClassification, ANOVAFScorerRegression, Chi2Scorer,
    GenericUnivariateSelect, MutualInfoScorerClassification,
    MutualInfoScorerRegression, SelectFdr, SelectFpr, SelectFwe, SelectKBest,
    SelectPercentile)
from .variance_threshold import VarianceThreshold


__all__ = ['ANOVAFScorerClassification',
           'ANOVAFScorerRegression',
           'CachedANOVAFScorerClassification',
           'CachedChi2Scorer',
           'CachedLimmaScorerClassification',
           'CachedMutualInfoScorerClassification',
           'CFS',
           'Chi2Scorer',
           'ColumnSelector',
           'DESeq2',
           'DreamVoom',
           'EdgeR',
           'EdgeRFilterByExpr',
           'FCBF',
           'GenericUnivariateSelect',
           'LimmaScorerClassification',
           'LimmaVoom',
           'MutualInfoScorerClassification',
           'MutualInfoScorerRegression',
           'RFE',
           'RFECV',
           'SelectFdr',
           'SelectFpr',
           'SelectFromModel',
           'SelectFwe',
           'SelectKBest',
           'SelectPercentile',
           'ReliefF',
           'VarianceThreshold']
