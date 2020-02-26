"""
The :mod:`sklearn.feature_selection` module implements feature selection
algorithms. It currently includes univariate filter selection methods and the
recursive feature elimination algorithm.
"""

from ._cached import (
    CachedANOVAFScorerClassification, CachedChi2Scorer,
    CachedMutualInfoScorerClassification)
from ._column import ColumnSelector
from ._from_model import SelectFromModel
from ._multivariate import CFS, FCBF, ReliefF
from ._rna_seq import (DESeq2, DreamVoom, EdgeR, EdgeRFilterByExpr, Limma,
                       LimmaVoom)
from ._rfe import RFE, RFECV
from ._univariate_selection import (
    ANOVAFScorerClassification, ANOVAFScorerRegression, Chi2Scorer,
    GenericUnivariateSelect, MutualInfoScorerClassification,
    MutualInfoScorerRegression, SelectFdr, SelectFpr, SelectFwe, SelectKBest,
    SelectPercentile)
from ._variance_threshold import VarianceThreshold


__all__ = ['ANOVAFScorerClassification',
           'ANOVAFScorerRegression',
           'CachedANOVAFScorerClassification',
           'CachedChi2Scorer',
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
           'Limma',
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
