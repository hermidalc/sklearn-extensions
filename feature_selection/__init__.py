"""
The :mod:`sklearn.feature_selection` module implements feature selection
algorithms. It currently includes univariate filter selection methods and the
recursive feature elimination algorithm.
"""

from ._base import ExtendedSelectorMixin
from ._cached import (
    CachedANOVAFScorerClassification,
    CachedANOVAFScorerRegression,
    CachedChi2Scorer,
    CachedMutualInfoScorerClassification,
    CachedMutualInfoScorerRegression,
)
from ._column import ColumnSelector
from ._custom_threshold import (
    ConfidenceThreshold,
    CorrelationThreshold,
    MeanThreshold,
    MedianThreshold,
)
from ._mutual_info import MutualInfoScorerClassification, MutualInfoScorerRegression
from ._from_model import SelectFromModel
from ._multivariate import CFS, FCBF, ReliefF
from ._nanostring import NanoStringEndogenousSelector
from ._rna_seq import (
    CountThreshold,
    DESeq2,
    DESeq2ZINBWaVE,
    DreamVoom,
    EdgeR,
    EdgeRZINBWaVE,
    EdgeRFilterByExpr,
    Limma,
    LimmaVoom,
)
from ._rfe import ExtendedRFE, ExtendedRFECV
from ._univariate_model import SelectFromUnivariateModel
from ._univariate_selection import (
    ANOVAFScorerClassification,
    ANOVAFScorerRegression,
    Chi2Scorer,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
)
from ._variance_threshold import VarianceThreshold


__all__ = [
    "ANOVAFScorerClassification",
    "ANOVAFScorerRegression",
    "CachedANOVAFScorerClassification",
    "CachedANOVAFScorerRegression",
    "CachedChi2Scorer",
    "CachedMutualInfoScorerClassification",
    "CachedMutualInfoScorerRegression",
    "CFS",
    "Chi2Scorer",
    "ColumnSelector",
    "ConfidenceThreshold",
    "CorrelationThreshold",
    "CountThreshold",
    "DESeq2",
    "DESeq2ZINBWaVE",
    "DreamVoom",
    "EdgeR",
    "EdgeRZINBWaVE",
    "EdgeRFilterByExpr",
    "ExtendedSelectorMixin",
    "FCBF",
    "GenericUnivariateSelect",
    "Limma",
    "LimmaVoom",
    "MeanThreshold",
    "MedianThreshold",
    "MutualInfoScorerClassification",
    "MutualInfoScorerRegression",
    "NanoStringEndogenousSelector",
    "ExtendedRFE",
    "ExtendedRFECV",
    "SelectFdr",
    "SelectFpr",
    "SelectFromModel",
    "SelectFromUnivariateModel",
    "SelectFwe",
    "SelectKBest",
    "SelectPercentile",
    "ReliefF",
    "VarianceThreshold",
]
