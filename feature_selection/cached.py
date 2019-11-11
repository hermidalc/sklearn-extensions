from .univariate_selection import (
    ANOVAFScorerClassification, Chi2Scorer, MutualInfoScorerClassification)
from .rna_seq import LimmaScorerClassification
from ..cached import CachedFitMixin


class CachedANOVAFScorerClassification(CachedFitMixin,
                                       ANOVAFScorerClassification):
    pass


class CachedChi2Scorer(CachedFitMixin, Chi2Scorer):
    pass


class CachedLimmaScorerClassification(CachedFitMixin,
                                      LimmaScorerClassification):
    pass


class CachedMutualInfoScorerClassification(CachedFitMixin,
                                           MutualInfoScorerClassification):
    pass
