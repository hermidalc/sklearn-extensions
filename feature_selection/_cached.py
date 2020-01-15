from ._univariate_selection import (
    ANOVAFScorerClassification, Chi2Scorer, MutualInfoScorerClassification)
from ..cached import CachedFitMixin


class CachedANOVAFScorerClassification(CachedFitMixin,
                                       ANOVAFScorerClassification):

    def __init__(self, memory):
        self.memory = memory


class CachedChi2Scorer(CachedFitMixin, Chi2Scorer):

    def __init__(self, memory):
        self.memory = memory


class CachedMutualInfoScorerClassification(CachedFitMixin,
                                           MutualInfoScorerClassification):

    def __init__(self, memory, discrete_features='auto', n_neighbors=3,
                 copy=True, random_state=None):
        self.memory = memory
        super().__init__(discrete_features=discrete_features,
                         n_neighbors=n_neighbors, copy=copy,
                         random_state=random_state)
