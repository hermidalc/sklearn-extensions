from ..cached import CachedFitMixin
from ._mutual_info import MutualInfoScorerClassification, MutualInfoScorerRegression
from ._univariate_selection import (
    ANOVAFScorerClassification,
    ANOVAFScorerRegression,
    Chi2Scorer,
)


class CachedANOVAFScorerClassification(CachedFitMixin, ANOVAFScorerClassification):
    def __init__(self, memory):
        self.memory = memory


class CachedANOVAFScorerRegression(CachedFitMixin, ANOVAFScorerRegression):
    def __init__(self, memory, center=True, force_finite=True):
        super().__init__(center=center, force_finite=force_finite)
        self.memory = memory


class CachedChi2Scorer(CachedFitMixin, Chi2Scorer):
    def __init__(self, memory):
        self.memory = memory


class CachedMutualInfoScorerClassification(
    CachedFitMixin, MutualInfoScorerClassification
):
    def __init__(
        self,
        memory,
        discrete_features="auto",
        n_neighbors=3,
        copy=True,
        random_state=None,
    ):
        super().__init__(
            discrete_features=discrete_features,
            n_neighbors=n_neighbors,
            copy=copy,
            random_state=random_state,
        )
        self.memory = memory


class CachedMutualInfoScorerRegression(CachedFitMixin, MutualInfoScorerRegression):
    def __init__(
        self,
        memory,
        discrete_features="auto",
        n_neighbors=3,
        copy=True,
        random_state=None,
    ):
        super().__init__(
            discrete_features=discrete_features,
            n_neighbors=n_neighbors,
            copy=copy,
            random_state=random_state,
        )
        self.memory = memory
