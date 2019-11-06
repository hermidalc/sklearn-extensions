from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier)
from ..cached import CachedFitMixin


class CachedGradientBoostingClassifier(CachedFitMixin,
                                       GradientBoostingClassifier):
    pass


class CachedRandomForestClassifier(CachedFitMixin, RandomForestClassifier):
    pass


class CachedExtraTreesClassifier(CachedFitMixin, ExtraTreesClassifier):
    pass
