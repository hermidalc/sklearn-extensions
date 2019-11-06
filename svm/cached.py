from sklearn.svm import LinearSVC
from ..cached import CachedFitMixin


class CachedLinearSVC(CachedFitMixin, LinearSVC):
    pass
