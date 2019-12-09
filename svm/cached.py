from sklearn.svm import LinearSVC
from ..cached import CachedFitMixin


class CachedLinearSVC(CachedFitMixin, LinearSVC):

    def __init__(self, memory, penalty='l2', loss='squared_hinge', dual=True,
                 tol=1e-2, C=1.0, multi_class='ovr', fit_intercept=True,
                 intercept_scaling=1, class_weight=None, verbose=0,
                 random_state=None, max_iter=1000):
        self.memory = memory
        super().__init__(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C,
                         multi_class=multi_class, fit_intercept=fit_intercept,
                         intercept_scaling=intercept_scaling,
                         class_weight=class_weight, verbose=verbose,
                         random_state=random_state, max_iter=max_iter)
