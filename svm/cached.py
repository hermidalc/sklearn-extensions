from sklearn.svm import LinearSVC, SVC
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


class CachedSVC(CachedFitMixin, SVC):

    def __init__(self, memory, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, shrinking=True, probability=False, tol=1e-3,
                 cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', break_ties=False,
                 random_state=None):
        self.memory = memory
        super().__init__(C=C, kernel=kernel, degree=degree, gamma=gamma,
                         coef0=coef0, shrinking=shrinking,
                         probability=probability, tol=tol,
                         cache_size=cache_size, class_weight=class_weight,
                         verbose=verbose, max_iter=max_iter,
                         decision_function_shape=decision_function_shape,
                         break_ties=break_ties, random_state=random_state)
