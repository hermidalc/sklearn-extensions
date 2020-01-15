from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier)
from ..cached import CachedFitMixin


class CachedGradientBoostingClassifier(CachedFitMixin,
                                       GradientBoostingClassifier):

    def __init__(self, memory, loss='deviance', learning_rate=0.1,
                 n_estimators=100, subsample=1.0, criterion='friedman_mse',
                 min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_depth=3,
                 min_impurity_decrease=0., min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False, presort='deprecated',
                 validation_fraction=0.1, n_iter_no_change=None, tol=1e-4,
                 ccp_alpha=0.0):
        self.memory = memory
        super().__init__(loss=loss, learning_rate=learning_rate,
                         n_estimators=n_estimators, criterion=criterion,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_depth=max_depth, init=init, subsample=subsample,
                         max_features=max_features, random_state=random_state,
                         verbose=verbose, max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease,
                         min_impurity_split=min_impurity_split,
                         warm_start=warm_start, presort=presort,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change, tol=tol,
                         ccp_alpha=ccp_alpha)


class CachedRandomForestClassifier(CachedFitMixin, RandomForestClassifier):

    def __init__(self, memory, n_estimators=100, criterion='gini',
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features='auto',
                 max_leaf_nodes=None, min_impurity_decrease=0.,
                 min_impurity_split=None, bootstrap=True, oob_score=False,
                 n_jobs=None, random_state=None, verbose=0, warm_start=False,
                 class_weight=None, ccp_alpha=0.0, max_samples=None):
        self.memory = memory
        super().__init__(n_estimators=n_estimators, criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease,
                         min_impurity_split=min_impurity_split,
                         bootstrap=bootstrap, oob_score=oob_score,
                         n_jobs=n_jobs, random_state=random_state,
                         verbose=verbose, warm_start=warm_start,
                         class_weight=class_weight, ccp_alpha=ccp_alpha,
                         max_samples=max_samples)


class CachedExtraTreesClassifier(CachedFitMixin, ExtraTreesClassifier):

    def __init__(self, memory, n_estimators=100, criterion='gini',
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features='auto',
                 max_leaf_nodes=None, min_impurity_decrease=0.,
                 min_impurity_split=None, bootstrap=False, oob_score=False,
                 n_jobs=None, random_state=None, verbose=0, warm_start=False,
                 class_weight=None, ccp_alpha=0.0, max_samples=None):
        self.memory = memory
        super().__init__(n_estimators=n_estimators, criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease,
                         min_impurity_split=min_impurity_split,
                         bootstrap=bootstrap, oob_score=oob_score,
                         n_jobs=n_jobs, random_state=random_state,
                         verbose=verbose, warm_start=warm_start,
                         class_weight=class_weight, ccp_alpha=ccp_alpha,
                         max_samples=max_samples)
