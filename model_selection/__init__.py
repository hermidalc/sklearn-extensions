"""
The :mod:`sklearn.model_selection` module.
"""

from ._split import (
    StratifiedGroupKFold, RepeatedStratifiedGroupKFold,
    StratifiedSampleFromGroupKFold, RepeatedStratifiedSampleFromGroupKFold,
    StratifiedGroupShuffleSplit, StratifiedSampleFromGroupShuffleSplit)

from ._validation import cross_val_score
from ._validation import cross_val_predict
from ._validation import cross_validate
from ._validation import learning_curve
from ._validation import permutation_test_score
from ._validation import validation_curve

from ._search import ExtendedGridSearchCV
from ._search import ExtendedRandomizedSearchCV
from ._search import fit_grid_point


__all__ = ['ExtendedGridSearchCV',
           'ExtendedRandomizedSearchCV',
           'StratifiedGroupKFold',
           'StratifiedSampleFromGroupKFold',
           'StratifiedGroupShuffleSplit',
           'StratifiedSampleFromGroupShuffleSplit',
           'RepeatedStratifiedGroupKFold',
           'RepeatedStratifiedSampleFromGroupKFold',
           'cross_val_predict',
           'cross_val_score',
           'cross_validate',
           'fit_grid_point',
           'learning_curve',
           'permutation_test_score',
           'validation_curve']
