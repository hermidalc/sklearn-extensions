"""
The :mod:`sklearn.model_selection` module.
"""

from ._split import StratifiedGroupShuffleSplit

from ._validation import cross_val_score
from ._validation import cross_val_predict
from ._validation import cross_validate
from ._validation import learning_curve
from ._validation import permutation_test_score
from ._validation import validation_curve

from ._search import ExtendedGridSearchCV
from ._search import ExtendedRandomizedSearchCV
from ._search import ExtendedParameterGrid
from ._search import ExtendedParameterSampler
from ._search import fit_grid_point


__all__ = ['ExtendedGridSearchCV',
           'ExtendedParameterGrid',
           'ExtendedParameterSampler',
           'ExtendedRandomizedSearchCV',
           'StratifiedGroupShuffleSplit',
           'cross_val_predict',
           'cross_val_score',
           'cross_validate',
           'fit_grid_point',
           'learning_curve',
           'permutation_test_score',
           'validation_curve']
